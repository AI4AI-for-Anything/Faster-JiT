"""D=64 bf16 QK RMSNorm + interleaved RoPE, with deterministic backward.

The two bf16 forward rounding boundaries and the bf16 norm-output gradient
boundary are explicit. Reductions are fp32; weight gradients use disjoint
partials followed by a fixed tree, never atomics. This is not a bitwise claim.
The caller owns the enable flag and fallback for unsupported inputs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel(X, WQ, WK, C, S, Q, K, V,
                    N: tl.constexpr, H: tl.constexpr, ROWS: tl.constexpr,
                    XB: tl.constexpr, XN: tl.constexpr,
                    CS0: tl.constexpr, CS1: tl.constexpr,
                    SS0: tl.constexpr, SS1: tl.constexpr,
                    WQS: tl.constexpr, WKS: tl.constexpr,
                    EPS: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    d = tl.arange(0, 64)
    h = row % H
    n = (row // H) % N
    b = row // (H * N)
    valid = row[:, None] < ROWS
    src = b[:, None] * XB + n[:, None] * XN + h[:, None] * 64 + d[None, :]
    xq = tl.load(X + src, valid, 0).to(tl.float32)
    xk = tl.load(X + src + H * 64, valid, 0).to(tl.float32)
    xv = tl.load(X + src + 2 * H * 64, valid, 0)
    wq = tl.load(WQ + d * WQS)
    wk = tl.load(WK + d * WKS)
    rq = tl.rsqrt(tl.sum(xq * xq, 1) * (1.0 / 64) + EPS)
    rk = tl.rsqrt(tl.sum(xk * xk, 1) * (1.0 / 64) + EPS)
    nq = (wq[None, :] * (xq * rq[:, None])).to(tl.bfloat16).to(tl.float32)
    nk = (wk[None, :] * (xk * rk[:, None])).to(tl.bfloat16).to(tl.float32)
    pair = tl.broadcast_to((d ^ 1)[None, :], (R, 64))
    pq = tl.gather(nq, pair, 1)
    pk = tl.gather(nk, pair, 1)
    sign = tl.where(d % 2 == 0, -1.0, 1.0)
    co = tl.load(C + n[:, None] * CS0 + d[None, :] * CS1, valid, 0)
    si = tl.load(S + n[:, None] * SS0 + d[None, :] * SS1, valid, 0)
    oq = nq * co + (pq * sign[None, :]) * si
    ok = nk * co + (pk * sign[None, :]) * si
    dst = ((b[:, None] * H + h[:, None]) * N + n[:, None]) * 64 + d[None, :]
    tl.store(Q + dst, oq, valid)
    tl.store(K + dst, ok, valid)
    tl.store(V + dst, xv, valid)


@triton.jit
def _backward_kernel(X, WQ, WK, C, S, DQ, DK, DV, DX, PQ, PK,
                     N: tl.constexpr, H: tl.constexpr, ROWS: tl.constexpr,
                     XB: tl.constexpr, XN: tl.constexpr,
                     CS0: tl.constexpr, CS1: tl.constexpr,
                     SS0: tl.constexpr, SS1: tl.constexpr,
                     WQS: tl.constexpr, WKS: tl.constexpr,
                     Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr, Q3: tl.constexpr,
                     K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
                     V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr, V3: tl.constexpr,
                     EPS: tl.constexpr, R: tl.constexpr):
    group = tl.program_id(0)
    row = group * R + tl.arange(0, R)
    d = tl.arange(0, 64)
    h = row % H
    n = (row // H) % N
    b = row // (H * N)
    valid = row[:, None] < ROWS
    src = b[:, None] * XB + n[:, None] * XN + h[:, None] * 64 + d[None, :]
    xq = tl.load(X + src, valid, 0).to(tl.float32)
    xk = tl.load(X + src + H * 64, valid, 0).to(tl.float32)
    gq_base = b[:, None] * Q0 + h[:, None] * Q1 + n[:, None] * Q2
    gk_base = b[:, None] * K0 + h[:, None] * K1 + n[:, None] * K2
    gv_base = b[:, None] * V0 + h[:, None] * V1 + n[:, None] * V2
    gq = tl.load(DQ + gq_base + d[None, :] * Q3, valid, 0).to(tl.float32)
    gk = tl.load(DK + gk_base + d[None, :] * K3, valid, 0).to(tl.float32)
    gqp = tl.load(DQ + gq_base + (d ^ 1)[None, :] * Q3, valid, 0).to(tl.float32)
    gkp = tl.load(DK + gk_base + (d ^ 1)[None, :] * K3, valid, 0).to(tl.float32)
    gv = tl.load(DV + gv_base + d[None, :] * V3, valid, 0)
    co = tl.load(C + n[:, None] * CS0 + d[None, :] * CS1, valid, 0)
    sp = tl.load(S + n[:, None] * SS0 + (d ^ 1)[None, :] * SS1, valid, 0)
    # Transpose of (-odd, even): (+odd, -even). Use partner sin,
    # so correctness does not rely on equal cosine/sine within each pair.
    sign = tl.where(d % 2 == 0, 1.0, -1.0)
    # Each multiply backward crosses its own fp32->bf16 boundary before
    # the two contributions accumulate into the bf16 norm-output gradient.
    # Rounding only after their fp32 sum does not implement eager autograd.
    direct_q = (gq * co).to(tl.bfloat16).to(tl.float32)
    direct_k = (gk * co).to(tl.bfloat16).to(tl.float32)
    rotated_q = ((gqp * sp) * sign[None, :]).to(tl.bfloat16).to(tl.float32)
    rotated_k = ((gkp * sp) * sign[None, :]).to(tl.bfloat16).to(tl.float32)
    dyq = (direct_q + rotated_q).to(tl.bfloat16).to(tl.float32)
    dyk = (direct_k + rotated_k).to(tl.bfloat16).to(tl.float32)
    wq = tl.load(WQ + d * WQS)
    wk = tl.load(WK + d * WKS)
    rq = tl.rsqrt(tl.sum(xq * xq, 1) * (1.0 / 64) + EPS)
    rk = tl.rsqrt(tl.sum(xk * xk, 1) * (1.0 / 64) + EPS)
    tq = dyq * wq[None, :]
    tk = dyk * wk[None, :]
    dotq = tl.sum(tq * xq, 1)
    dotk = tl.sum(tk * xk, 1)
    # Preserve the original autograd chain's multiplication order rather
    # than rewriting the derivative through a normalized-x identity.
    aq = ((dotq * -0.5) * ((rq * rq) * rq)) * (1.0 / 64)
    ak = ((dotk * -0.5) * ((rk * rk) * rk)) * (1.0 / 64)
    dxq = tq * rq[:, None] + aq[:, None] * (xq * 2.0)
    dxk = tk * rk[:, None] + ak[:, None] * (xk * 2.0)
    dst = row[:, None] // H * (3 * H * 64) + h[:, None] * 64 + d[None, :]
    tl.store(DX + dst, dxq, valid)
    tl.store(DX + dst + H * 64, dxk, valid)
    tl.store(DX + dst + 2 * H * 64, gv, valid)
    dwq = tl.sum(dyq * (xq * rq[:, None]), 0)
    dwk = tl.sum(dyk * (xk * rk[:, None]), 0)
    tl.store(PQ + group * 64 + d, dwq)
    tl.store(PK + group * 64 + d, dwk)


@triton.jit
def _weight_reduce(PQ, PK, WQ, WK, GROUPS: tl.constexpr, REDUCE: tl.constexpr):
    d = tl.program_id(0)
    g = tl.arange(0, REDUCE)
    q = tl.load(PQ + g * 64 + d, g < GROUPS, 0)
    k = tl.load(PK + g * 64 + d, g < GROUPS, 0)
    tl.store(WQ + d, tl.sum(q, 0))
    tl.store(WK + d, tl.sum(k, 0))


def _validate(qkv, weight_q, weight_k, cos, sin):
    if qkv.ndim != 3 or qkv.dtype != torch.bfloat16 or not qkv.is_cuda:
        raise ValueError("qknorm_rope requires CUDA bf16 [B,N,3*H*64]")
    if qkv.stride(-1) != 1 or qkv.shape[-1] % 192 or min(qkv.shape) <= 0:
        raise ValueError("qknorm_rope requires nonempty D=64 input with last stride 1")
    for weight in (weight_q, weight_k):
        if weight.shape != (64,) or weight.dtype != torch.float32 or weight.device != qkv.device:
            raise ValueError("qknorm_rope requires fp32 norm weights [64] on input device")
    for trig in (cos, sin):
        if trig.shape != (qkv.shape[1], 64) or trig.dtype != torch.float32 or trig.device != qkv.device:
            raise ValueError("qknorm_rope requires fp32 cos/sin [N,64] on input device")
        if trig.requires_grad:
            raise ValueError("qknorm_rope does not differentiate fixed cos/sin buffers")


@torch.library.custom_op("jit_fused::qknorm_rope", mutates_args=())
def qknorm_rope(qkv: torch.Tensor, weight_q: torch.Tensor, weight_k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor, eps: float
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate(qkv, weight_q, weight_k, cos, sin)
    b, n, width = qkv.shape
    h = width // 192
    shape = (b, h, n, 64)
    q = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    _forward_kernel[(triton.cdiv(b * n * h, 16),)](
        qkv, weight_q, weight_k, cos, sin, q, k, v,
        n, h, b * n * h, *qkv.stride()[:2], *cos.stride(), *sin.stride(),
        weight_q.stride(0), weight_k.stride(0), eps, 16,
        num_warps=4, enable_fp_fusion=False)
    return q, k, v


@qknorm_rope.register_fake
def _forward_fake(qkv, weight_q, weight_k, cos, sin, eps):
    shape = (qkv.shape[0], qkv.shape[-1] // 192, qkv.shape[1], 64)
    return tuple(qkv.new_empty(shape) for _ in range(3))


@torch.library.custom_op("jit_fused::qknorm_rope_backward", mutates_args=())
def _backward_op(qkv: torch.Tensor, weight_q: torch.Tensor, weight_k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor,
                 dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor, eps: float
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, n, width = qkv.shape
    h = width // 192
    rows = b * n * h
    # Q/K and their RoPE/RMS backward intermediates are live together.
    # A smaller row tile reduces per-thread register demand at eight warps,
    # aiming to expose more concurrent memory requests. Keep this fixed;
    # no timing-dependent autotuning or nondeterministic reduction is used.
    backward_rows = 16
    groups = triton.cdiv(rows, backward_rows)
    dx = torch.empty(qkv.shape, device=qkv.device, dtype=qkv.dtype)
    dwq = torch.empty_like(weight_q, memory_format=torch.contiguous_format)
    dwk = torch.empty_like(weight_k, memory_format=torch.contiguous_format)
    pq = torch.empty((groups, 64), device=qkv.device, dtype=torch.float32)
    pk = torch.empty_like(pq)
    _backward_kernel[(groups,)](
        qkv, weight_q, weight_k, cos, sin, dq, dk, dv, dx, pq, pk,
        n, h, rows, *qkv.stride()[:2], *cos.stride(), *sin.stride(),
        weight_q.stride(0), weight_k.stride(0),
        *dq.stride(), *dk.stride(), *dv.stride(), eps, backward_rows,
        num_warps=8, enable_fp_fusion=False)
    _weight_reduce[(64,)](pq, pk, dwq, dwk, groups,
                         triton.next_power_of_2(groups), num_warps=8,
                         enable_fp_fusion=False)
    return dx, dwq, dwk


@_backward_op.register_fake
def _backward_fake(qkv, weight_q, weight_k, cos, sin, dq, dk, dv, eps):
    return (qkv.new_empty(qkv.shape), weight_q.new_empty(weight_q.shape),
            weight_k.new_empty(weight_k.shape))


def _setup_context(ctx, inputs, output):
    qkv, weight_q, weight_k, cos, sin, eps = inputs
    ctx.save_for_backward(qkv, weight_q, weight_k, cos, sin)
    ctx.eps = eps


def _autograd_backward(ctx, dq, dk, dv):
    qkv, weight_q, weight_k, cos, sin = ctx.saved_tensors
    shape = (qkv.shape[0], qkv.shape[-1] // 192, qkv.shape[1], 64)
    dq = qkv.new_zeros(shape) if dq is None else dq
    dk = qkv.new_zeros(shape) if dk is None else dk
    dv = qkv.new_zeros(shape) if dv is None else dv
    dx, dwq, dwk = _backward_op(qkv, weight_q, weight_k, cos, sin,
                               dq, dk, dv, ctx.eps)
    return dx, dwq, dwk, None, None, None


qknorm_rope.register_autograd(_autograd_backward, setup_context=_setup_context)
