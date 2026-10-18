"""SwiGLU backward plus deterministic bias partial reduction (conservative N2).

The private activation op is valid ONLY after BiasGradientLinear's detached-bias
projection. Its bias argument reconnects that projection's bias gradient; it is
not an independent mathematical activation function. FP32 bias reduction sums
BF16-rounded dx12, matching the compiled production reduction's input dtype.
Reduction grouping changes; no bitwise claim. No atomics or runtime tuning.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


class BiasGradientLinear(torch.nn.Linear):
    def forward(self, x, fused_bias_grad=False):
        # The caller makes one shared guard decision for projection+activation.
        return F.linear(x, self.weight, self.bias.detach() if fused_bias_grad else self.bias)


def supported(x, linear):
    return (x.is_cuda and x.ndim == 3 and x.shape[0] == 128
            and x.shape[1] in (256, 288) and x.shape[2] == 768
            and x.dtype in (torch.float32, torch.bfloat16)
            and isinstance(linear, BiasGradientLinear)
            and linear.weight.shape == (4096, 768)
            and linear.weight.dtype == torch.float32 and linear.bias is not None
            and linear.bias.dtype == torch.float32 and linear.bias.shape == (4096,)
            and linear.weight.device == x.device and linear.bias.device == x.device
            and torch.is_autocast_enabled('cuda')
            and torch.get_autocast_dtype('cuda') == torch.bfloat16)


@triton.jit
def _forward(X,Y,N:tl.constexpr,H:tl.constexpr,T:tl.constexpr,
             XB:tl.constexpr,XN:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=i//H
    col=i%H
    src=(row//N)*XB+(row%N)*XN+col
    a=tl.load(X+src,i<T*H,0).to(tl.float32)
    b=tl.load(X+src+H,i<T*H,0).to(tl.float32)
    tl.store(Y+i,(a*tl.sigmoid(a))*b,i<T*H)


@triton.jit
def _backward_partial(X,G,DX,P,N:tl.constexpr,H:tl.constexpr,T:tl.constexpr,
                      XB:tl.constexpr,XN:tl.constexpr,
                      GB:tl.constexpr,GN:tl.constexpr,GC:tl.constexpr,
                      R:tl.constexpr,C:tl.constexpr):
    row=tl.program_id(0)*R+tl.arange(0,R)
    col=tl.program_id(1)*C+tl.arange(0,C)
    src=(row[:,None]//N)*XB+(row[:,None]%N)*XN+col[None,:]
    mask=(row[:,None]<T)&(col[None,:]<H)
    a=tl.load(X+src,mask,0).to(tl.float32)
    b=tl.load(X+src+H,mask,0).to(tl.float32)
    g=tl.load(G+(row[:,None]//N)*GB+(row[:,None]%N)*GN+col[None,:]*GC,mask,0).to(tl.float32)
    s=tl.sigmoid(a)
    # BF16 materialization is shared by GEMM input and bias accumulation.
    da=((g*b)*(s*(1.+a*(1.-s)))).to(tl.bfloat16)
    db=(g*(a*s)).to(tl.bfloat16)
    dst=row[:,None]*(2*H)+col[None,:]
    tl.store(DX+dst,da,mask)
    tl.store(DX+dst+H,db,mask)
    pa=tl.sum(tl.where(mask,da.to(tl.float32),0.),axis=0)
    pb=tl.sum(tl.where(mask,db.to(tl.float32),0.),axis=0)
    tl.store(P+tl.program_id(0)*(2*H)+col,pa,col<H)
    tl.store(P+tl.program_id(0)*(2*H)+col+H,pb,col<H)


@triton.jit
def _reduce_bias(P,DB,W:tl.constexpr,PARTS:tl.constexpr,R:tl.constexpr,C:tl.constexpr):
    row=tl.arange(0,R)
    col=tl.program_id(0)*C+tl.arange(0,C)
    values=tl.load(P+row[:,None]*W+col[None,:],(row[:,None]<PARTS)&(col[None,:]<W),0)
    tl.store(DB+col,tl.sum(values,axis=0),col<W)


def _validate(x):
    if (not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 3
            or min(x.shape)<=0 or x.shape[-1]%2 or x.stride(-1)!=1
            or any(s<0 for s in x.stride())):
        raise ValueError('SwiGLU bias fusion requires CUDA BF16 [B,N,2H], last stride 1')


@torch.library.custom_op('jit_swiglu_bias::activation',mutates_args=())
def _activation(x:torch.Tensor,bias:torch.Tensor)->torch.Tensor:
    _validate(x)
    if bias.dtype!=torch.float32 or bias.shape!=(x.shape[-1],) or bias.device!=x.device:
        raise ValueError('SwiGLU bias token must be original FP32 projection bias')
    b,n,w=x.shape
    y=torch.empty((b,n,w//2),device=x.device,dtype=x.dtype)
    _forward[(triton.cdiv(b*n*w//2,1024),)](x,y,n,w//2,b*n,*x.stride()[:2],1024,num_warps=4)
    return y


@_activation.register_fake
def _fake(x,bias):
    return x.new_empty((*x.shape[:2],x.shape[-1]//2))


@torch.library.custom_op('jit_swiglu_bias::backward',mutates_args=())
def _backward(x:torch.Tensor,g:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    _validate(x)
    b,n,w=x.shape
    if (g.shape!=(b,n,w//2) or g.dtype!=x.dtype or g.device!=x.device
            or any(s<0 for s in g.stride())):
        raise ValueError('SwiGLU bias fusion dy has invalid shape, dtype, device or strides')
    dx=torch.empty(x.shape,device=x.device,dtype=x.dtype)
    parts=triton.cdiv(b*n,64)
    p=torch.empty((parts,w),device=x.device,dtype=torch.float32)
    db=torch.empty((w,),device=x.device,dtype=torch.float32)
    _backward_partial[(parts,triton.cdiv(w//2,128))](
        x,g,dx,p,n,w//2,b*n,*x.stride()[:2],*g.stride(),64,128,num_warps=8)
    _reduce_bias[(triton.cdiv(w,16),)](p,db,w,parts,triton.next_power_of_2(parts),16,num_warps=8)
    return dx,db


@_backward.register_fake
def _backward_fake(x,g):
    return x.new_empty(x.shape),x.new_empty((x.shape[-1],),dtype=torch.float32)


def _setup(ctx,inputs,output):
    ctx.save_for_backward(inputs[0])


def _autograd(ctx,g):
    if g is None:
        return None,None
    return _backward(ctx.saved_tensors[0],g)


_activation.register_autograd(_autograd,setup_context=_setup)
