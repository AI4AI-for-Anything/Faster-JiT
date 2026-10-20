#!/usr/bin/env python3
"""Render the Faster-JiT README figures from archived measurements; no GPU work.

Inputs: history.json (full-check results) and time-budget.json (final-version profile).
Outputs: 01-benchmark.svg, 02-history.svg, 03-time-budget.svg and ../change-views/module-map.svg.
Usage: python docs/performance-views/render.py   (needs matplotlib)
"""
from pathlib import Path
import json
import math
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
RED = '#8C0000'
LIGHT = '#D7BDBF'
GREY = '#A99A9C'
MUTED = '#79696B'
INK = '#292122'
GRID = '#EADFE0'
CARD = '#FAF5F5'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'svg.hashsalt': 'faster-jit-crimson', 'font.size': 12})


def save(fig, path):
    target = path.with_suffix('.svg')
    fig.savefig(target, facecolor='white', metadata={'Date': None})
    target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)


def frame(ax, axis='y'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0, labelsize=11.5, colors=MUTED, pad=9)
    ax.grid(axis=axis, color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def load_history():
    data = json.loads((ROOT / 'history.json').read_text())
    for row in data['rows']:
        assert math.isclose(row['reference_ms'] / row['candidate_ms'], row['speedup'], abs_tol=6e-4), row
    return data


def hero():
    final = load_history()['rows'][-1]
    values = [final['reference_ms'], final['candidate_ms']]
    fig = plt.figure(figsize=(9, 4.5), facecolor='white')
    fig.text(.06, .88, 'Faster-JiT, optimized with an AI agent', color=RED, weight='bold', fontsize=19)
    fig.text(.06, .78, 'JiT-B/16 · ImageNet 256×256 · 2 H200-class GPUs · 128 images per GPU · BF16, no FP8',
             color=MUTED, fontsize=10.5)
    ax = fig.add_axes([.24, .24, .39, .45])
    ax.barh([1, 0], values, color=[LIGHT, RED], height=.24, zorder=3)
    ax.set_yticks([1, 0], ['Reference', 'Faster-JiT'])
    for y, value in zip([1, 0], values):
        ax.text(value + 2.5, y, f'{value:.2f} ms', va='center', color=INK, fontsize=11.5, weight='bold')
    ax.set_xlim(0, 135)
    ax.set_ylim(-.6, 1.6)
    ax.set_xticks([0, 25, 50, 75, 100])
    frame(ax, 'x')
    ax.tick_params(labelsize=10)
    ax.set_xlabel('Time per training step (ms) ↓', color=MUTED, labelpad=12, fontsize=11)
    fig.text(.85, .54, f"{final['speedup']:.2f}×", color=RED, fontsize=35, weight='bold', ha='center')
    fig.text(.85, .43, 'step speedup', color=MUTED, fontsize=12, ha='center')
    fig.text(.85, .36, 'full check: PASS', color=MUTED, fontsize=10, ha='center')
    fig.text(.06, .055, 'Reference → candidate → reference in independent processes; reference = mean of the two '
             'reference runs.', color=MUTED, fontsize=9.5)
    save(fig, ROOT / '01-benchmark')


def history():
    data = load_history()
    rows = data['rows']
    height = 9.55
    top = lambda inches: 1 - inches / height       # figure fraction from inches below the top edge
    fig = plt.figure(figsize=(9, height), facecolor='white')
    fig.text(.09, top(.36), 'Faster-JiT', fontsize=11, weight='bold', color=RED)
    fig.text(.09, top(.79), 'Agent Optimization History', fontsize=21.5, weight='bold', color=INK)
    fig.text(.94, top(.65), f"{rows[-1]['speedup']:.2f}×", fontsize=29, ha='right', weight='bold', color=RED)
    fig.text(.94, top(.88), 'step speedup', fontsize=10.5, ha='right', color=MUTED)
    fig.text(.09, top(1.31), f"Cumulative full-check speedup · 2 H200-class GPUs · 128 images per GPU · "
             f"{data['rounds_accepted']} of {data['rounds_total']} rounds accepted", fontsize=11, color=MUTED)
    ax = fig.add_axes([.09, top(4.25), .85, 2.67 / height])
    x = list(range(len(rows)))
    values = [row['speedup'] for row in rows]
    ax.plot(x, values, color=RED, linewidth=2.3, zorder=2)
    for xi, value, row in zip(x, values, rows):
        if row['compiled_ema']:
            style = dict(color=GREY, markerfacecolor='white')
        elif row['bench'] == 'original':
            style = dict(color=RED, markerfacecolor='white')
        else:
            style = dict(color=RED, markerfacecolor=RED)
        ax.plot([xi], [value], marker='o', markersize=7, zorder=3, markeredgewidth=1.8, **style)
    for i, value in enumerate(values):
        ax.annotate(f'{value:.2f}×', (x[i], value), xytext=(0, 11 if i % 2 == 0 else -23),
                    textcoords='offset points', ha='center', fontsize=11,
                    color=RED if i == len(rows) - 1 else INK, weight='bold')
    ax.set_xticks(x, [f"R{row['round']:02d}" for row in rows])
    ax.tick_params(axis='x', labelsize=10)
    ax.set_yticks([1.0, 1.2, 1.4, 1.6])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.1f}×'))
    ax.set_ylim(1.0, 1.7)
    ax.set_xlim(-.6, len(rows) - .4)
    frame(ax)
    legend = [Line2D([], [], marker='o', color=RED, markerfacecolor=RED, markersize=7, linestyle='none',
                     label='Full check, revised bench'),
              Line2D([], [], marker='o', color=RED, markerfacecolor='white', markersize=7, markeredgewidth=1.8,
                     linestyle='none', label='Full check, original bench'),
              Line2D([], [], marker='o', color=GREY, markerfacecolor='white', markersize=7, markeredgewidth=1.8,
                     linestyle='none', label='Included compiled EMA (withdrawn)')]
    ax.legend(handles=legend, loc='upper left', frameon=False, fontsize=9.5, labelcolor=INK,
              handletextpad=.3, borderaxespad=0)
    for i, row in enumerate(rows):
        bx, by = .09 + (i % 2) * .44, top(5.2 + (i // 2) * .62)
        fig.add_artist(FancyBboxPatch((bx, by - .32 / height), .41, .5 / height,
                       boxstyle='round,pad=0.004,rounding_size=0.007',
                       transform=fig.transFigure, facecolor=CARD, edgecolor='none'))
        fig.text(bx + .014, by - .07 / height, f"R{row['round']:02d}", fontsize=11.5, weight='bold',
                 color=GREY if row.get('withdrawn') else RED, va='center')
        label = textwrap.fill(row['label'], width=34, break_long_words=False, break_on_hyphens=False)
        fig.text(bx + .079, by - .07 / height, label, fontsize=11, color=INK, weight='bold', va='center')
    fig.text(.09, top(9.3), 'R13–R17 measured a compiled EMA update that failed the EMA check added at R26; R26 '
             'replaced it with a foreach\nEMA and kept the other changes. Target was 3×; the run was stopped at '
             'R67 without reaching 2×. Cumulative configurations,\nnot isolated ablations.',
             fontsize=9.3, color=MUTED, linespacing=1.4)
    save(fig, ROOT / '02-history')


def time_budget():
    data = json.loads((ROOT / 'time-budget.json').read_text())
    parts = data['kernel_ms_per_gpu_step']
    steps = data['step_ms']
    gemm = parts['GEMM']
    gemm_at_peak = data['gemm_flops_per_gpu_step'] / (data['bf16_matmul_peak_tflops'] * 1e12) * 1e3
    fig = plt.figure(figsize=(9, 5.6), facecolor='white')
    fig.text(.06, .91, 'Why Faster-JiT stops at 1.56×', color=INK, weight='bold', fontsize=19)
    fig.text(.06, .845, 'Final configuration · kernel time per GPU per step (profiler) · step targets from the '
             'full-check reference', color=MUTED, fontsize=10.2)
    ax = fig.add_axes([.17, .38, .79, .36])
    ax.set_ylim(-.6, 1.45)
    ax.set_xlim(0, 105)
    left = 0
    names = list(parts)
    for name in names:
        value = parts[name]
        ax.barh(1, value, left=left, height=.42, color=RED if name == 'GEMM' else LIGHT,
                edgecolor='white', linewidth=2, zorder=3)
        left += value
    ax.text(gemm / 2, 1, f'GEMM {gemm:.1f} ms', color='white', ha='center', va='center', fontsize=11,
            weight='bold', zorder=4)
    ax.text(left + 1.2, 1, f'{left:.1f} ms\nkernel time', color=INK, va='center', fontsize=10, zorder=6,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5), linespacing=1.3)
    ax.barh(0, gemm_at_peak, height=.42, color='white', edgecolor=RED, linewidth=1.6, hatch='///', zorder=3)
    ax.text(gemm_at_peak + 1.2, 0, f'{gemm_at_peak:.1f} ms: same GEMM FLOPs at the measured\nBF16 matmul peak '
            f"({data['bf16_matmul_peak_tflops']} TFLOP/s)", color=INK, va='center', fontsize=10, zorder=6,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5), linespacing=1.3)
    for value, label in [(steps['target_3x'], '3× target'), (steps['target_2x'], '2×'),
                         (steps['faster_jit'], 'measured step')]:
        ax.axvline(value, color=INK if label == 'measured step' else RED, linestyle=(0, (4, 3)), linewidth=1.2,
                   zorder=5)
        ax.text(value, 1.5, f'{label}\n{value:.1f} ms', ha='center', va='bottom', fontsize=9.5,
                color=INK if label == 'measured step' else RED)
    ax.set_yticks([1, 0], ['Faster-JiT', 'GEMM at peak'])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    frame(ax, 'x')
    ax.tick_params(labelsize=10.5)
    ax.set_xlabel('ms per GPU per training step', color=MUTED, labelpad=10, fontsize=10.5)
    others = [f'{name} {parts[name]:.1f}' for name in names if name != 'GEMM']
    fig.text(.06, .19, 'Light segments, left to right (ms): ' + ' · '.join(others[:3]) + '\n'
             + ' · '.join(others[3:]), fontsize=9.5, color=INK, linespacing=1.5)
    budget = steps['target_3x'] - gemm
    lo, hi = (data['gemm_flops_per_gpu_step'] / (ms * 1e9) / data['bf16_matmul_peak_tflops'] * 100
              for ms in sorted(data['gemm_ms_range_across_final_profiles'], reverse=True))
    fig.text(.06, .055, f'At 3×, everything except GEMM would have to fit in about {budget:.0f} ms per step '
             f'(≈{left - gemm:.0f} ms of kernel time now). GEMMs already run\nat {lo:.0f}–{hi:.0f}% of the '
             'measured BF16 matmul peak. Kernel sums overlap and exclude host gaps: an estimate, not a check.',
             fontsize=9.3, color=MUTED, linespacing=1.4)
    save(fig, ROOT / '03-time-budget')


def modulemap():
    fig = plt.figure(figsize=(9, 6.6), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(.055, .92, 'What changed in Faster-JiT', fontsize=23, color=INK, weight='bold')
    cards = [
        ('Attention', 'FlexAttention for dense BF16 attention\nQ/K cast to BF16 after RoPE',
         'model_jit.py\nJIT_SDPA_BACKEND=flex', '1 file'),
        ('Compilation & graphs', 'Whole JiT forward + outer denoiser compiled\nCUDA graphs · Inductor tuning',
         'model_jit.py · denoiser.py\nJIT_COMPILE_MODEL · JIT_COMPILE_DENOISER\nTORCHINDUCTOR_* tuning', '2 files'),
        ('Fused Triton kernels', 'QK RMSNorm + RoPE with fixed-order backward\nSwiGLU + bias-gradient partials',
         'util/fused_qknorm_rope.py\nutil/fused_swiglu_bias.py · model_jit.py', '3 files'),
        ('Activations & optimizer', 'BF16 in-context tokens · contiguous residual\nFused AdamW · foreach EMA',
         'model_jit.py · engine_jit.py\ndenoiser.py', '3 files'),
    ]
    for i, (title, description, paths, count) in enumerate(cards):
        x, y = .055 + (i % 2) * .455, .49 - (i // 2) * .39
        ax.add_patch(FancyBboxPatch((x, y), .425, .34,
                     boxstyle='round,pad=0.009,rounding_size=0.012', facecolor=CARD, edgecolor=GRID))
        ax.plot([x, x + .425], [y + .34, y + .34], color=RED, linewidth=2, solid_capstyle='round')
        ax.text(x + .014, y + .30, f'{i + 1:02d}', fontsize=11, color=RED, weight='bold')
        ax.text(x + .41, y + .30, count, fontsize=10, color=MUTED, ha='right')
        ax.text(x + .014, y + .254, title, fontsize=13, color=INK, weight='bold')
        ax.text(x + .014, y + .198, description, fontsize=10.3, color=INK, va='top', linespacing=1.6)
        ax.text(x + .014, y + .035, paths, fontsize=9.5, color=MUTED, linespacing=1.5)
    save(fig, ROOT.parent / 'change-views/module-map')


if __name__ == '__main__':
    hero()
    history()
    time_budget()
    modulemap()
