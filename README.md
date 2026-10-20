# Faster-JiT

[English](README.md) | [简体中文](README.zh-CN.md)

**1.56× faster JiT-B/16 training steps on Hopper GPUs, optimized by AI agents.**

![Faster-JiT: 1.56× training-step speedup on two H200-class GPUs](docs/performance-views/01-benchmark.svg)

| Implementation | Time / training step (ms) ↓ | Speedup ↑ |
|---|---:|---:|
| Reference | 100.02 | 1.00× |
| **Faster-JiT** | **64.04** | **1.56×** |

Measured with a full check on **2 H200-class GPUs** (Hopper): JiT-B/16 on ImageNet 256×256, the upstream recipe with **128 images per GPU** (global batch 256), BF16 autocast with FP32 master weights, AdamW states and both EMAs, **no FP8**. The timer runs the repository's real `main_jit.py`; reference, candidate and reference run in independent processes, and the candidate then passed the check's numerical gates. The exact ratio is **1.5618×**. The goal was 3×; neither 3× nor the later 2× fallback was reached.

## Quick Start

Prepare the [environment and ImageNet](docs/training.md). The optimizations are environment switches on the upstream trainer; the launcher sources them and runs the upstream JiT-B/16 command (8 GPUs by default, 128 images per GPU):

```bash
export IMAGENET_PATH=/path/to/imagenet   # contains train/
export OUTPUT_DIR=/path/to/output
bash scripts/train_faster_jit_b16.sh
```

Equivalently, `source scripts/configs/faster.env` and use the upstream `torchrun ... main_jit.py` command. Tested with **PyTorch 2.8.0 + CUDA 12.8 on Hopper GPUs**. With no switch set, the code runs the upstream path.

## Agent Optimization History

Each point is a full check on two H200-class GPUs with 128 images per GPU.

![JiT optimization history, from 1.25× to 1.56×](docs/performance-views/02-history.svg)

## What changed

- **Attention:** FlexAttention replaces the upstream math attention (FP32 `QK^T` and softmax); Q and K are cast to BF16 after RoPE.
- **Compilation:** the whole JiT forward and the outer denoiser (noise sampling and loss) are compiled with CUDA graphs, and Inductor tunes the pointwise and reduction kernels.
- **Fused Triton kernels:** QK RMSNorm + RoPE with a fixed-order (atomic-free) backward, and SwiGLU with its bias gradient reduced from per-tile partials.
- **Activations and optimizer:** the in-context tokens stay BF16 instead of promoting the residual stream to FP32, the early residual stream is made contiguous, AdamW runs fused, and both EMAs are updated with foreach operations.

![Optimization map: attention, compilation, fused kernels, activations and optimizer](docs/change-views/module-map.svg)

See the [switch table](docs/training.md#switches) and the [measured configuration](scripts/configs/faster.env). The changes alter floating-point rounding; they are not lossless.

## Validation

The final configuration passed the full check: structural and exact-item checks (recipe, data, random-number fingerprints, call counts, learning rates), an isolated EMA check (4 steps × 366 tensors, bitwise identical to the reference update) and, because 1,574 of 8,760 recorded tensors were bitwise identical rather than all, a statistical check. That check ran 300 post-warm-up training steps with three paired seeds; the tolerance δ is twice the reference's own seed-to-seed standard deviation of each loss window.

| Loss window (steps) | Mean difference / reference σ | Within δ = 2σ |
|---|---:|:---:|
| 50–100 | −0.51 | ✓ |
| 100–150 | +0.02 | ✓ |
| 150–200 | −0.05 | ✓ |
| 200–250 | −0.75 | ✓ |
| 250–300 | +0.04 | ✓ |

See the [validation method](docs/training.md#validation). Full 600-epoch training and FID have not been evaluated; the **1.56×** result measures training-step speed.

## Why it stopped at 1.56×

GEMM kernels take about 25–26 ms of each 64 ms step and already run at 90–93% of the BF16 matmul throughput measured on this GPU. A 3× step (33.3 ms) would leave about 7–10 ms for everything else, which takes about 34 ms of kernel time now; FP8 was excluded. The optimization run was stopped after 67 rounds.

![Time budget of the final configuration](docs/performance-views/03-time-budget.svg)

See [why it stopped here](docs/training.md#why-it-stopped-here) and the directions that were tried and not adopted.

## More AI4AI-for-Anything projects

[Faster-RAE](https://github.com/AI4AI-for-Anything/Faster-RAE) · [Faster-LLaVA](https://github.com/AI4AI-for-Anything/Faster-LLaVA) · [Faster-DMD](https://github.com/AI4AI-for-Anything/Faster-DMD) · [Faster-SelfForcing](https://github.com/AI4AI-for-Anything/Faster-SelfForcing)

## Upstream JiT documentation

Based on [LTH14/JiT](https://github.com/LTH14/JiT) at `cbc743a`, by Tianhong Li and Kaiming He. [Paper](https://arxiv.org/abs/2511.13720) · [MIT license](LICENSE). The original README is kept in [README.upstream.md](README.upstream.md) and below.

<details>
<summary>Original JiT README: dataset, installation, training and evaluation</summary>

## Just image Transformer (JiT) for Pixel-space Diffusion

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2511.13720-b31b1b.svg)](https://arxiv.org/abs/2511.13720)&nbsp;

<p align="center">
  <img src="demo/visual.jpg" width="100%">
</p>


This is a PyTorch/GPU re-implementation of the paper [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720):

```
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```

JiT adopts a minimalist and self-contained design for pixel-level high-resolution image diffusion. 
The original implementation was in JAX+TPU. This re-implementation is in PyTorch+GPU.

<p align="center">
  <img src="demo/jit.jpg" width="40%">
</p>

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Installation

Download the code:
```
git clone https://github.com/LTH14/JiT.git
cd JiT
```

A suitable [conda](https://conda.io/) environment named `jit` can be created and activated with:

```
conda env create -f environment.yaml
conda activate jit
```

If you get ```undefined symbol: iJIT_NotifyEvent``` when importing ```torch```, simply
```
pip uninstall torch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
Check this [issue](https://github.com/conda/conda/issues/13812#issuecomment-2071445372) for more details.

### Training
The below training scripts have been tested on 8 H200 GPUs.

Example script for training JiT-B/16 on ImageNet 256x256 for 600 epochs:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval
```

Example script for training JiT-B/32 on ImageNet 512x512 for 600 epochs:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 512 --noise_scale 2.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval
```

Example script for training JiT-H/16 on ImageNet 256x256 for 600 epochs:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-H/16 \
--proj_dropout 0.2 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval
```

### Evaluation

PyTorch pre-trained models are available [here](https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/AK218ZNvXKSv74igVvht4PQ?rlkey=14gjrblmljewpl6ygxzlr3njm&st=ffkl77al&dl=0).

Evaluate pre-trained JiT-B:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 (or JiT-B/32) \
--img_size 256 (or 512) --noise_scale 1.0 (or 2.0) \
--gen_bsz 256 --num_images 50000 --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

Evaluate pre-trained JiT-L:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-L/16 (or JiT-L/32) \
--img_size 256 (or 512) --noise_scale 1.0 (or 2.0) \
--gen_bsz 256 --num_images 50000 --cfg 2.4 (or 2.5) --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

Evaluate pre-trained JiT-H:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-H/16 (or JiT-H/32) \
--img_size 256 (or 512) --noise_scale 1.0 (or 2.0) \
--gen_bsz 256 --num_images 50000 --cfg 2.2 (or 2.3) --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

We use a customized [```torch-fidelity```](https://github.com/LTH14/torch-fidelity)
to evaluate FID and IS against a reference image folder or statistics. You can use ```prepare_ref.py```
to prepare the reference image folder, or directly use our pre-computed reference stats
under ```fid_stats```.

### Acknowledgements

We thank Google TPU Research Cloud (TRC) for granting us access to TPUs, and the MIT
ORCD Seed Fund Grants for supporting GPU resources.

### Contact

If you have any questions, feel free to contact me through email (tianhong@mit.edu). Enjoy!

</details>
