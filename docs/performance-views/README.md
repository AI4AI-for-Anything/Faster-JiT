# Performance figures

Regenerate all figures with `python docs/performance-views/render.py` (matplotlib only, no GPU).

- `01-benchmark.svg`: the final full check of commit `19b7549`. Baseline runs 100.06 and 99.98 ms per step (mean 100.02 ms), candidate 64.04 ms (median of three 30-step windows: 64.04 / 69.61 / 63.96 ms), 1.5618×. Two H200-class GPUs, 128 images per GPU.
- `02-history.svg`: the twelve adopted configurations, each from its own full check with freshly measured baseline runs ([history.json](history.json)). Open red markers were judged by the check before the EMA semantic check was added at round 26; grey markers are configurations that contained the compiled EMA update, which failed that check and was replaced at round 26. Points are cumulative configurations, not isolated ablations.
- `03-time-budget.svg`: kernel time per GPU per step of the final configuration from an Nsight Systems profile ([time-budget.json](time-budget.json)), compared with the full-check step time and the 2× and 3× targets. Kernel sums can overlap and do not include host gaps, so this figure is an estimate rather than a check. The BF16 matmul peak (783 TFLOP/s) was measured with a large `torch.matmul` microbenchmark on this GPU model.
- `../change-views/module-map.svg`: the changed files grouped by area.
