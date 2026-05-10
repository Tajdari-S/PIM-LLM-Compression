# Bitmap + Mantissa-Borrow BF16 Weight Compression

Lossless-ish compression of BF16 LLM weights by exploiting the narrow exponent distribution within local weight windows.

## Method

We evaluate the bitmap+mantissa-borrow compression scheme across four language models spanning a range of scales: Mistral-7B-v0.1, Qwen2.5-7B, Phi-2, and TinyLlama-1.1B. For each model, weights stored in BF16 are compressed by partitioning them into non-overlapping windows of size W and, within each window, selecting the base exponent L that maximizes the number of weights whose 8-bit BF16 exponent falls within the range [L, L+2^r−1]. Weights within this range are encoded with a 1-bit bitmap flag (0) and an r-bit exponent offset; weights outside the range (outliers) are flagged with a 1-bit bitmap (1) and their lower 3 mantissa bits repurposed to store the 3 least significant bits of the excess exponent. Compression ratio is measured as storage savings relative to uncompressed BF16, computed as (16N − (8W_count + N + rN + 8N)) / 16N, where W_count is the number of windows. Quality degradation is quantified via perplexity on the WikiText-2 test set using a sliding-window evaluation (context length 16,384, stride 512) for the fixed configuration r=3, W=4096. To characterize the scheme's behavior across the compression–quality trade-off space, we sweep the exponent range parameter r ∈ {2, 3, 4, 5, 6} and window size W ∈ {16, 32, 64, 128, 256, 512, 1024, 2048, 4096}, reporting savings percentage and the fraction of weights flagged as outliers (bitmap=1) for each (W, r) pair.

## Encoding layout (per weight, BF16 = 16 bits)

```
Normal weight  (bitmap=0):  [sign 1b][bitmap 1b][exp_offset r bits][mantissa 7b]  → r+9 bits stored
Outlier weight (bitmap=1):  [sign 1b][bitmap 1b][exp_low3 3b][mantissa_high4 4b] → 9 bits + base overhead
Window header:               8 bits (base exponent L)
```

Savings formula: `(16N − (W_count×8 + N×1 + N×r + N×8)) / (16N) × 100`

## Models evaluated

| Model | Parameters |
|---|---|
| Mistral-7B-v0.1 | 7.24B |
| Qwen2.5-7B | 7.6B |
| Phi-2 | 2.78B |
| TinyLlama-1.1B-Chat-v1.0 | 1.10B |

## Results summary

At r=3, W=4096:
- Storage savings: ~25% across all models
- Outlier fraction: 0.13–0.21%
- WikiText-2 perplexity increase: +0.14% to +0.45%

## Repository structure

```
bitmap_mantissa_borrow/
├── README.md
├── code/
│   ├── eval_ppl_bitmap_mantissa.py      # WikiText-2 perplexity evaluation
│   ├── per_layer_bitmap_mantissa.py     # per-layer outlier/savings analysis
│   ├── plot_bitmap_sweep_parallel.py    # main sweep: GPU large models + CPU small models in parallel
│   ├── plot_bitmap_sweep_gpu.py         # GPU-only sweep (single process)
│   ├── plot_bitmap_ppl_and_sweep.py     # original CPU sweep (ctypes C kernel)
│   └── plot_bitmap_comparison.py        # comparison figures
├── results/
│   ├── ppl/                             # WikiText-2 perplexity results (r=3 and r=4)
│   ├── sweep/                           # savings + outlier % across all (W, r)
│   └── per_layer/                       # per-layer breakdown (W=4096, r=3 and r=4)
└── figures/
    ├── bitmap_ppl_r3_W4096.{png,pdf}            # Fig 1: PPL comparison bar chart
    ├── bitmap_sweep_savings_outliers.{png,pdf}   # Fig 2: savings & outlier % sweep
    ├── bitmap_per_layer_rcap3.{png,pdf}          # per-layer savings (r=3)
    ├── bitmap_per_layer_rcap4.{png,pdf}          # per-layer savings (r=4)
    ├── bitmap_ppl_comparison_rcap3.{png,pdf}     # PPL comparison (r=3)
    └── bitmap_ppl_comparison_rcap4.{png,pdf}     # PPL comparison (r=4)
```

## Usage

### 1. Evaluate perplexity (requires model weights)

```bash
python3 code/eval_ppl_bitmap_mantissa.py \
  --model-path models/mistralai_Mistral-7B-v0.1 \
  --r-cap 3 --window-size 4096 \
  --out results/ppl/
```

### 2. Run compression sweep (GPU + CPU parallel)

```bash
python3 code/plot_bitmap_sweep_parallel.py \
  --models-dir models \
  --ppl-bitmap results/ppl \
  --out results/sweep \
  --window-sizes 16,32,64,128,256,512,1024,2048,4096 \
  --r-values 2,3,4,5,6
```

Large models (≥2B weights) run on GPU; small models run on CPU via a compiled C kernel.
Each model runs in its own process. Wall time ≈ time for one 7B model (~4 min on A100).

### 3. Regenerate figures from saved results

```bash
python3 code/plot_bitmap_sweep_parallel.py \
  --models-dir models \
  --ppl-bitmap results/ppl \
  --out results/sweep \
  --plot-only
```

## Dependencies

```
torch >= 2.0
safetensors
matplotlib
numpy
gcc (for C kernel compilation)
```
