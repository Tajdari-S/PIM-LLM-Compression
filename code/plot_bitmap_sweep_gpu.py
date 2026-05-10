#!/usr/bin/env python3
"""
plot_bitmap_sweep_gpu.py

GPU-parallel bitmap+mantissa-borrow sweep.
Each tensor is loaded once and all (W, r) combos run from GPU memory.
Window histograms are computed with scatter_add in uint8-friendly chunks.

Falls back to CPU numpy when CUDA is unavailable.
"""

import os, math
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None

# ── fonts ─────────────────────────────────────────────────────────────────────
for _ttf in [
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf",
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf",
]:
    if os.path.isfile(_ttf): _fm.fontManager.addfont(_ttf)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 13, "axes.labelsize": 13,
    "axes.titlesize": 14, "legend.fontsize": 11, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
})

MODELS = [
    "mistralai_Mistral-7B-v0.1",
    "microsoft_phi-2",
    "Qwen_Qwen2.5-7B",
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0",
]
SHORT = {
    "mistralai_Mistral-7B-v0.1":          "Mistral-7B",
    "microsoft_phi-2":                    "Phi-2",
    "Qwen_Qwen2.5-7B":                    "Qwen2.5-7B",
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
}
C_ORIG   = "#aaaaaa"
C_BITMAP = "#e15759"
R_COLORS = {2:"#1f77b4", 3:"#ff7f0e", 4:"#2ca02c", 5:"#d62728", 6:"#9467bd"}

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Budget: keep index tensor (int64) under this many bytes per chunk.
# idx tensor = CHUNK * W * 8 bytes → CHUNK = IDX_BUDGET / (W * 8)
# Also cap histogram rows: hist = CHUNK * 256 * 4 bytes.
IDX_BUDGET   = 256 * 1024 * 1024   # 256 MB
HIST_BUDGET  = 256 * 1024 * 1024   # 256 MB  → CHUNK ≤ 256MB/(256*4) = 256K rows


def _chunk_size(W: int) -> int:
    by_idx  = max(1, IDX_BUDGET  // (W * 8))
    by_hist = max(1, HIST_BUDGET // (256 * 4))
    return min(by_idx, by_hist)


# ── GPU kernel ─────────────────────────────────────────────────────────────────

def count_windows_gpu(full: torch.Tensor, span: int) -> tuple:
    """
    full : (B_total, W) uint8 tensor on GPU — exponent bytes (0-255).
    Returns (outlier_count, underflow_count).
    """
    B_total, W = full.shape
    if B_total == 0:
        return 0, 0
    maxL  = max(0, 255 - span)
    CHUNK = _chunk_size(W)
    dev   = full.device

    # Precompute index ranges for sliding-window coverage (shared across chunks)
    L_range = torch.arange(maxL + 1, device=dev, dtype=torch.long)   # (maxL+1,)
    hi_idx  = (L_range + span + 1).clamp(max=256)                     # (maxL+1,)

    total_out = total_und = 0

    for start in range(0, B_total, CHUNK):
        chunk = full[start : start + CHUNK]         # (B, W) uint8, view — no copy
        B = chunk.shape[0]

        # ── batched histogram ──────────────────────────────────────────────────
        # offset each row by b*256 so scatter_add indexes into the right row
        offsets = torch.arange(B, device=dev, dtype=torch.int64).unsqueeze(1) * 256
        indices = chunk.to(torch.int64) + offsets           # (B, W) int64
        del offsets

        hist_flat = torch.zeros(B * 256, dtype=torch.int32, device=dev)
        ones      = torch.ones (B * W,   dtype=torch.int32, device=dev)
        hist_flat.scatter_add_(0, indices.view(-1), ones)
        del indices, ones

        hist = hist_flat.view(B, 256)                       # (B, 256) int32
        del hist_flat

        # ── prefix sum ────────────────────────────────────────────────────────
        pre = torch.zeros(B, 257, dtype=torch.int64, device=dev)
        pre[:, 1:] = hist.to(torch.int64).cumsum(dim=1)
        del hist

        # ── best_L via sliding window of width (span+1) ───────────────────────
        cov   = pre[:, hi_idx] - pre[:, L_range]            # (B, maxL+1)
        best_L = cov.argmax(dim=1).to(torch.int32)          # (B,)
        del pre, cov

        # ── count ─────────────────────────────────────────────────────────────
        chunk_i = chunk.to(torch.int32)
        bL      = best_L.unsqueeze(1)                        # (B, 1) broadcast
        total_out += int((chunk_i > bL + span).sum())
        total_und += int((chunk_i < bL).sum())
        del chunk_i, bL, best_L

    del L_range, hi_idx
    return total_out, total_und


# ── CPU fallback ───────────────────────────────────────────────────────────────

def count_windows_cpu(exps_np: np.ndarray, W: int, span: int) -> tuple:
    """Pure-numpy fallback operating on uint8 exponent array."""
    N    = len(exps_np)
    maxL = max(0, 255 - span)
    num_full = N // W
    rem      = N % W
    out = und = 0
    for b in range(num_full + (1 if rem else 0)):
        sl = exps_np[b*W : b*W + (W if b < num_full else rem)].astype(np.int32)
        hist = np.bincount(sl, minlength=256)
        pre  = np.empty(257, np.int64); pre[0] = 0; np.cumsum(hist, out=pre[1:])
        cov  = pre[np.minimum(np.arange(maxL+1) + span + 1, 256)] - pre[:maxL+1]
        bL   = int(np.argmax(cov))
        out += int(np.sum(sl > bL + span))
        und += int(np.sum(sl < bL))
    return out, und


# ── weight loader ─────────────────────────────────────────────────────────────

def load_u16(model_path):
    files = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))
    ftype = "safetensors"
    if not files:
        files = sorted(f for f in os.listdir(model_path) if f.endswith(".bin"))
        ftype = "bin"
    for fname in files:
        fp = os.path.join(model_path, fname)
        td = load_file(fp) if ftype == "safetensors" else torch.load(fp, map_location="cpu")
        if not isinstance(td, dict): continue
        for name, t in td.items():
            if not isinstance(t, torch.Tensor) or t.numel() == 0: continue
            if t.dtype not in SUPPORTED_DTYPES: continue
            if t.dtype != torch.bfloat16: t = t.to(torch.bfloat16)
            yield name, t.view(torch.int16).numpy().view(np.uint16).ravel()


# ── sweep ─────────────────────────────────────────────────────────────────────

def sweep_model(model_path, window_sizes, r_values):
    """
    Load each tensor once; process all (W, r) from GPU memory.
    Returns {(W, r): {"savings": float, "outlier_pct": float, "underflow_pct": float}}
    """
    use_gpu = (DEVICE.type == "cuda")
    spans   = {r: (1 << r) - 1 for r in r_values}

    # Accumulators: [outlier, underflow, total_N, total_windows]
    acc = {(W, r): [0, 0, 0, 0] for W in window_sizes for r in r_values}

    n_tensors = 0
    for name, u16 in load_u16(model_path):
        N = len(u16)
        n_tensors += 1

        # Exponent bytes as uint8 (1 B/weight)
        exps_np = ((u16 >> 7) & 0xFF).astype(np.uint8)

        if use_gpu:
            exps_gpu = torch.from_numpy(exps_np).to(DEVICE)  # uint8, N bytes on GPU

        for W in window_sizes:
            if N < W:
                continue
            num_full = N // W
            rem      = N % W

            if use_gpu:
                full = exps_gpu[:num_full * W].view(num_full, W)  # view, no alloc

            for r in r_values:
                span = spans[r]
                maxL = max(0, 255 - span)

                if use_gpu:
                    out, und = count_windows_gpu(full, span)
                else:
                    out, und = count_windows_cpu(exps_np[:num_full * W], W, span)

                # Partial tail window (CPU — always small)
                if rem > 0:
                    tail = exps_np[num_full * W:].astype(np.int32)
                    hist = np.bincount(tail, minlength=256)
                    pre  = np.empty(257, np.int64); pre[0] = 0; np.cumsum(hist, out=pre[1:])
                    cov  = pre[np.minimum(np.arange(maxL+1) + span + 1, 256)] - pre[:maxL+1]
                    bL   = int(np.argmax(cov))
                    out += int(np.sum(tail > bL + span))
                    und += int(np.sum(tail < bL))

                acc[(W, r)][0] += out
                acc[(W, r)][1] += und
                acc[(W, r)][2] += N
                acc[(W, r)][3] += num_full + (1 if rem else 0)

        if use_gpu:
            del exps_gpu

    print(f"  Processed {n_tensors} tensors", flush=True)

    results = {}
    for (W, r), (outl, undf, total_N, total_win) in acc.items():
        if total_N == 0:
            continue
        baseline   = 16 * total_N
        compressed = total_win * 8 + total_N * 1 + total_N * r + total_N * 8
        savings    = (baseline - compressed) / baseline * 100
        results[(W, r)] = dict(
            savings       = savings,
            outlier_pct   = outl / total_N * 100,
            underflow_pct = undf / total_N * 100,
        )
        print(f"  W={W:5d} r={r}  saved={savings:.2f}%  "
              f"outlier={outl/total_N*100:.3f}%  underflow={undf/total_N*100:.3f}%",
              flush=True)
    return results


# ── Figure 1: PPL bar chart ────────────────────────────────────────────────────

def plot_ppl(ppl_dir, out_dir, dpi=150):
    ppl_orig = {}; ppl_comp = {}; pct_inc = {}
    for m in MODELS:
        fp = os.path.join(ppl_dir, f"{m}_rcap3_bitmap_ppl.txt")
        if not os.path.isfile(fp):
            continue
        with open(fp) as f:
            for line in f:
                if line.startswith("Baseline"):
                    ppl_orig[m] = float(line.split()[-1])
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        int(parts[0])
                        ppl_comp[m] = float(parts[3])
                        pct_inc[m]  = float(parts[5].rstrip("%"))
                    except ValueError:
                        pass

    avail = [m for m in MODELS if m in ppl_orig and m in ppl_comp]
    if not avail:
        print("No PPL data found — skipping Figure 1")
        return

    n  = len(avail); bw = 0.30; x = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - bw/2, [ppl_orig[m] for m in avail], bw,
           color=C_ORIG, edgecolor="white", label="Original BF16", zorder=3)
    ax.bar(x + bw/2, [ppl_comp[m] for m in avail], bw,
           color=C_BITMAP, edgecolor="white", label="Bitmap+Mant (r=3, W=4096)", zorder=3)
    for i, m in enumerate(avail):
        po, pc, pct = ppl_orig[m], ppl_comp[m], pct_inc.get(m, 0)
        ax.text(i + bw/2, pc + 0.02, f"+{pct:.2f}%",
                ha="center", fontsize=10, fontweight="bold", color=C_BITMAP)
        ax.text(i - bw/2, po + 0.02, f"{po:.3f}",
                ha="center", fontsize=9, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in avail], rotation=15, ha="right")
    ax.set_ylabel("WikiText-2 Perplexity (↓ better)")
    ax.set_title("Original vs Bitmap+Mantissa-Borrow  (r=3, W=4096)", fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    all_v = [ppl_orig[m] for m in avail] + [ppl_comp[m] for m in avail]
    ax.set_ylim(min(all_v) * 0.97, max(all_v) * 1.08)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "bitmap_ppl_r3_W4096.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.2)
    print(f"Saved: {p}")
    plt.close(fig)


# ── Figure 2: sweep ────────────────────────────────────────────────────────────

def plot_sweep(all_sweep, window_sizes, r_values, out_dir, dpi=150):
    avail = [m for m in MODELS if m in all_sweep]
    n = len(avail)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 9), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    for ci, m in enumerate(avail):
        res = all_sweep[m]
        axes[0, ci].set_title(SHORT[m], fontweight="bold", pad=6)
        for r in r_values:
            color = R_COLORS.get(r, "#333")
            Ws  = [W for W in window_sizes if (W, r) in res]
            if not Ws:
                continue
            sav = [res[(W, r)]["savings"]     for W in Ws]
            out = [res[(W, r)]["outlier_pct"] for W in Ws]
            axes[0, ci].plot(Ws, sav, color=color, marker="o", lw=2, label=f"r={r}")
            axes[1, ci].plot(Ws, out, color=color, marker="o", lw=2, label=f"r={r}")

        for ax, ylabel in zip(axes[:, ci], ["Savings vs BF16 (%)", "Outlier % (bitmap=1)"]):
            ax.set_xscale("log", base=2)
            ax.set_xticks(window_sizes)
            ax.set_xticklabels([str(w) for w in window_sizes], rotation=40, ha="right")
            ax.set_xlabel("Window size W")
            if ci == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
            ax.legend(fontsize=9, frameon=True)

    fig.suptitle("Bitmap+Mantissa-Borrow: Savings & Outlier % vs W and r",
                 fontsize=14, fontweight="bold")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "bitmap_sweep_savings_outliers.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.2)
    print(f"Saved: {p}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir",   default="models")
    ap.add_argument("--ppl-bitmap",   default="results_ppl_bitmap")
    ap.add_argument("--out",          default="results_summary")
    ap.add_argument("--window-sizes", default="16,32,64,128,256,512,1024,2048,4096")
    ap.add_argument("--r-values",     default="2,3,4,5,6")
    ap.add_argument("--dpi",          type=int, default=150)
    ap.add_argument("--model",        default=None, help="single model key")
    args = ap.parse_args()

    window_sizes = [int(x) for x in args.window_sizes.split(",")]
    r_values     = [int(x) for x in args.r_values.split(",")]

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(0)}  ({total:.1f} GB)")

    print("=== Figure 1: PPL comparison ===")
    plot_ppl(args.ppl_bitmap, args.out, args.dpi)

    print("\n=== Figure 2: savings + outlier sweep ===")
    models_to_run = MODELS if args.model is None else [args.model]
    avail = [m for m in models_to_run
             if os.path.isdir(os.path.join(args.models_dir, m))]

    all_sweep = {}
    for m in avail:
        print(f"\n--- {SHORT[m]} ---", flush=True)
        all_sweep[m] = sweep_model(
            os.path.join(args.models_dir, m), window_sizes, r_values)

    # Save results to text files so --plot-only works next time
    for m, res in all_sweep.items():
        fp = os.path.join(args.out, f"{m}_bitmap_sweep.txt")
        with open(fp, "w") as f:
            f.write(f"# model={m}\n")
            f.write(f"{'W':>6}  {'r':>2}  {'savings':>8}  {'outlier_pct':>12}  {'underflow_pct':>14}\n")
            for W in window_sizes:
                for r in r_values:
                    if (W, r) not in res:
                        continue
                    d = res[(W, r)]
                    f.write(f"{W:6d}  {r:2d}  {d['savings']:8.4f}"
                            f"  {d['outlier_pct']:12.6f}  {d['underflow_pct']:14.6f}\n")
        print(f"Results saved: {fp}")

    plot_sweep(all_sweep, window_sizes, r_values, args.out, args.dpi)
    print("\nAll done.")


if __name__ == "__main__":
    main()
