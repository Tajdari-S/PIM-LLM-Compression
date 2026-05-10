#!/usr/bin/env python3
"""
plot_bitmap_ppl_and_sweep.py

Figure 1: Perplexity bar chart — Original vs Bitmap+Mantissa-Borrow (r=3, W=4096)
Figure 2: Savings % and Outlier % sweep over r=2..6 and W=16..4096
"""

import ctypes, math, os, tempfile, time
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
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
MODEL_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
C_ORIG   = "#aaaaaa"
C_BITMAP = "#e15759"

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# ── C kernel for outlier counting ─────────────────────────────────────────────
_C_SRC = r"""
#include <stdint.h>
#include <string.h>

/*
 * For each window of W BF16 u16 weights, find optimal base L using
 * histogram, then count:
 *   outlier  = e > L + span   (bitmap bit set in bitmap+mantissa scheme)
 *   underflow = e < L          (clipped to L)
 * Accumulates into out[0]=outlier_count, out[1]=underflow_count, out[2]=total
 */
void count_outliers(const uint16_t *u16, int64_t N, int W, int64_t span,
                    int64_t *out) {
    out[0] = 0; out[1] = 0; out[2] = N;
    int64_t num_full = N / W;
    int64_t rem      = N % W;

    for (int64_t b = 0; b <= num_full; b++) {
        const uint16_t *row;
        int w;
        if (b < num_full) {
            row = u16 + b * W;
            w   = W;
        } else {
            if (rem == 0) break;
            row = u16 + num_full * W;
            w   = (int)rem;
        }

        /* histogram of exponent bits */
        int hist[256] = {0};
        for (int i = 0; i < w; i++)
            hist[(row[i] >> 7) & 0xFF]++;

        /* prefix sum */
        int pre[257]; pre[0] = 0;
        for (int v = 0; v < 256; v++) pre[v+1] = pre[v] + hist[v];

        /* find best L */
        int maxL = 255 - (int)span;
        if (maxL < 0) maxL = 0;
        int best_L = 0, best_cov = 0;
        for (int L = 0; L <= maxL; L++) {
            int hi = L + (int)span + 1;
            if (hi > 256) hi = 256;
            int cov = pre[hi] - pre[L];
            if (cov > best_cov) { best_cov = cov; best_L = L; }
        }

        /* count */
        for (int i = 0; i < w; i++) {
            int e = (row[i] >> 7) & 0xFF;
            if (e > best_L + (int)span) out[0]++;
            else if (e < best_L)        out[1]++;
        }
    }
}
"""

def _build_lib():
    td  = tempfile.mkdtemp()
    src = os.path.join(td, "out.c")
    so  = os.path.join(td, "out.so")
    with open(src, "w") as f: f.write(_C_SRC)
    if os.system(f"gcc -O3 -march=native -shared -fPIC -o {so} {src} 2>/dev/null"):
        return None
    lib = ctypes.CDLL(so)
    lib.count_outliers.restype  = None
    lib.count_outliers.argtypes = [
        ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64,
        ctypes.c_int, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    ]
    return lib

_LIB = _build_lib()


def load_u16(model_path):
    """Yield (name, u16_flat) for all BF16-convertible tensors."""
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


def sweep_model(model_path, window_sizes, r_values):
    """
    Returns {(W, r): {"savings": float, "outlier_pct": float, "underflow_pct": float}}
    """
    results = {}
    pu16 = ctypes.POINTER(ctypes.c_uint16)
    pi64 = ctypes.POINTER(ctypes.c_int64)

    for W in window_sizes:
        for r in r_values:
            span  = int((1 << r) - 1)
            total_outlier = np.zeros(3, dtype=np.int64)  # [outlier, underflow, total]
            total_windows = 0
            total_N       = 0

            for name, u16 in load_u16(model_path):
                N = len(u16)
                if N < W: continue
                buf = np.zeros(3, dtype=np.int64)
                u16c = np.ascontiguousarray(u16, dtype=np.uint16)
                _LIB.count_outliers(
                    u16c.ctypes.data_as(pu16),
                    ctypes.c_int64(N), ctypes.c_int(W), ctypes.c_int64(span),
                    buf.ctypes.data_as(pi64),
                )
                total_outlier += buf
                num_full = N // W
                total_windows += num_full + (1 if N % W else 0)
                total_N += N

            if total_N == 0: continue
            baseline   = 16 * total_N
            compressed = total_windows * 8 + total_N * 1 + total_N * r + total_N * 8
            savings    = (baseline - compressed) / baseline * 100
            results[(W, r)] = dict(
                savings       = savings,
                outlier_pct   = total_outlier[0] / total_N * 100,
                underflow_pct = total_outlier[1] / total_N * 100,
            )
            print(f"  W={W:5d} r={r}  saved={savings:.2f}%  "
                  f"outlier={total_outlier[0]/total_N*100:.3f}%  "
                  f"underflow={total_outlier[1]/total_N*100:.3f}%", flush=True)
    return results


# ── Figure 1: PPL comparison ───────────────────────────────────────────────────

def plot_ppl(ppl_dir_bitmap, out_dir, dpi=150):
    ppl_orig = {}
    ppl_comp = {}
    pct_inc  = {}

    for m in MODELS:
        fp = os.path.join(ppl_dir_bitmap, f"{m}_rcap3_bitmap_ppl.txt")
        if not os.path.isfile(fp): continue
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
    n  = len(avail)
    bw = 0.30
    x  = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_orig = ax.bar(x - bw/2, [ppl_orig[m] for m in avail], bw,
                       color=C_ORIG, edgecolor="white", label="Original BF16", zorder=3)
    bars_comp = ax.bar(x + bw/2, [ppl_comp[m] for m in avail], bw,
                       color=C_BITMAP, edgecolor="white", label="Bitmap+Mant (r=3, W=4096)", zorder=3)

    for i, m in enumerate(avail):
        po, pc = ppl_orig[m], ppl_comp[m]
        pct = pct_inc.get(m, 0)
        lbl = f"+{pct:.2f}%"
        ax.text(i + bw/2, pc + 0.02, lbl, ha="center", fontsize=10,
                fontweight="bold", color=C_BITMAP)
        ax.text(i - bw/2, po + 0.02, f"{po:.3f}", ha="center", fontsize=9, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in avail], rotation=15, ha="right")
    ax.set_ylabel("WikiText-2 Perplexity (↓ better)")
    ax.set_title("Original vs Bitmap+Mantissa-Borrow  (r=3, W=4096)", fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    # auto y-range with small head room
    all_vals = [ppl_orig[m] for m in avail] + [ppl_comp[m] for m in avail]
    ymin = min(all_vals) * 0.97
    ymax = max(all_vals) * 1.08
    ax.set_ylim(ymin, ymax)

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "bitmap_ppl_r3_W4096.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.2)
    print(f"Saved: {p}")
    plt.close(fig)


# ── Figure 2: sweep savings + outlier% ────────────────────────────────────────

R_COLORS = {2:"#1f77b4", 3:"#ff7f0e", 4:"#2ca02c", 5:"#d62728", 6:"#9467bd"}

def plot_sweep(all_sweep, window_sizes, r_values, out_dir, dpi=150):
    avail_models = [m for m in MODELS if m in all_sweep]
    n = len(avail_models)
    if n == 0: return

    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 9), constrained_layout=True)
    if n == 1: axes = axes.reshape(2, 1)

    for ci, m in enumerate(avail_models):
        res   = all_sweep[m]
        label = SHORT[m]
        axes[0, ci].set_title(label, fontweight="bold", pad=6)

        for r in r_values:
            color = R_COLORS.get(r, "#333")
            Ws  = [W for W in window_sizes if (W, r) in res]
            if not Ws: continue
            sav = [res[(W, r)]["savings"]     for W in Ws]
            out = [res[(W, r)]["outlier_pct"] for W in Ws]

            axes[0, ci].plot(Ws, sav, color=color, marker="o", lw=2, label=f"r={r}")
            axes[1, ci].plot(Ws, out, color=color, marker="o", lw=2, label=f"r={r}")

        for ri, (ax, ylabel) in enumerate(zip(axes[:, ci],
                ["Savings vs BF16 (%)", "Outlier % (bitmap=1)"])):
            ax.set_xscale("log", base=2)
            ax.set_xticks(window_sizes)
            ax.set_xticklabels([str(w) for w in window_sizes], rotation=40, ha="right")
            ax.set_xlabel("Window size W")
            if ci == 0: ax.set_ylabel(ylabel)
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
    args = ap.parse_args()

    window_sizes = [int(x) for x in args.window_sizes.split(",")]
    r_values     = [int(x) for x in args.r_values.split(",")]

    # Figure 1 — PPL (uses existing saved results)
    print("=== Figure 1: PPL comparison ===")
    plot_ppl(args.ppl_bitmap, args.out, args.dpi)

    # Figure 2 — sweep
    print("\n=== Figure 2: savings + outlier sweep ===")
    avail = [m for m in MODELS if os.path.isdir(os.path.join(args.models_dir, m))]
    all_sweep = {}
    for m in avail:
        print(f"\n--- {SHORT[m]} ---", flush=True)
        path = os.path.join(args.models_dir, m)
        all_sweep[m] = sweep_model(path, window_sizes, r_values)

    plot_sweep(all_sweep, window_sizes, r_values, args.out, args.dpi)
    print("\nAll done.")


if __name__ == "__main__":
    main()
