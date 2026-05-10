#!/usr/bin/env python3
"""
plot_bitmap_sweep_parallel.py

Fast parallel bitmap+mantissa-borrow sweep:
  - One OS process per model (multiprocessing spawn — safe for CUDA).
  - Large models (≥2B weights) → GPU via chunked scatter_add histograms.
  - Small models             → CPU via compiled C kernel (ctypes, gcc -O3).
  - Within each process a background thread pre-fetches tensors from disk
    so I/O and computation overlap.
  - Results saved as text so --plot-only can regenerate figures instantly.
"""

import ctypes, os, queue, tempfile, threading
import concurrent.futures
import multiprocessing as mp
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
    "mathtext.fontset": "stix", "font.size": 25, "axes.labelsize": 25,
    "axes.titlesize": 25, "legend.fontsize": 20, "xtick.labelsize": 20,
    "ytick.labelsize": 20, "axes.spines.top": False, "axes.spines.right": False,
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
# Models with ≥ this many weights use GPU; others use CPU.
GPU_THRESHOLD_WEIGHTS = 2_000_000_000

C_ORIG   = "#aaaaaa"
C_BITMAP = "#e15759"
R_COLORS = {2:"#1f77b4", 3:"#ff7f0e", 4:"#2ca02c", 5:"#d62728", 6:"#9467bd"}

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

IDX_BUDGET  = 256 * 1024 * 1024   # 256 MB — keeps idx (int64) under limit
HIST_BUDGET = 256 * 1024 * 1024   # 256 MB — keeps hist (int32) under limit


# ── C kernel for CPU path ─────────────────────────────────────────────────────

_C_SRC = r"""
#include <stdint.h>
#include <string.h>

void count_outliers_exps(const uint8_t *exps, int64_t N, int W, int64_t span,
                          int64_t *out_outlier, int64_t *out_underflow) {
    *out_outlier = 0; *out_underflow = 0;
    int64_t num_full = N / W;
    int64_t rem      = N % W;
    int maxL = 255 - (int)span; if (maxL < 0) maxL = 0;

    for (int64_t b = 0; b <= num_full; b++) {
        const uint8_t *row;
        int w;
        if (b < num_full) { row = exps + b*W; w = W; }
        else { if (rem == 0) break; row = exps + num_full*W; w = (int)rem; }

        int hist[256] = {0};
        for (int i = 0; i < w; i++) hist[row[i]]++;

        int pre[257]; pre[0] = 0;
        for (int v = 0; v < 256; v++) pre[v+1] = pre[v] + hist[v];

        int best_L = 0, best_cov = 0;
        for (int L = 0; L <= maxL; L++) {
            int hi = L + (int)span + 1;
            if (hi > 256) hi = 256;
            int cov = pre[hi] - pre[L];
            if (cov > best_cov) { best_cov = cov; best_L = L; }
        }

        for (int i = 0; i < w; i++) {
            int e = row[i];
            if (e > best_L + (int)span) (*out_outlier)++;
            else if (e < best_L)        (*out_underflow)++;
        }
    }
}
"""

def _build_c_lib():
    td  = tempfile.mkdtemp()
    src = os.path.join(td, "oc.c")
    so  = os.path.join(td, "oc.so")
    with open(src, "w") as f:
        f.write(_C_SRC)
    if os.system(f"gcc -O3 -march=native -shared -fPIC -o {so} {src} 2>/dev/null"):
        return None
    lib = ctypes.CDLL(so)
    lib.count_outliers_exps.restype  = None
    lib.count_outliers_exps.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64, ctypes.c_int, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
    ]
    return lib

_C_LIB = _build_c_lib()


# ── weight loader + background prefetch thread ────────────────────────────────

def _load_gen(model_path):
    """Yield (name, exps_uint8) from all weight tensors."""
    files = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))
    ftype = "safetensors"
    if not files:
        files = sorted(f for f in os.listdir(model_path) if f.endswith(".bin"))
        ftype = "bin"
    for fname in files:
        fp = os.path.join(model_path, fname)
        td = (load_file(fp) if ftype == "safetensors"
              else torch.load(fp, map_location="cpu"))
        if not isinstance(td, dict):
            continue
        for name, t in td.items():
            if not isinstance(t, torch.Tensor) or t.numel() == 0:
                continue
            if t.dtype not in SUPPORTED_DTYPES:
                continue
            if t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)
            u16  = t.view(torch.int16).numpy().view(np.uint16).ravel()
            exps = ((u16 >> 7) & 0xFF).astype(np.uint8)
            yield name, exps


def _prefetch(model_path, q, maxbuf=4):
    """Background thread: loads tensors and puts them on q."""
    for item in _load_gen(model_path):
        q.put(item)
    q.put(None)  # sentinel


# ── GPU counting ──────────────────────────────────────────────────────────────

def _gpu_chunk_size(W):
    return max(1, min(IDX_BUDGET // (W * 8),
                      HIST_BUDGET // (256 * 4)))


def count_windows_gpu(full: torch.Tensor, span: int, dev) -> tuple:
    """(B_total, W) uint8 tensor → (outlier_count, underflow_count)."""
    B_total, W = full.shape
    if B_total == 0:
        return 0, 0
    maxL  = max(0, 255 - span)
    CHUNK = _gpu_chunk_size(W)
    L_rng = torch.arange(maxL + 1, device=dev, dtype=torch.long)
    hi    = (L_rng + span + 1).clamp(max=256)
    total_out = total_und = 0

    for s in range(0, B_total, CHUNK):
        c  = full[s : s + CHUNK]          # (B, W) uint8 view
        B  = c.shape[0]
        off = torch.arange(B, device=dev, dtype=torch.int64).unsqueeze(1) * 256
        idx = c.to(torch.int64) + off
        del off
        hf  = torch.zeros(B * 256, dtype=torch.int32, device=dev)
        hf.scatter_add_(0, idx.view(-1),
                        torch.ones(B * W, dtype=torch.int32, device=dev))
        del idx
        pre = torch.zeros(B, 257, dtype=torch.int64, device=dev)
        pre[:, 1:] = hf.view(B, 256).to(torch.int64).cumsum(1)
        del hf
        cov    = pre[:, hi] - pre[:, L_rng]
        best_L = cov.argmax(1).to(torch.int32)
        del pre, cov
        ci  = c.to(torch.int32)
        bLv = best_L.unsqueeze(1)
        total_out += int((ci > bLv + span).sum())
        total_und += int((ci < bLv).sum())
        del ci, bLv, best_L

    return total_out, total_und


# ── CPU counting (C kernel) ───────────────────────────────────────────────────

def count_windows_cpu(exps_np: np.ndarray, W: int, span: int) -> tuple:
    N = len(exps_np)
    if N == 0:
        return 0, 0
    ec   = np.ascontiguousarray(exps_np, dtype=np.uint8)
    out  = ctypes.c_int64(0)
    und  = ctypes.c_int64(0)
    _C_LIB.count_outliers_exps(
        ec.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int64(N), ctypes.c_int(W), ctypes.c_int64(span),
        ctypes.byref(out), ctypes.byref(und),
    )
    return out.value, und.value


# ── per-model sweep (runs inside a worker process) ────────────────────────────

def _sweep_model_worker(model_key, model_path, window_sizes, r_values, device_str):
    """
    Called inside a spawned subprocess.  Loads tensors with a prefetch thread,
    processes each tensor on GPU or CPU, accumulates counts, returns results dict.
    """
    dev     = torch.device(device_str)
    use_gpu = (dev.type == "cuda" and torch.cuda.is_available())
    if use_gpu:
        torch.cuda.set_device(0)

    spans = {r: (1 << r) - 1 for r in r_values}
    acc   = {(W, r): [0, 0, 0, 0] for W in window_sizes for r in r_values}
    # acc[(W,r)] = [outlier, underflow, total_N, total_windows]

    q = queue.Queue(maxsize=6)
    th = threading.Thread(target=_prefetch, args=(model_path, q), daemon=True)
    th.start()

    n_tensors = 0
    while True:
        item = q.get()
        if item is None:
            break
        name, exps_np = item
        N = len(exps_np)
        n_tensors += 1

        if use_gpu:
            exps_gpu = torch.from_numpy(exps_np).to(dev)  # uint8

        for W in window_sizes:
            if N < W:
                continue
            num_full = N // W
            rem      = N % W

            if use_gpu:
                full = exps_gpu[:num_full * W].view(num_full, W)

            for r in r_values:
                span = spans[r]
                if use_gpu:
                    out, und = count_windows_gpu(full, span, dev)
                else:
                    out, und = count_windows_cpu(exps_np[:num_full * W], W, span)

                # Partial tail — always CPU (small)
                if rem > 0:
                    tail = exps_np[num_full * W:]
                    to, tu = count_windows_cpu(tail, rem, span)  # single window
                    out += to; und += tu

                acc[(W, r)][0] += out
                acc[(W, r)][1] += und
                acc[(W, r)][2] += N
                acc[(W, r)][3] += num_full + (1 if rem else 0)

        if use_gpu:
            del exps_gpu

    th.join()

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
        print(f"  [{SHORT.get(model_key, model_key):18s}]"
              f"  W={W:5d} r={r}  saved={savings:.2f}%"
              f"  outlier={outl/total_N*100:.3f}%", flush=True)
    return results


def _worker_entry(args):
    """Top-level function that can be pickled by multiprocessing."""
    model_key, model_path, window_sizes, r_values, device_str = args
    results = _sweep_model_worker(
        model_key, model_path, window_sizes, r_values, device_str)
    return model_key, results


# ── results persistence ───────────────────────────────────────────────────────

def save_results(all_sweep, out_dir, window_sizes, r_values):
    os.makedirs(out_dir, exist_ok=True)
    for m, res in all_sweep.items():
        fp = os.path.join(out_dir, f"{m}_bitmap_sweep.txt")
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
    print(f"Results saved to {out_dir}/")


def load_results(out_dir, models, window_sizes, r_values):
    all_sweep = {}
    for m in models:
        fp = os.path.join(out_dir, f"{m}_bitmap_sweep.txt")
        if not os.path.isfile(fp):
            continue
        res = {}
        with open(fp) as f:
            for line in f:
                if line.startswith("#") or line.lstrip().startswith("W"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                W, r = int(parts[0]), int(parts[1])
                res[(W, r)] = dict(
                    savings       = float(parts[2]),
                    outlier_pct   = float(parts[3]),
                    underflow_pct = float(parts[4]),
                )
        if res:
            all_sweep[m] = res
    return all_sweep


# ── Figure 1: PPL ─────────────────────────────────────────────────────────────

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
        print("No PPL data — skipping Figure 1")
        return

    n = len(avail); bw = 0.30; x = np.arange(n)
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
    allv = [ppl_orig[m] for m in avail] + [ppl_comp[m] for m in avail]
    ax.set_ylim(min(allv) * 0.97, max(allv) * 1.08)
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
    ap.add_argument("--plot-only",    action="store_true",
                    help="skip computation, regenerate figures from saved results")
    ap.add_argument("--only",         default=None,
                    help="comma-separated model keys to run (subset of MODELS)")
    args = ap.parse_args()

    window_sizes = [int(x) for x in args.window_sizes.split(",")]
    r_values     = [int(x) for x in args.r_values.split(",")]

    print("=== Figure 1: PPL comparison ===")
    plot_ppl(args.ppl_bitmap, args.out, args.dpi)

    print("\n=== Figure 2: savings + outlier sweep ===")

    if args.plot_only:
        all_sweep = load_results(args.out, MODELS, window_sizes, r_values)
        if not all_sweep:
            print("No saved results found — run without --plot-only first.")
            return
    else:
        only  = set(args.only.split(",")) if args.only else None
        avail = [m for m in MODELS
                 if os.path.isdir(os.path.join(args.models_dir, m))
                 and (only is None or m in only)]

        # Decide device per model based on rough weight count
        def _device_for(m):
            path = os.path.join(args.models_dir, m)
            files = ([f for f in os.listdir(path) if f.endswith(".safetensors")]
                     or [f for f in os.listdir(path) if f.endswith(".bin")])
            total_bytes = sum(os.path.getsize(os.path.join(path, f)) for f in files)
            # BF16: 2 bytes/weight → estimate weight count
            est_weights = total_bytes / 2
            use_gpu = (torch.cuda.is_available() and
                       est_weights >= GPU_THRESHOLD_WEIGHTS)
            return "cuda" if use_gpu else "cpu"

        tasks = [
            (m, os.path.join(args.models_dir, m), window_sizes, r_values,
             _device_for(m))
            for m in avail
        ]

        print("Model assignments:")
        for m, path, ws, rs, dev in tasks:
            print(f"  {SHORT.get(m, m):22s} → {dev}")
        print()

        all_sweep = {}
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(tasks), mp_context=ctx
        ) as exe:
            futs = {exe.submit(_worker_entry, t): t[0] for t in tasks}
            for fut in concurrent.futures.as_completed(futs):
                model_key, res = fut.result()
                all_sweep[model_key] = res
                print(f"  --> {SHORT.get(model_key, model_key)} done "
                      f"({len(res)} (W,r) entries)", flush=True)

        save_results(all_sweep, args.out, window_sizes, r_values)

    plot_sweep(all_sweep, window_sizes, r_values, args.out, args.dpi)
    print("\nAll done.")


if __name__ == "__main__":
    main()
