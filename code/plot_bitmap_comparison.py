#!/usr/bin/env python3
"""
plot_bitmap_comparison.py

Compare baseline exponent-only scheme vs bitmap+mantissa-borrow scheme at W=4096, r=4.

Figure 1: PPL — 3 bars per model: Original | Baseline W=4096 | Bitmap W=4096
Figure 2: Per-layer loss% and rms_norm% — side-by-side bars per layer group
"""

import glob, math, os, re
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

for _ttf in [
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf",
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf",
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-italic.otf",
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bolditalic.otf",
]:
    if os.path.isfile(_ttf):
        _fm.fontManager.addfont(_ttf)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 17,
    "legend.fontsize": 13, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})

SHORT = {
    "mistralai_Mistral-7B-v0.1":          "Mistral-7B",
    "microsoft_phi-2":                    "Phi-2",
    "Qwen_Qwen2.5-7B":                    "Qwen2.5-7B",
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
}
MODEL_ORDER = [
    "mistralai_Mistral-7B-v0.1",
    "microsoft_phi-2",
    "Qwen_Qwen2.5-7B",
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0",
]
LAYER_GROUPS = {
    "Embedding": lambda n: "embed_tokens" in n or "lm_head" in n,
    "Attn Q/K":  lambda n: re.search(r"self_attn\.[qk]_proj", n) is not None,
    "Attn V/O":  lambda n: re.search(r"self_attn\.[vo]_proj", n) is not None,
    "MLP":       lambda n: "mlp" in n and "proj" in n,
    "Norm":      lambda n: "norm" in n or "layernorm" in n,
}
GROUPS = list(LAYER_GROUPS)

C_ORIG   = "#aaaaaa"
C_BASE   = "#4e79a7"
C_BITMAP = "#e15759"


def _layer_group(name):
    for g, pred in LAYER_GROUPS.items():
        if pred(name):
            return g
    return None


# ── data loaders ───────────────────────────────────────────────────────────────

def load_ppl_file(path):
    """Returns (ppl_orig, ppl_comp, saved_pct, ppl_pct_inc) for the first data row."""
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    W = int(parts[0])
                except ValueError:
                    continue
                try:
                    return (float(parts[2]), float(parts[3]),
                            float(parts[1].rstrip("%")), float(parts[5].rstrip("%")))
                except (ValueError, IndexError):
                    continue
    return None


def load_ppl_dir(ppl_dir, r_cap=4, W=4096):
    data = {}
    for fp in glob.glob(os.path.join(ppl_dir, "*_ppl.txt")):
        model = re.sub(r"_rcap\d+.*", "", os.path.basename(fp))
        model = re.sub(r"_bitmap$", "", model)
        row = load_ppl_file(fp)
        if row:
            data[model] = row   # (orig, comp, saved%, %inc)
    return data


def load_layer_file(path):
    """Returns list of {layer, numel, loss_pct, rms_norm} dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith(("Model:", "layer", "-", "TOTAL")):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                numel    = int(parts[-6].replace(",", ""))
                loss_pct = float(parts[-3].rstrip("%"))
                snr_str  = parts[-2]
                rms_norm = float(parts[-1].rstrip("%"))
            except (ValueError, IndexError):
                continue
            name = " ".join(parts[:-6]).strip()
            rows.append(dict(layer=name, numel=numel,
                             loss_pct=loss_pct, rms_norm=rms_norm))
    return rows


def load_layer_dir(layer_dir, r_cap=4, W=4096, suffix="per_layer"):
    data = {}
    pat = re.compile(rf"^(.+)_W{W}_rcap{r_cap}.*_{suffix}\.txt$")
    for fp in glob.glob(os.path.join(layer_dir, "*.txt")):
        m = pat.match(os.path.basename(fp))
        if not m:
            continue
        data[m.group(1)] = load_layer_file(fp)
    return data


def _group_aggregate(rows, metric):
    gv, gw = {}, {}
    for r in rows:
        g = _layer_group(r["layer"])
        if g is None:
            continue
        gv[g] = gv.get(g, 0.0) + r[metric] * r["numel"]
        gw[g] = gw.get(g, 0.0) + r["numel"]
    return {g: gv[g] / gw[g] for g in gv if gw[g] > 0}


# ── Figure 1: PPL comparison ───────────────────────────────────────────────────

def plot_ppl(base_ppl, bitmap_ppl, models, out_dir, r_cap=4, dpi=200):
    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 5.8), sharey=False)
    if ncols == 1:
        axes = [axes]

    bar_w = 0.22
    xs = {"orig": 0, "base": bar_w + 0.06, "bitmap": 2 * (bar_w + 0.06)}

    for ci, model in enumerate(models):
        ax  = axes[ci]
        bd  = base_ppl.get(model)
        bmd = bitmap_ppl.get(model)
        if not bd and not bmd:
            ax.set_visible(False)
            continue

        ppl_orig = (bd or bmd)[0]
        all_vals = [ppl_orig]
        if bd:   all_vals.append(bd[1])
        if bmd:  all_vals.append(bmd[1])
        use_log = max(all_vals) / min(all_vals) > 50

        ax.bar(xs["orig"], ppl_orig, bar_w, color=C_ORIG, edgecolor="white",
               label="Original", zorder=3)

        if bd:
            pct = bd[3]
            lbl = f"+{pct/1000:.0f}k%" if pct >= 1000 else f"+{pct:.2f}%"
            ax.bar(xs["base"], bd[1], bar_w, color=C_BASE, edgecolor="white",
                   label="Baseline", zorder=3)
            ax.annotate(lbl, xy=(xs["base"], bd[1]), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        fontsize=11, fontweight="bold", color=C_BASE)

        if bmd:
            pct = bmd[3]
            lbl = f"+{pct:.2f}%" if abs(pct) >= 0.005 else "≈0%"
            ax.bar(xs["bitmap"], bmd[1], bar_w, color=C_BITMAP,
                   edgecolor="white", label="Bitmap+Mant", zorder=3)
            ax.annotate(lbl, xy=(xs["bitmap"], bmd[1]), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        fontsize=11, fontweight="bold", color=C_BITMAP)

        if use_log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
            ax.set_ylabel("Perplexity, log scale (↓ better)" if ci == 0 else "")
        else:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 1.18)
            ax.set_ylabel("Perplexity (↓ better)" if ci == 0 else "")

        ax.set_xticks(list(xs.values()))
        ax.set_xticklabels(["Original", "Baseline\nW=4096", "Bitmap\nW=4096"],
                           rotation=25, ha="right")
        ax.set_title(SHORT.get(model, model), fontweight="bold", pad=8)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

    handles = [
        plt.Rectangle((0,0),1,1, facecolor=C_ORIG,   edgecolor="white", label="Original"),
        plt.Rectangle((0,0),1,1, facecolor=C_BASE,   edgecolor="white", label=f"Baseline (exp-only, r={r_cap})"),
        plt.Rectangle((0,0),1,1, facecolor=C_BITMAP, edgecolor="white", label=f"Bitmap+MantissaBorrow (r={r_cap})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"WikiText-2 PPL: Baseline vs Bitmap+Mantissa-Borrow  (W=4096, r={r_cap})",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(h_pad=2.0, w_pad=1.5, rect=[0, 0.08, 1, 0.96])

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"bitmap_ppl_comparison_rcap{r_cap}.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.3)
    print(f"Saved: {p}")
    plt.close(fig)


# ── Figure 2: per-layer comparison ────────────────────────────────────────────

def plot_layer(base_layer, bitmap_layer, models, out_dir, r_cap=4, dpi=200):
    ncols = len(models)
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.8 * nrows), sharey=False)
    if ncols == 1:
        axes = [[axes[r]] for r in range(nrows)]

    bar_w = 0.30
    x_pos = np.arange(len(GROUPS))

    for ci, model in enumerate(models):
        bd  = base_layer.get(model, [])
        bmd = bitmap_layer.get(model, [])

        for ri, (metric, ylabel) in enumerate([("loss_pct",  "Loss rate (%)"),
                                               ("rms_norm",  "RMS norm error (%)")]):
            ax = axes[ri][ci]
            plotted = False

            if bd:
                agg = _group_aggregate(bd, metric)
                ys  = [agg.get(g, 0) for g in GROUPS]
                ax.bar(x_pos - bar_w/2, ys, bar_w, color=C_BASE,
                       edgecolor="white", label="Baseline", zorder=3)
                plotted = True

            if bmd:
                agg = _group_aggregate(bmd, metric)
                ys  = [agg.get(g, 0) for g in GROUPS]
                ax.bar(x_pos + bar_w/2, ys, bar_w, color=C_BITMAP,
                       edgecolor="white", label="Bitmap+Mant", zorder=3)
                plotted = True

            if not plotted:
                ax.set_visible(False)
                continue

            ax.set_xticks(x_pos)
            ax.set_xticklabels(GROUPS, rotation=30, ha="right")
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
            fmt = "%.4f%%" if metric == "loss_pct" else "%.2f%%"
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            if ci == 0:
                ax.set_ylabel(ylabel)
            if ri == 0:
                ax.set_title(SHORT.get(model, model), fontweight="bold", pad=8)

    handles = [
        plt.Rectangle((0,0),1,1, facecolor=C_BASE,   edgecolor="white", label=f"Baseline (exp-only, r={r_cap})"),
        plt.Rectangle((0,0),1,1, facecolor=C_BITMAP, edgecolor="white", label=f"Bitmap+MantissaBorrow (r={r_cap})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Per-Layer: Baseline vs Bitmap+Mantissa-Borrow  (W=4096, r={r_cap})",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(h_pad=2.0, w_pad=1.5, rect=[0, 0.06, 1, 0.97])

    p = os.path.join(out_dir, f"bitmap_per_layer_rcap{r_cap}.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.3)
    print(f"Saved: {p}")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppl-base",    default="results_ppl")
    ap.add_argument("--ppl-bitmap",  default="results_ppl_bitmap")
    ap.add_argument("--layer-base",  default="results_per_layer")
    ap.add_argument("--layer-bitmap",default="results_per_layer_bitmap")
    ap.add_argument("--out",         default="results_summary")
    ap.add_argument("--r-cap",       type=int, default=4)
    ap.add_argument("--dpi",         type=int, default=200)
    args = ap.parse_args()

    base_ppl   = load_ppl_dir(args.ppl_base,    r_cap=args.r_cap, W=4096)
    bitmap_ppl = load_ppl_dir(args.ppl_bitmap,  r_cap=args.r_cap, W=4096)
    base_layer  = load_layer_dir(args.layer_base,   r_cap=args.r_cap, W=4096, suffix="per_layer")
    bitmap_layer= load_layer_dir(args.layer_bitmap, r_cap=args.r_cap, W=4096, suffix="per_layer")

    all_models = set(base_ppl) | set(bitmap_ppl) | set(base_layer) | set(bitmap_layer)
    models = [m for m in MODEL_ORDER if m in all_models]
    for m in sorted(all_models):
        if m not in models:
            models.append(m)

    print(f"Models: {[SHORT.get(m,m) for m in models]}")
    print(f"Base PPL:    {list(base_ppl)}")
    print(f"Bitmap PPL:  {list(bitmap_ppl)}")
    print(f"Base layer:  {list(base_layer)}")
    print(f"Bitmap layer:{list(bitmap_layer)}")

    os.makedirs(args.out, exist_ok=True)
    plot_ppl(base_ppl, bitmap_ppl, models, args.out, args.r_cap, args.dpi)
    if base_layer or bitmap_layer:
        plot_layer(base_layer, bitmap_layer, models, args.out, args.r_cap, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
