#!/usr/bin/env python3
"""
per_layer_bitmap_mantissa.py

Per-layer analysis of bitmap + mantissa-borrow exponent delta encoding.

Scheme (per block of W BF16 weights):
  1. Choose optimal base b (max-coverage optimal-L, identical to baseline).
  2. In-range weights  (e in [b, b+span]):  exact exponent + mantissa reconstruction.
  3. Underflow weights (e < b):             exponent clipped to b (same as baseline).
  4. Overflow weights  (e > b+span):
       - bitmap[i] = 1
       - lower 3 mantissa bits store lower 3 bits of (e - b - span)
       - upper 4 mantissa bits kept; lower 3 zeroed at decode
       - reconstructed exp = b + span + overflow_low3

Overhead vs. baseline: +1 bit/weight (bitmap)
  → savings at W=4096, r=4: (16-13)/16 = 18.75%  (baseline: 25%)

Usage:
  python3 per_layer_bitmap_mantissa.py /path/to/model \
      --window-sizes 4096 --r-cap 4 --out results_per_layer_bitmap
"""

import argparse
import math
import os

import numpy as np
import torch

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def bits_to_mb(b):
    return b / 8 / 1024 / 1024


def bf16_u16_to_float32(u16):
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _find_optimal_L(exp_blk, span):
    nc, W = exp_blk.shape
    span = int(span)
    if W <= 32:
        sb = np.sort(exp_blk, axis=1)
        covs = np.stack(
            [(sb <= sb[:, j:j+1] + span).sum(axis=1, dtype=np.int32) - j
             for j in range(W)],
            axis=1,
        )
        best_j = np.argmax(covs, axis=1)
        return sb[np.arange(nc, dtype=np.int64), best_j].astype(np.int64)
    else:
        BINS = 256
        row_offsets = np.arange(nc, dtype=np.int64) * BINS
        flat_idx = (exp_blk + row_offsets[:, None]).ravel()
        counts = (np.bincount(flat_idx, minlength=nc * BINS)
                  .astype(np.int16).reshape(nc, BINS))
        prefix = np.zeros((nc, BINS + 1), dtype=np.int32)
        np.cumsum(counts, axis=1, out=prefix[:, 1:])
        max_L = max(0, BINS - 1 - span)
        wc = prefix[:, span + 1: max_L + span + 2] - prefix[:, : max_L + 1]
        return np.argmax(wc, axis=1).astype(np.int64)


def _encode_decode_block(u16_blk, span):
    """
    Apply bitmap+mantissa-borrow codec to a 2-D block (nc, W) of uint16.
    Returns recon_u16 (same shape).
    """
    nc, W = u16_blk.shape
    exp_blk = ((u16_blk >> np.uint16(7)) & np.uint16(0xFF)).astype(np.int64)
    best_L  = np.clip(_find_optimal_L(exp_blk, span), 0, 255).astype(np.int64)
    L       = best_L[:, np.newaxis]                         # (nc, 1)

    delta    = exp_blk - L                                  # (nc, W)
    overflow = delta > span
    underflow = delta < 0

    # --- base reconstructed exponent (clip at both ends) ---
    exp_r = np.clip(delta, 0, span) + L                     # (nc, W)

    # --- overflow: recover lower 3 exponent bits from mantissa ---
    overflow_amount = np.where(overflow, delta - span, np.int64(0))
    overflow_low3   = (overflow_amount & np.int64(7)).astype(np.int64)
    exp_r = np.where(overflow,
                     np.clip(exp_r + overflow_low3, 0, 255),
                     exp_r).astype(np.uint16)

    # --- mantissa: zero lower 3 bits for overflow weights ---
    mant   = (u16_blk & np.uint16(0x7F))
    mant_r = np.where(overflow,
                      (mant & np.uint16(0x78)).astype(np.uint16),
                      mant)

    sign = u16_blk & np.uint16(0x8000)
    return (sign | (exp_r << np.uint16(7)) | mant_r.astype(np.uint16)).astype(np.uint16)


def _analyze_layer(u16, W, r_cap):
    """Return per-layer stats dict for one tensor at given W, r_cap."""
    N = len(u16)
    if N == 0:
        return None
    span    = np.int64((1 << r_cap) - 1)
    num_full = N // W
    rem      = N % W
    total_windows = num_full + (1 if rem else 0)
    cw = max(1, 2_000_000 // W) if W <= 32 else min(max(num_full, 1), 5000)

    total_loss  = 0
    sum_sq_orig = np.float64(0)
    sum_sq_err  = np.float64(0)

    def _block(u16_blk):
        nonlocal total_loss, sum_sq_orig, sum_sq_err
        f32_orig = bf16_u16_to_float32(u16_blk.ravel())
        recon    = _encode_decode_block(u16_blk, span)
        f32_recon = bf16_u16_to_float32(recon.ravel())
        total_loss  += int((recon != u16_blk).sum())
        err          = f32_orig.astype(np.float64) - f32_recon.astype(np.float64)
        sum_sq_orig += float(np.sum(f32_orig.astype(np.float64) ** 2))
        sum_sq_err  += float(np.sum(err ** 2))

    if num_full > 0:
        for cs in range(0, num_full, cw):
            ce = min(cs + cw, num_full)
            _block(u16[cs * W : ce * W].reshape(ce - cs, W))

    if rem:
        s      = num_full * W
        u16_t  = u16[s:].reshape(1, rem)
        recon_t = _encode_decode_block(u16_t, span)
        f32_o   = bf16_u16_to_float32(u16_t.ravel())
        f32_r   = bf16_u16_to_float32(recon_t.ravel())
        total_loss  += int((recon_t != u16_t).sum())
        err_t        = f32_o.astype(np.float64) - f32_r.astype(np.float64)
        sum_sq_orig += float(np.sum(f32_o.astype(np.float64) ** 2))
        sum_sq_err  += float(np.sum(err_t ** 2))

    baseline_bits   = 16 * N
    # base (8 bits/window) + bitmap (1 bit/weight) + delta (r bits/weight) + sign+mant (8 bits/weight)
    compressed_bits = total_windows * 8 + N * 1 + N * r_cap + N * 8
    saved_pct = (baseline_bits - compressed_bits) / baseline_bits * 100
    loss_pct  = total_loss / N * 100
    snr_db    = (10 * math.log10(sum_sq_orig / sum_sq_err)
                 if sum_sq_err > 0 else math.inf)
    rms_norm  = (math.sqrt(sum_sq_err / sum_sq_orig) * 100
                 if sum_sq_orig > 0 else 0.0)

    return dict(
        N=N, windows=total_windows,
        saved_pct=saved_pct, loss_count=total_loss, loss_pct=loss_pct,
        snr_db=snr_db, rms_norm_pct=rms_norm,
        base_mb=bits_to_mb(baseline_bits), comp_mb=bits_to_mb(compressed_bits),
    )


def load_tensors(model_path):
    files = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))
    ftype = "safetensors"
    if not files:
        files = sorted(f for f in os.listdir(model_path) if f.endswith(".bin"))
        ftype = "bin"
    for fname in files:
        fp = os.path.join(model_path, fname)
        tensors = load_file(fp) if ftype == "safetensors" else torch.load(fp, map_location="cpu")
        if not isinstance(tensors, dict):
            continue
        for name, t in tensors.items():
            if not isinstance(t, torch.Tensor) or t.numel() == 0:
                continue
            if t.dtype not in SUPPORTED_DTYPES:
                continue
            if t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)
            yield name, t.view(torch.int16).numpy().astype(np.uint16).ravel()


def run(model_path, window_sizes, r_cap, out_dir):
    model_name = os.path.basename(model_path.rstrip("/"))
    os.makedirs(out_dir, exist_ok=True)

    for W in window_sizes:
        print(f"\n{'='*90}")
        print(f"  {model_name}  r_cap={r_cap}  W={W}  [bitmap+mantissa-borrow]")
        print(f"{'='*90}")
        header = (f"{'layer':<55} {'numel':>10} {'saved%':>8} {'loss_ct':>9} "
                  f"{'loss%':>7} {'SNR(dB)':>9} {'rms_norm%':>10}")
        print(header)
        print("-" * len(header))

        rows = []
        for name, u16 in load_tensors(model_path):
            if len(u16) < W:
                continue
            st = _analyze_layer(u16, W, r_cap)
            if st is None:
                continue
            snr_str = "inf" if math.isinf(st["snr_db"]) else f"{st['snr_db']:.2f}"
            print(f"{name:<55} {st['N']:>10,} {st['saved_pct']:>7.2f}% "
                  f"{st['loss_count']:>9,} {st['loss_pct']:>6.3f}% "
                  f"{snr_str:>9} {st['rms_norm_pct']:>9.4f}%")
            rows.append({"layer": name, **st})

        if not rows:
            print("  (no eligible tensors)")
            continue

        total_N    = sum(r["N"] for r in rows)
        total_loss = sum(r["loss_count"] for r in rows)
        total_base = sum(r["base_mb"] for r in rows)
        total_comp = sum(r["comp_mb"] for r in rows)
        agg_saved  = (total_base - total_comp) / total_base * 100 if total_base else 0
        agg_loss   = total_loss / total_N * 100 if total_N else 0

        print("-" * len(header))
        print(f"{'TOTAL':<55} {total_N:>10,} {agg_saved:>7.2f}% "
              f"{total_loss:>9,} {agg_loss:>6.3f}%")

        out_path = os.path.join(out_dir, f"{model_name}_W{W}_rcap{r_cap}_bitmap_per_layer.txt")
        with open(out_path, "w") as f:
            f.write(f"Model: {model_name}  r_cap={r_cap}  W={W}  scheme=bitmap+mantissa-borrow\n")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for r in rows:
                snr_str = "inf" if math.isinf(r["snr_db"]) else f"{r['snr_db']:.2f}"
                f.write(f"{r['layer']:<55} {r['N']:>10,} {r['saved_pct']:>7.2f}% "
                        f"{r['loss_count']:>9,} {r['loss_pct']:>6.3f}% "
                        f"{snr_str:>9} {r['rms_norm_pct']:>9.4f}%\n")
            f.write("-" * len(header) + "\n")
            f.write(f"{'TOTAL':<55} {total_N:>10,} {agg_saved:>7.2f}% "
                    f"{total_loss:>9,} {agg_loss:>6.3f}%\n")
        print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--window-sizes", default="4096")
    ap.add_argument("--r-cap", type=int, default=4)
    ap.add_argument("--out", default="results_per_layer_bitmap")
    args = ap.parse_args()
    window_sizes = [int(x) for x in args.window_sizes.split(",")]
    run(args.model_path, window_sizes, args.r_cap, args.out)


if __name__ == "__main__":
    main()
