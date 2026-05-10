#!/usr/bin/env python3
"""
eval_ppl_bitmap_mantissa.py

WikiText-2 perplexity evaluation for bitmap + mantissa-borrow exponent delta encoding.

Scheme (per block of W BF16 weights):
  - In-range  (e in [b, b+2^r-1]):  exact reconstruction
  - Underflow (e < b):               exponent clipped to b
  - Overflow  (e > b+2^r-1):
      bitmap[i]=1; lower 3 mantissa bits store lower 3 bits of (e - b - span);
      upper 4 mantissa bits kept; lower 3 zeroed at decode;
      reconstructed exp = b + span + overflow_low3

Overhead vs. baseline exponent-only scheme: +1 bit/weight for bitmap
  savings at W=4096, r=4: (16-13)/16 = 18.75%

Usage:
  python3 eval_ppl_bitmap_mantissa.py /path/to/model \
      --window-sizes 4096 --r-cap 4 --device cuda --out results_ppl_bitmap
"""

import argparse
import math
import os

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# ── codec ──────────────────────────────────────────────────────────────────────

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
    """Bitmap+mantissa-borrow codec for a (nc, W) uint16 block."""
    nc, W   = u16_blk.shape
    exp_blk = ((u16_blk >> np.uint16(7)) & np.uint16(0xFF)).astype(np.int64)
    best_L  = np.clip(_find_optimal_L(exp_blk, span), 0, 255).astype(np.int64)
    L       = best_L[:, np.newaxis]

    delta    = exp_blk - L
    overflow = delta > span

    # Base reconstructed exponent (clip underflow to L, overflow to L+span)
    exp_r = np.clip(delta, 0, span) + L

    # Overflow: recover lower 3 bits of (e - L - span) from mantissa
    overflow_amount = np.where(overflow, delta - span, np.int64(0))
    overflow_low3   = (overflow_amount & np.int64(7))
    exp_r = np.where(overflow,
                     np.clip(exp_r + overflow_low3, 0, 255),
                     exp_r).astype(np.uint16)

    # Mantissa: zero lower 3 bits for overflow weights (they encoded exponent bits)
    mant   = u16_blk & np.uint16(0x7F)
    mant_r = np.where(overflow,
                      (mant & np.uint16(0x78)).astype(np.uint16),
                      mant)

    sign = u16_blk & np.uint16(0x8000)
    return (sign | (exp_r << np.uint16(7)) | mant_r.astype(np.uint16)).astype(np.uint16)


def compress_decompress_bitmap(t_bf16, W, r_cap):
    """Apply bitmap+mantissa-borrow codec to a BF16 tensor. Returns same shape/dtype."""
    orig_shape = t_bf16.shape
    u16 = t_bf16.view(torch.int16).numpy().astype(np.uint16).ravel()
    N   = len(u16)
    span     = np.int64((1 << r_cap) - 1)
    num_full = N // W
    rem      = N % W
    cw = max(1, 2_000_000 // W) if W <= 32 else min(max(num_full, 1), 5000)
    recon = u16.copy()

    if num_full > 0:
        for cs in range(0, num_full, cw):
            ce = min(cs + cw, num_full)
            blk = u16[cs * W : ce * W].reshape(ce - cs, W)
            recon[cs * W : ce * W] = _encode_decode_block(blk, span).ravel()

    if rem:
        blk_t = u16[num_full * W :].reshape(1, rem)
        recon[num_full * W :] = _encode_decode_block(blk_t, span).ravel()

    return (torch.from_numpy(recon.view(np.int16))
            .view(torch.bfloat16)
            .reshape(orig_shape))


def _savings_pct(N, W, r_cap):
    num_full = N // W
    total_windows = num_full + (1 if N % W else 0)
    baseline   = 16 * N
    compressed = total_windows * 8 + N * 1 + N * r_cap + N * 8
    return (baseline - compressed) / baseline * 100


# ── perplexity ─────────────────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, device, n_tokens=16384, stride=512):
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                        trust_remote_code=True)
    text = "\n\n".join(ds["text"])
    enc  = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    if input_ids.numel() > n_tokens:
        input_ids = input_ids[:n_tokens]

    seq_len = input_ids.numel()
    block   = min(getattr(model.config, "max_position_embeddings", 2048), 2048)

    nlls = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, seq_len - 1, stride):
            end        = min(begin + block, seq_len)
            chunk      = input_ids[begin:end].unsqueeze(0).to(device)
            ctx_len    = max(0, begin + block - seq_len) if end == seq_len else max(0, block - stride)
            target_len = end - begin - ctx_len
            if target_len <= 0:
                continue
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                out = model(chunk)
            logits = out.logits[0, :-1][-target_len:]
            labels = chunk[0, 1:].long()[-target_len:]
            nll    = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            nlls.append((nll.item(), target_len))

    total_nll = sum(n for n, _ in nlls)
    total_tok = sum(t for _, t in nlls)
    return math.exp(total_nll / total_tok) if total_tok > 0 else float("inf")


def apply_compression(model, W, r_cap):
    seen_ptrs = set()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype not in SUPPORTED_DTYPES:
                continue
            if param.numel() < W:
                continue
            ptr = param.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            orig_dtype = param.dtype
            t     = param.data.to(torch.bfloat16).cpu().contiguous()
            recon = compress_decompress_bitmap(t, W, r_cap).to(orig_dtype)
            param.data.copy_(recon.to(param.device))


# ── model ID map ───────────────────────────────────────────────────────────────

HF_ID_MAP = {
    "TinyLlama_TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft_phi-2":                    "microsoft/phi-2",
    "mistralai_Mistral-7B-v0.1":          "mistralai/Mistral-7B-v0.1",
    "Qwen_Qwen2.5-7B":                    "Qwen/Qwen2.5-7B",
}


def _hf_id(model_name):
    return HF_ID_MAP.get(model_name, model_name.replace("_", "/", 1))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--window-sizes", default="4096")
    ap.add_argument("--r-cap",   type=int, default=4)
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out",     default="results_ppl_bitmap")
    ap.add_argument("--n-tokens", type=int, default=16384)
    args = ap.parse_args()

    window_sizes = [int(x) for x in args.window_sizes.split(",")]
    device       = torch.device(args.device)
    model_name   = os.path.basename(args.model_path.rstrip("/"))
    hf_id        = _hf_id(model_name)
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading tokenizer for {hf_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model ({model_name}) onto {device} ...")
    cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    if not hasattr(cfg, "pad_token_id") or cfg.pad_token_id is None:
        cfg.pad_token_id = cfg.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, config=cfg, dtype=torch.bfloat16,
        device_map=str(device), trust_remote_code=True,
    )
    model.eval()

    print("Evaluating baseline perplexity ...")
    ppl_orig = compute_perplexity(model, tokenizer, device, n_tokens=args.n_tokens)
    print(f"  Baseline PPL = {ppl_orig:.3f}")

    results = []
    for W in window_sizes:
        print(f"\n--- W={W}  r_cap={args.r_cap}  [bitmap+mantissa-borrow] ---")
        model_w = AutoModelForCausalLM.from_pretrained(
            hf_id, config=cfg, dtype=torch.bfloat16,
            device_map=str(device), trust_remote_code=True,
        )
        model_w.eval()

        print("  Applying compression ...")
        apply_compression(model_w, W, args.r_cap)

        print("  Evaluating compressed perplexity ...")
        ppl_comp = compute_perplexity(model_w, tokenizer, device, n_tokens=args.n_tokens)

        total_N = sum(
            p.numel() for p in model_w.parameters()
            if p.dtype in SUPPORTED_DTYPES and p.numel() >= W
        )
        saved    = _savings_pct(total_N, W, args.r_cap)
        delta    = ppl_comp - ppl_orig
        pct_inc  = (ppl_comp / ppl_orig - 1) * 100
        print(f"  Compressed PPL = {ppl_comp:.3f}  (Δ={delta:+.3f}, {pct_inc:+.2f}%)")
        print(f"  Savings = {saved:.2f}%")

        results.append(dict(W=W, r_cap=args.r_cap, saved_pct=saved,
                            ppl_orig=ppl_orig, ppl_comp=ppl_comp,
                            ppl_delta=delta, ppl_pct_inc=pct_inc))
        del model_w
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  {model_name}  r_cap={args.r_cap}  scheme=bitmap+mantissa-borrow")
    print(f"{'='*70}")
    print(f"  Baseline PPL: {ppl_orig:.3f}")
    print(f"  {'W':>6}  {'saved%':>8}  {'ppl_orig':>10}  {'ppl_comp':>10}  "
          f"{'Δppl':>8}  {'%inc':>8}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['W']:>6}  {r['saved_pct']:>7.2f}%  {r['ppl_orig']:>10.3f}  "
              f"{r['ppl_comp']:>10.3f}  {r['ppl_delta']:>+8.3f}  {r['ppl_pct_inc']:>+7.2f}%")

    out_path = os.path.join(args.out, f"{model_name}_rcap{args.r_cap}_bitmap_ppl.txt")
    with open(out_path, "w") as f:
        f.write(f"Model: {model_name}  r_cap={args.r_cap}  scheme=bitmap+mantissa-borrow\n")
        f.write(f"Baseline PPL: {ppl_orig:.3f}\n\n")
        f.write(f"{'W':>6}  {'saved%':>8}  {'ppl_orig':>10}  {'ppl_comp':>10}  "
                f"{'Δppl':>8}  {'%inc':>8}\n")
        f.write("-" * 62 + "\n")
        for r in results:
            f.write(f"{r['W']:>6}  {r['saved_pct']:>7.2f}%  {r['ppl_orig']:>10.3f}  "
                    f"{r['ppl_comp']:>10.3f}  {r['ppl_delta']:>+8.3f}  "
                    f"{r['ppl_pct_inc']:>+7.2f}%\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
