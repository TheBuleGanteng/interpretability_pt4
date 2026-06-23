#!/usr/bin/env python3
"""Standalone feasibility smoke test.

End-to-end check that the pieces fit together:
  1. Load google/gemma-3-4b-pt in 4-bit.
  2. Run a short text input and capture the residual stream at one middle layer.
  3. Load the matching Gemma Scope 2 residual-stream SAE for that layer (SAELens).
  4. Encode the activations and report tensor shapes + reconstruction error.

This is a feasibility check, not the full harness. The 4-bit model load needs a
CUDA GPU; the SAE load + encode runs on CPU. Run with:

    .venv/bin/python smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch  # noqa: E402

from interp.config import CONFIG  # noqa: E402
from interp.model_loading import capture_residual_stream, load_model_4bit  # noqa: E402
from interp.sae_loading import encode_decode, load_residual_sae  # noqa: E402

PROMPT = "The capital of France is"


def main() -> None:
    cfg = CONFIG
    print("=" * 70)
    print("Mechanistic-interpretability smoke test")
    print(f"  model:   {cfg.model_name}")
    print(f"  layer:   {cfg.layer} (residual stream, resid_post)")
    print(f"  SAE:     {cfg.sae_release}  width={cfg.sae_width} l0={cfg.sae_l0}")
    print(f"  prompt:  {PROMPT!r}")
    print("=" * 70)

    if not torch.cuda.is_available():
        print(
            "\n[!] No CUDA GPU visible (torch.cuda.is_available() is False).\n"
            "    The 4-bit model load via bitsandbytes requires a CUDA GPU, so\n"
            "    the model half of this smoke test cannot run on this machine.\n"
            "    Run it on a CUDA box to exercise the full path.\n"
        )

    # --- 1 & 2: load model, capture residual stream ----------------------
    print("[1/3] Loading model in 4-bit (NF4)...")
    loaded = load_model_4bit(cfg.model_name, compute_dtype=torch.bfloat16)
    print("      done.")

    print(f"[2/3] Capturing residual stream at layer {cfg.layer}...")
    acts, info = capture_residual_stream(loaded, PROMPT, cfg.layer)
    print(f"      input_ids shape: {info['input_ids_shape']}")
    print(f"      residual stream shape: {info['resid_shape']}  (d_model={info['d_model']})")
    print(f"      (model has {info['n_layers']} layers)")

    # --- 3: load SAE, encode, reconstruction error -----------------------
    print(f"[3/3] Loading Gemma Scope 2 residual SAE for layer {cfg.layer}...")
    loaded_sae = load_residual_sae(
        layer=cfg.layer,
        width=cfg.sae_width,
        l0=cfg.sae_l0,
        device="cpu",
    )
    print(f"      release={loaded_sae.release}  sae_id={loaded_sae.sae_id}")
    print(f"      SAE d_in={loaded_sae.sae.cfg.d_in}  d_sae={loaded_sae.sae.cfg.d_sae}")

    result = encode_decode(loaded_sae.sae, acts)
    print("\n--- results ---")
    print(f"  activations shape:    {info['resid_shape']}")
    print(f"  feature acts shape:   {result['feature_acts_shape']}")
    print(f"  reconstruction MSE:   {result['mse']:.6f}")
    print(f"  mean per-token L2:    {result['l2']:.4f}")
    print(f"  FVU (1 - R^2):        {result['fvu']:.4f}")
    print(f"  mean L0 (active feats): {result['l0']:.1f}")
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
