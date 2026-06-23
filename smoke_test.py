#!/usr/bin/env python3
"""Standalone feasibility smoke test, across multiple Gemma 3 model sizes.

End-to-end check that the pieces fit together, run once per model in
MODEL_CONFIGS (4B baseline, 12B, 27B):
  1. Load the model in 4-bit.
  2. Run a batch of text and capture the residual stream at a middle layer.
  3. Load the matching Gemma Scope 2 residual-stream SAE for that layer (SAELens).
  4. Encode the activations and report tensor shapes + reconstruction error.
  5. Free GPU memory before moving to the next model.

This is a feasibility check, not the full harness. The 4-bit model load needs a
CUDA GPU; the SAE load + encode runs on CPU. Run with:

    .venv/bin/python smoke_test.py

The SAE release strings and per-model available layers below were resolved from
the SAELens pretrained-SAE registry (get_pretrained_saes_directory), NOT guessed:
  - 4B  gemma-scope-2-4b-pt-res   layers 9, 17, 22, 29   -> layer 17
  - 12B gemma-scope-2-12b-pt-res  layers 12, 24, 31, 41  -> layer 24
  - 27B gemma-scope-2-27b-pt-res  layers 16, 31, 40, 53  -> layer 31
All use width=16k, l0=medium so reconstruction numbers are comparable.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch  # noqa: E402

from interp.model_loading import capture_residual_stream, load_model_4bit  # noqa: E402
from interp.sae_loading import encode_decode, load_residual_sae  # noqa: E402

# Keep SAE width/l0 fixed across models so the FVU/L0 numbers are comparable.
SAE_WIDTH = "16k"
SAE_L0 = "medium"

# One entry per model. `available_layers` is the registry-verified set of layers
# with a trained width-16k/l0-medium residual SAE; `layer` is a middle layer
# chosen FROM that set (not n_layers/2). See module docstring for provenance.
MODEL_CONFIGS = [
    {
        "model_name": "google/gemma-3-4b-pt",  # known-good baseline
        "sae_release": "gemma-scope-2-4b-pt-res",
        "available_layers": (9, 17, 22, 29),
        "layer": 17,
    },
    {
        "model_name": "google/gemma-3-12b-pt",
        "sae_release": "gemma-scope-2-12b-pt-res",
        "available_layers": (12, 24, 31, 41),
        "layer": 24,
    },
    {
        "model_name": "google/gemma-3-27b-pt",
        "sae_release": "gemma-scope-2-27b-pt-res",
        "available_layers": (16, 31, 40, 53),
        "layer": 31,
    },
]

# FVU on a single 6-token prompt is statistically meaningless. To get a
# believable reconstruction reading we run a small batch of generic English
# text (a few hundred tokens) through the model and compute the metrics over
# every captured token. A handful of plain sentences is plenty for a smoke
# test; the SAE was trained on monology/pile-uncopyrighted, generic prose.
TEXT = " ".join(
    [
        "The capital of France is Paris, a city on the river Seine.",
        "In the morning the streets were quiet and the air was cold.",
        "Scientists have long studied how the brain stores memories.",
        "A small boat drifted slowly across the calm grey water.",
        "The library held thousands of books on every imaginable subject.",
        "She opened the window and listened to the rain on the roof.",
        "Economists disagree about the long-term effects of the policy.",
        "The mountain trail climbed steeply through the pine forest.",
        "He poured a cup of coffee and read the newspaper at the table.",
        "The committee met on Tuesday to discuss the annual budget.",
        "Children played in the park while their parents talked nearby.",
        "The old clock on the wall had stopped at a quarter past three.",
        "Engineers tested the new bridge under heavy simulated traffic.",
        "A gentle wind moved through the tall grass on the open plain.",
        "The recipe called for flour, sugar, butter, and two fresh eggs.",
        "Historians continue to debate the causes of the ancient war.",
        "The train arrived at the station exactly on schedule that day.",
        "Bright stars filled the clear sky above the silent desert.",
        "The teacher explained the problem again using a simple diagram.",
        "Farmers harvested the wheat before the first autumn frost came.",
        "The museum's new exhibit drew large crowds over the weekend.",
        "A long line of cars stretched down the busy city avenue.",
        "The doctor reviewed the chart and recommended further rest.",
        "Waves crashed against the rocks at the base of the tall cliff.",
        "The software update fixed several bugs reported by early users.",
        "Birds gathered on the wire as the sun began to set in the west.",
    ]
)


def run_one_model(model_cfg: dict, text: str) -> dict | None:
    """Run the full load -> capture -> SAE -> metrics check for one model.

    Returns a summary dict {model, layer, fvu, l0, tokens} on success, or None
    if the model/SAE mismatch on d_in (which is reported and treated as a skip).
    GPU memory is always freed in the finally block before returning, so the
    next (larger) model has room. Exceptions (e.g. OOM) propagate to the caller,
    which logs and continues — but memory is freed here first either way.
    """
    name = model_cfg["model_name"]
    layer = model_cfg["layer"]
    release = model_cfg["sae_release"]
    available_layers = tuple(model_cfg["available_layers"])

    loaded = None
    loaded_sae = None
    acts = None
    try:
        # --- 1 & 2: load model, capture residual stream ------------------
        print("[1/3] Loading model in 4-bit (NF4)...")
        loaded = load_model_4bit(name, compute_dtype=torch.bfloat16)
        print("      done.")

        print(f"[2/3] Capturing residual stream at layer {layer}...")
        acts, info = capture_residual_stream(loaded, text, layer)
        print(f"      input_ids shape: {info['input_ids_shape']}")
        print(
            f"      residual stream shape: {info['resid_shape']}  "
            f"(d_model={info['d_model']})"
        )
        print(f"      (model has {info['n_layers']} layers)")

        # --- 3: load SAE, encode, reconstruction error -------------------
        print(f"[3/3] Loading Gemma Scope 2 residual SAE for layer {layer}...")
        loaded_sae = load_residual_sae(
            layer=layer,
            width=SAE_WIDTH,
            l0=SAE_L0,
            device="cpu",
            release=release,
            available_layers=available_layers,
        )
        d_in = loaded_sae.sae.cfg.d_in
        d_model = info["d_model"]
        match = d_model == d_in
        print(f"      release={loaded_sae.release}  sae_id={loaded_sae.sae_id}")
        print(
            f"      SAE d_in={d_in}  d_sae={loaded_sae.sae.cfg.d_sae}  "
            f"model d_model={d_model}  match={match}"
        )

        # d_in sanity check: a mismatch is the silent-wrong-answer trap. Stop
        # this model and report rather than computing a meaningless FVU.
        if not match:
            print(
                f"  [!] MISMATCH: model d_model={d_model} != SAE d_in={d_in}. "
                f"Skipping {name} (model/SAE coordinates do not line up)."
            )
            return None

        # num_tokens = product of all dims except the trailing d_model.
        num_tokens = 1
        for dim in info["resid_shape"][:-1]:
            num_tokens *= dim

        result = encode_decode(loaded_sae.sae, acts)
        print("\n--- results ---")
        print(f"  activations shape:    {info['resid_shape']}")
        print(f"  feature acts shape:   {result['feature_acts_shape']}")
        print(f"  tokens evaluated:     {num_tokens}")
        print(f"  reconstruction MSE:   {result['mse']:.6f}")
        print(f"  mean per-token L2:    {result['l2']:.4f}")
        print(f"  FVU (1 - R^2):        {result['fvu']:.4f}")
        print(f"  mean L0 (active feats): {result['l0']:.1f}")
        return {
            "model": name,
            "layer": layer,
            "fvu": result["fvu"],
            "l0": result["l0"],
            "tokens": num_tokens,
        }
    finally:
        # Free GPU memory between models — essential, or 27B on top of 4B+12B
        # OOMs. Drop refs to the model, captured activations and SAE, then
        # collect and empty the CUDA cache. Runs on success AND on exception.
        del loaded, loaded_sae, acts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    print("=" * 70)
    print("Mechanistic-interpretability smoke test (multi-model)")
    print(f"  models:  {', '.join(m['model_name'] for m in MODEL_CONFIGS)}")
    print(f"  SAE:     Gemma Scope 2 residual  width={SAE_WIDTH} l0={SAE_L0}")
    print(f"  text:    {len(TEXT)} chars across a small batch of sentences")
    print("=" * 70)

    if not torch.cuda.is_available():
        print(
            "\n[!] No CUDA GPU visible (torch.cuda.is_available() is False).\n"
            "    The 4-bit model load via bitsandbytes requires a CUDA GPU, so\n"
            "    the model half of this smoke test cannot run on this machine.\n"
            "    Run it on a CUDA box to exercise the full path.\n"
        )
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nGPU: {gpu_name}")
        # 27B in 4-bit is ~14-16GB of weights; with activations + SAE on top a
        # 16GB T4 will likely OOM. 4B (~3.5GB) and 12B (~7GB) fit a T4 fine.
        wants_27b = any("27b" in m["model_name"] for m in MODEL_CONFIGS)
        if wants_27b and "T4" in gpu_name:
            print(
                "[!] 27B in the list and the runtime looks like a T4 (16GB).\n"
                "    27B 4-bit will likely OOM here; it should fit an A100 (40GB).\n"
                "    On Colab Pro+: Runtime -> Change runtime type -> A100 for 27B.\n"
                "    4B and 12B will still run and report on a T4."
            )

    summaries: list[dict] = []
    for model_cfg in MODEL_CONFIGS:
        name = model_cfg["model_name"]
        print("\n" + "=" * 70)
        print(f"MODEL: {name}  (layer {model_cfg['layer']})")
        print("=" * 70)
        try:
            summary = run_one_model(model_cfg, TEXT)
            if summary is not None:
                summaries.append(summary)
        except Exception as exc:  # noqa: BLE001 — keep the run alive per-model
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                "out of memory" in str(exc).lower()
            )
            if is_oom:
                print(f"\n  SKIPPED (out of memory): {name}")
                print("      27B needs an A100 (40GB); a T4 (16GB) will OOM.")
            else:
                print(f"\n  FAILED: {type(exc).__name__}: {exc}")
            # Make sure nothing the failed run left behind blocks the next model.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    # --- one-line summary table for everything that ran ------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if not summaries:
        print("  No models completed successfully.")
    else:
        print(f"  {'model':<24} {'layer':>5} {'FVU':>10} {'L0':>8}")
        for s in summaries:
            print(
                f"  {s['model']:<24} {s['layer']:>5} "
                f"{s['fvu']:>10.4f} {s['l0']:>8.1f}"
            )
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
