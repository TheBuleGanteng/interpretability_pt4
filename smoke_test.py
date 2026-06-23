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

from interp.model_loading import capture_residual_stream, load_model  # noqa: E402
from interp.sae_loading import encode_decode, load_residual_sae  # noqa: E402

# Keep SAE width/l0 fixed across models so the FVU/L0 numbers are comparable.
SAE_WIDTH = "16k"
SAE_L0 = "medium"

# Bit-width is the second experimental axis: bf16 (uncompressed) -> 8bit -> 4bit.
# Columns in the summary tables are always shown in this order, regardless of the
# per-model run order below.
DISPLAY_BITWIDTHS = ("bf16", "8bit", "4bit")

# One entry per model. `available_layers` is the registry-verified set of layers
# with a trained width-16k/l0-medium residual SAE; `layer` is a middle layer
# chosen FROM that set (not n_layers/2). See module docstring for provenance.
#
# `run_bitwidths` is the order conditions are *executed* in for that model. For
# 4B/12B all three fit a 40GB A100 in any order. For 27B we run the two that fit
# first (4bit ~14GB, 8bit ~27GB) and put bf16 (~54GB, expected OOM on 40GB) LAST,
# so the one expected failure is the final condition of the whole run.
MODEL_CONFIGS = [
    {
        "model_name": "google/gemma-3-4b-pt",  # known-good baseline
        "sae_release": "gemma-scope-2-4b-pt-res",
        "available_layers": (9, 17, 22, 29),
        "layer": 17,
        "run_bitwidths": ("bf16", "8bit", "4bit"),
    },
    {
        "model_name": "google/gemma-3-12b-pt",
        "sae_release": "gemma-scope-2-12b-pt-res",
        "available_layers": (12, 24, 31, 41),
        "layer": 24,
        "run_bitwidths": ("bf16", "8bit", "4bit"),
    },
    {
        "model_name": "google/gemma-3-27b-pt",
        "sae_release": "gemma-scope-2-27b-pt-res",
        "available_layers": (16, 31, 40, 53),
        "layer": 31,
        # bf16 last: 27B bf16 (~54GB) is the expected OOM on a 40GB A100.
        "run_bitwidths": ("4bit", "8bit", "bf16"),
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


def run_condition(loaded_sae, name: str, precision: str, layer: int, text: str) -> dict | None:
    """Run one (model x bit-width) condition and return its metrics.

    Loads `name` at `precision`, captures the residual stream at `layer` over
    `text`, then measures reconstruction with the ALREADY-LOADED `loaded_sae`
    (the fixed instrument — never reloaded or varied per bit-width). Returns
    {fvu, l0, tokens} on success, or None on a d_in mismatch (reported as a
    skip). The model + activations are always freed in the finally block before
    returning so the next condition starts from a clean GPU. Exceptions (e.g.
    OOM) propagate to the caller — memory is freed here first either way.
    """
    loaded = None
    acts = None
    try:
        # --- load model at this precision, capture residual stream -------
        print(f"[1/2] Loading {name} at {precision}...")
        loaded = load_model(name, precision=precision, compute_dtype=torch.bfloat16)
        print("      done.")

        print(f"[2/2] Capturing residual stream at layer {layer}...")
        acts, info = capture_residual_stream(loaded, text, layer)
        d_in = loaded_sae.sae.cfg.d_in
        d_model = info["d_model"]
        match = d_model == d_in
        print(f"      input_ids shape: {info['input_ids_shape']}")
        print(
            f"      residual stream shape: {info['resid_shape']}  "
            f"(d_model={d_model})"
        )
        print(f"      SAE sae_id={loaded_sae.sae_id}  d_in={d_in}  match={match}")

        # d_in sanity check: a mismatch is the silent-wrong-answer trap. Skip
        # this condition and report rather than computing a meaningless FVU.
        if not match:
            print(
                f"  [!] MISMATCH: model d_model={d_model} != SAE d_in={d_in}. "
                f"Skipping {name} @ {precision}."
            )
            return None

        # num_tokens = product of all dims except the trailing d_model.
        num_tokens = 1
        for dim in info["resid_shape"][:-1]:
            num_tokens *= dim

        # encode_decode is precision-agnostic: it casts to float32 for the
        # metric. The SAE (fixed instrument) is unchanged across bit-widths.
        # The FVU denominator is read straight from the result (single source of
        # truth — the same scalar that feeds fvu = mse / var), not recomputed.
        result = encode_decode(loaded_sae.sae, acts)
        print(f"  tokens evaluated:     {num_tokens}")
        print(f"  reconstruction MSE:   {result['mse']:.6f}")
        print(f"  FVU denominator (variance): {result['fvu_denominator']:.4f}")
        print(f"  FVU (1 - R^2):        {result['fvu']:.4f}")
        print(f"  mean L0 (active feats): {result['l0']:.1f}")
        return {"fvu": result["fvu"], "l0": result["l0"], "tokens": num_tokens}
    finally:
        # Free GPU memory between EVERY condition — each bit-width is a fresh
        # model load, and a lingering previous model would cause a spurious OOM
        # on a condition that should fit. Runs on success AND on exception. The
        # SAE is intentionally NOT freed here; it persists across this model's
        # bit-widths and is freed by the caller when moving to the next model.
        del loaded, acts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _is_oom(exc: Exception) -> bool:
    """True if `exc` looks like a CUDA out-of-memory error."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


def main() -> None:
    print("=" * 70)
    print("Compression-gradient experiment: SAE reconstruction vs bit-width")
    print(f"  models:    {', '.join(m['model_name'] for m in MODEL_CONFIGS)}")
    print(f"  bit-widths: {', '.join(DISPLAY_BITWIDTHS)}  (model precision is the variable)")
    print(f"  SAE:       Gemma Scope 2 residual  width={SAE_WIDTH} l0={SAE_L0}  (fixed instrument)")
    print(f"  text:      {len(TEXT)} chars across a small batch of sentences")
    print("=" * 70)

    if not torch.cuda.is_available():
        print(
            "\n[!] No CUDA GPU visible (torch.cuda.is_available() is False).\n"
            "    The 8-bit/4-bit loads via bitsandbytes require a CUDA GPU, so\n"
            "    the model half of this experiment cannot run on this machine.\n"
            "    Run it on a CUDA box (A100 recommended) to exercise the matrix.\n"
        )
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nGPU: {gpu_name}")
        # The matrix is sized for a 40GB A100: everything fits EXCEPT 27B bf16
        # (~54GB), which is the documented hardware ceiling and OOM-skips. On a
        # smaller GPU, bf16 12B (~24GB) and all 27B conditions will likely OOM.
        large_gpu = any(
            tag in gpu_name for tag in ("A100", "H100", "H200", "A6000", "L40", "80GB")
        )
        if not large_gpu:
            print(
                "[!] This does not look like an A100/large GPU. Expect OOM on\n"
                "    bf16 12B (~24GB) and all 27B conditions. Smaller conditions\n"
                "    (4B all precisions, 12B 8bit/4bit) should still report.\n"
                "    On Colab Pro+: Runtime -> Change runtime type -> A100."
            )

    # results[model_name][precision] -> {"fvu":..,"l0":..} or a status string.
    results: dict[str, dict[str, object]] = {}

    for model_cfg in MODEL_CONFIGS:
        name = model_cfg["model_name"]
        layer = model_cfg["layer"]
        release = model_cfg["sae_release"]
        available_layers = tuple(model_cfg["available_layers"])
        results.setdefault(name, {})

        print("\n" + "=" * 70)
        print(f"MODEL: {name}  (layer {layer})")
        print("=" * 70)

        # Load the SAE ONCE per model and reuse it across all its bit-widths.
        loaded_sae = None
        try:
            print(f"Loading Gemma Scope 2 residual SAE (once for all bit-widths)...")
            loaded_sae = load_residual_sae(
                layer=layer,
                width=SAE_WIDTH,
                l0=SAE_L0,
                device="cpu",
                release=release,
                available_layers=available_layers,
            )
            print(
                f"      release={loaded_sae.release}  sae_id={loaded_sae.sae_id}  "
                f"d_in={loaded_sae.sae.cfg.d_in}  d_sae={loaded_sae.sae.cfg.d_sae}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED to load SAE for {name}: {type(exc).__name__}: {exc}")
            for precision in DISPLAY_BITWIDTHS:
                results[name][precision] = "SAE-FAIL"
            continue

        try:
            for precision in model_cfg["run_bitwidths"]:
                print(f"\n--- {name} @ {precision} ---")
                try:
                    metrics = run_condition(loaded_sae, name, precision, layer, TEXT)
                    results[name][precision] = metrics if metrics is not None else "MISMATCH"
                except Exception as exc:  # noqa: BLE001 — keep the matrix alive
                    if _is_oom(exc):
                        print(f"\n  SKIPPED (out of memory): {name} @ {precision}")
                        if "27b" in name and precision == "bf16":
                            print("      (expected: 27B bf16 ~54GB > 40GB A100)")
                        results[name][precision] = "N/A (OOM)"
                    else:
                        print(f"\n  FAILED: {type(exc).__name__}: {exc}")
                        results[name][precision] = "FAIL"
                    # Defensive re-clean after a failed condition.
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            # Free the SAE when moving to the next model.
            del loaded_sae
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _print_gradient_tables(results)
    print("\nExperiment complete.")


def _cell(entry: object, key: str) -> str:
    """Format one table cell: a metric value, or a status string as-is."""
    if isinstance(entry, dict):
        return f"{entry[key]:.4f}" if key == "fvu" else f"{entry[key]:.1f}"
    if entry is None:
        return "-"
    return str(entry)  # "N/A (OOM)", "FAIL", "MISMATCH", "SAE-FAIL"


def _print_gradient_tables(results: dict) -> None:
    """Print FVU and L0 as model x bit-width gradient tables."""
    col_w = 11
    header = "  {:<16}".format("model") + "".join(
        "{:>{w}}".format(bw, w=col_w) for bw in DISPLAY_BITWIDTHS
    )

    print("\n" + "=" * 70)
    print("SUMMARY — compression gradient")
    print("=" * 70)

    print("\nFVU by (model x bit-width)")
    print(header)
    for model_cfg in MODEL_CONFIGS:
        name = model_cfg["model_name"]
        row = "  {:<16}".format(name.replace("google/", ""))
        for bw in DISPLAY_BITWIDTHS:
            row += "{:>{w}}".format(_cell(results.get(name, {}).get(bw), "fvu"), w=col_w)
        print(row)

    print("\nL0 (active feats) by (model x bit-width)")
    print(header)
    for model_cfg in MODEL_CONFIGS:
        name = model_cfg["model_name"]
        row = "  {:<16}".format(name.replace("google/", ""))
        for bw in DISPLAY_BITWIDTHS:
            row += "{:>{w}}".format(_cell(results.get(name, {}).get(bw), "l0"), w=col_w)
        print(row)


if __name__ == "__main__":
    main()
