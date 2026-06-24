#!/usr/bin/env python3
"""Standalone feasibility smoke test, across multiple Gemma 3 model sizes.

End-to-end check that the pieces fit together, run once per (model x bit-width)
in MODEL_CONFIGS (4B baseline, 12B):
  1. Load the model at the chosen precision (bf16 / 8bit / 4bit).
  2. STREAM a Pile corpus, packed into fixed-length token chunks, and capture
     the residual stream at a middle layer one chunk at a time.
  3. Load the matching Gemma Scope 2 residual-stream SAE for that layer (SAELens).
  4. Encode each chunk and accumulate reconstruction sums INCREMENTALLY.
  5. Free GPU memory before moving to the next condition / model.

This is a feasibility check, not the full harness. The 8/4-bit model load needs a
CUDA GPU; the SAE load + encode runs on CPU. Run with:

    .venv/bin/python smoke_test.py

The SAE release strings and per-model available layers below were resolved from
the SAELens pretrained-SAE registry (get_pretrained_saes_directory), NOT guessed:
  - 4B  gemma-scope-2-4b-pt-res   layers 9, 17, 22, 29   -> layer 17
  - 12B gemma-scope-2-12b-pt-res  layers 12, 24, 31, 41  -> layer 24
All use width=16k, l0=medium so reconstruction numbers are comparable.

Reconstruction is measured over a properly sized sample (default N_TOKENS) streamed
from the SAE's own training distribution (the Pile), not a handful of hand-written
sentences, so the FVU/L0 numbers are statistically stable. FVU is accumulated as a
ratio of SUMS across chunks (numerator / denominator divided once at the end), never
as an average of per-chunk FVUs. Each condition also reports FVU at increasing token
counts so you can SEE the metric stabilize as the sample grows.

Two measurements run independently, each toggleable from the first Colab cell (or an
env var of the same name):
  MEASURE_SAE         — SAE reconstruction FVU/L0 (the existing path).
  MEASURE_NOISE_FLOOR — activation-perturbation control: how much quantization moves
                        the residual stream itself (independent of the SAE), compared
                        to the SAE's own reconstruction residual, so a FLAT FVU can be
                        interpreted (genuine-but-survived vs below instrument resolution).
A MEASURE_BEHAVIOR flag is intentionally NOT added here (a separate future task); the
toggle structure is left extensible for it.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch  # noqa: E402

from interp.corpus import iter_token_batches, load_pile_stream  # noqa: E402
from interp.model_loading import capture_resid_from_input_ids, load_model  # noqa: E402
from interp.noise_floor import measure_perturbation  # noqa: E402
from interp.sae_loading import encode_decode_sums, load_residual_sae  # noqa: E402


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from the environment (1/true/yes/on), else `default`."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, else `default`."""
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw.strip() else default


# --- Measurement toggles (set in the first Colab cell, or via env var) -----
# Each measurement runs only if enabled. With MEASURE_NOISE_FLOOR off the behavior
# is exactly the original FVU path: no extra compute, no bf16-reference passes.
MEASURE_SAE = _env_bool("MEASURE_SAE", True)
MEASURE_NOISE_FLOOR = _env_bool("MEASURE_NOISE_FLOOR", True)
# MEASURE_BEHAVIOR — reserved for a future task; not implemented here.

# Keep SAE width/l0 fixed across models so the FVU/L0 numbers are comparable.
SAE_WIDTH = "16k"
SAE_L0 = "medium"

# --- Corpus sampling ------------------------------------------------------
# How many tokens of streamed Pile text each condition's HEADLINE metrics are
# measured over. ~315 hand-written tokens (the old sample) was far too small for
# a stable FVU; tens of thousands of in-distribution tokens is the fix. This is
# also the token budget for the noise-floor control (so it covers the SAME tokens).
N_TOKENS = _env_int("N_TOKENS", 50_000)

# We measure FVU at increasing token counts so we can SEE where it stops moving —
# that stability is the evidence the sample is big enough. Each condition streams
# up to MAX_TOKENS (capped below) and snapshots FVU as it crosses each checkpoint.
STABILITY_CHECKPOINTS = (5_000, 20_000, 50_000, 100_000, 200_000, 500_000)

# Hard cap on how many tokens any single FVU condition streams. The 200k/500k
# checkpoints make a full run long (500k x 6 conditions); DIAL THIS DOWN (e.g.
# MAX_TOKENS=100_000) for a quick run. Checkpoints above MAX_TOKENS are skipped.
# Streaming + sum-based accumulation keep memory flat regardless of this value:
# each activation chunk is freed right after its sums are accumulated.
MAX_TOKENS = _env_int("MAX_TOKENS", 500_000)

# Shuffle the streamed Pile with a FIXED seed so token difficulty is averaged
# rather than stream-order dependent, while staying deterministic — every
# condition re-iterates the same shuffled order and therefore sees identical
# tokens (essential for the noise-floor per-token comparison). Set SHUFFLE_SEED
# to a negative value to disable shuffling.
SHUFFLE_SEED = _env_int("SHUFFLE_SEED", 0)
SHUFFLE_BUFFER = _env_int("SHUFFLE_BUFFER", 10_000)

# Activations are streamed and accumulated in fixed-length packed chunks; the
# residual-stream tensor for a whole sample would never fit in memory. CHUNK_LEN
# tokens per sequence, BATCH_SIZE sequences per forward -> BATCH_SIZE*CHUNK_LEN
# tokens accumulated and then FREED per step.
CHUNK_LEN = 512
BATCH_SIZE = 4


def _active_checkpoints() -> list[int]:
    """Sorted stability checkpoints that fit within the MAX_TOKENS budget."""
    return [c for c in sorted(STABILITY_CHECKPOINTS) if c <= MAX_TOKENS]

# Bit-width is the second experimental axis: bf16 (uncompressed) -> 8bit -> 4bit.
# Columns in the summary tables are always shown in this order, regardless of the
# per-model run order below.
DISPLAY_BITWIDTHS = ("bf16", "8bit", "4bit")

# One entry per model. `available_layers` is the registry-verified set of layers
# with a trained width-16k/l0-medium residual SAE; `layer` is a middle layer
# chosen FROM that set (not n_layers/2). See module docstring for provenance.
#
# `run_bitwidths` is the order conditions are *executed* in for that model. For
# both 4B and 12B all three bit-widths fit a 40GB A100 in any order, so every
# condition in the matrix is expected to produce numbers.
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
]

def _metrics_from_sums(acc: dict) -> dict:
    """Reduce accumulated reconstruction SUMS into the headline metrics.

    This is the ONE place the ratios are formed — after every chunk's sums have
    been added up. FVU is numerator/denominator divided ONCE here, never an
    average of per-chunk FVUs:

      fvu_denominator = var_sum = sum_x2 - sum_x**2 / n_elements  (total squared
                        deviation from the global scalar mean over ALL elements)
      fvu             = sse / var_sum            (== concatenate-everything FVU)
      mse             = sse / n_elements
      l0              = active_sum / n_tokens
    """
    n_elem = acc["n_elements"]
    var_sum = acc["sum_x2"] - acc["sum_x"] ** 2 / n_elem
    return {
        "fvu": acc["sse"] / var_sum,
        "mse": acc["sse"] / n_elem,
        "l0": acc["active_sum"] / acc["n_tokens"],
        "fvu_denominator": var_sum,
        "tokens": acc["n_tokens"],
    }


def run_condition(loaded_sae, name: str, precision: str, layer: int, dataset) -> dict | None:
    """Run one (model x bit-width) condition over the streamed corpus.

    Loads `name` at `precision`, then streams the Pile `dataset` packed into
    fixed-length token chunks, capturing the residual stream at `layer` and
    accumulating reconstruction SUMS one chunk at a time with the ALREADY-LOADED
    `loaded_sae` (the fixed instrument — never reloaded or varied per bit-width).

    FVU is accumulated as a ratio of sums (numerator = total squared error,
    denominator = total variance sum) and divided ONCE at the end — averaging
    per-chunk FVUs would be statistically wrong. Each activation chunk is freed
    immediately after its sums are accumulated, so memory stays flat regardless
    of how many tokens are processed.

    Snapshots FVU/L0/MSE at each STABILITY_CHECKPOINTS token count (those within
    the MAX_TOKENS budget) so the metric can be watched stabilizing; processes up
    to the largest such checkpoint. Returns the HEADLINE metrics (at N_TOKENS)
    plus a `stability` list of per-checkpoint snapshots, or None on a d_in
    mismatch. The model is always freed in the finally block; OOM/other
    exceptions propagate after cleanup.
    """
    loaded = None
    try:
        # --- load model at this precision --------------------------------
        print(f"[1/2] Loading {name} at {precision}...")
        loaded = load_model(name, precision=precision, compute_dtype=torch.bfloat16)
        print("      done.")

        d_in = loaded_sae.sae.cfg.d_in
        checkpoints = _active_checkpoints()  # honors the MAX_TOKENS budget
        target_tokens = min(MAX_TOKENS, checkpoints[-1]) if checkpoints else MAX_TOKENS

        print(
            f"[2/2] Streaming residual stream at layer {layer} in "
            f"{BATCH_SIZE}x{CHUNK_LEN}-token chunks up to {target_tokens} tokens..."
        )

        # Accumulators (kept as the numerator/denominator building blocks; we
        # NEVER average per-chunk FVUs). All reduced in float64 inside the chunk
        # helper, then summed here.
        acc = {
            "sse": 0.0,
            "sum_x": 0.0,
            "sum_x2": 0.0,
            "n_elements": 0,
            "n_tokens": 0,
            "active_sum": 0.0,
        }
        snapshots: list[dict] = []
        next_ckpt = 0  # index into `checkpoints` of the next milestone to record
        verified = False

        batches = iter_token_batches(
            loaded.tokenizer, dataset, CHUNK_LEN, BATCH_SIZE, target_tokens
        )
        for input_ids in batches:
            acts = capture_resid_from_input_ids(loaded, input_ids, layer)

            # d_in sanity check on the first chunk: a mismatch is the silent-
            # wrong-answer trap. Skip the whole condition rather than reporting a
            # meaningless FVU.
            if not verified:
                d_model = acts.shape[-1]
                match = d_model == d_in
                print(f"      first chunk shape: {tuple(input_ids.shape)} -> "
                      f"resid {tuple(acts.shape)} (d_model={d_model})")
                print(f"      SAE sae_id={loaded_sae.sae_id}  d_in={d_in}  match={match}")
                if not match:
                    print(
                        f"  [!] MISMATCH: model d_model={d_model} != SAE d_in={d_in}. "
                        f"Skipping {name} @ {precision}."
                    )
                    del acts
                    return None
                verified = True

            # Accumulate this chunk's sums, then FREE the activation chunk — the
            # per-chunk activations are the memory risk now, not the model.
            sums = encode_decode_sums(loaded_sae.sae, acts)
            del acts
            for key in acc:
                acc[key] += sums[key]

            # Record a stability snapshot each time we cross a checkpoint.
            while next_ckpt < len(checkpoints) and acc["n_tokens"] >= checkpoints[next_ckpt]:
                snap = _metrics_from_sums(acc)
                snap["target"] = checkpoints[next_ckpt]
                snapshots.append(snap)
                print(
                    f"      ~{checkpoints[next_ckpt]:>6} tok (actual {snap['tokens']:>6}): "
                    f"FVU={snap['fvu']:.4f}  L0={snap['l0']:.1f}"
                )
                next_ckpt += 1
            if next_ckpt >= len(checkpoints):
                break

        if acc["n_tokens"] == 0:
            print(f"  [!] No tokens streamed for {name} @ {precision}.")
            return None

        # If the stream ran dry before some checkpoints, record them at whatever
        # was reached so the headline still resolves (won't happen for pile-10k).
        while next_ckpt < len(checkpoints):
            snap = _metrics_from_sums(acc)
            snap["target"] = checkpoints[next_ckpt]
            snapshots.append(snap)
            next_ckpt += 1

        # Headline = the snapshot at N_TOKENS (the configured sample size).
        headline = next(
            (s for s in snapshots if s["target"] == N_TOKENS), snapshots[-1]
        )
        print(f"  --- headline @ {headline['tokens']} tokens ---")
        print(f"  tokens evaluated:           {headline['tokens']}")
        print(f"  reconstruction MSE:         {headline['mse']:.6f}")
        print(f"  FVU denominator (var sum):  {headline['fvu_denominator']:.4f}")
        print(f"  FVU (1 - R^2):              {headline['fvu']:.4f}")
        print(f"  mean L0 (active feats):     {headline['l0']:.1f}")
        return {
            "fvu": headline["fvu"],
            "l0": headline["l0"],
            "mse": headline["mse"],
            "fvu_denominator": headline["fvu_denominator"],
            "tokens": headline["tokens"],
            "stability": snapshots,
        }
    finally:
        # Free GPU memory between EVERY condition — each bit-width is a fresh
        # model load, and a lingering previous model would cause a spurious OOM
        # on a condition that should fit. Runs on success AND on exception. The
        # SAE is intentionally NOT freed here; it persists across this model's
        # bit-widths and is freed by the caller when moving to the next model.
        del loaded
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
    print(f"  corpus:    streamed Pile, packed into {CHUNK_LEN}-token chunks"
          + (f", shuffled seed={SHUFFLE_SEED} buf={SHUFFLE_BUFFER}" if SHUFFLE_SEED >= 0 else ", unshuffled"))
    print(f"  sample:    headline N_TOKENS={N_TOKENS}  MAX_TOKENS={MAX_TOKENS}")
    print(f"  stability: {', '.join(str(c) for c in _active_checkpoints())}"
          + (f"  (checkpoints > MAX_TOKENS skipped)" if _active_checkpoints() != list(STABILITY_CHECKPOINTS) else ""))
    print(f"  measure:   SAE={MEASURE_SAE}  NOISE_FLOOR={MEASURE_NOISE_FLOOR}")
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
        # The matrix is sized for a 40GB A100: all 6 conditions fit, the largest
        # single load being 12B bf16 (~24GB). On a smaller GPU (e.g. a T4),
        # 12B bf16 will likely OOM.
        large_gpu = any(
            tag in gpu_name for tag in ("A100", "H100", "H200", "A6000", "L40", "80GB")
        )
        if not large_gpu:
            print(
                "[!] This does not look like an A100/large GPU. Expect OOM on\n"
                "    bf16 12B (~24GB), which needs a large GPU. Smaller conditions\n"
                "    (4B all precisions, 12B 8bit/4bit) should still report.\n"
                "    On Colab Pro+: Runtime -> Change runtime type -> A100."
            )

    # Open the streamed corpus ONCE, shuffled with a FIXED seed (deterministic).
    # The same dataset object is re-iterated for every condition (streaming
    # datasets restart on re-iteration), so all conditions — FVU and noise-floor
    # alike — see identical tokens, keeping the matrix directly comparable.
    print("\nOpening streamed Pile corpus...")
    shuffle_seed = SHUFFLE_SEED if SHUFFLE_SEED >= 0 else None
    dataset, dataset_source = load_pile_stream(
        shuffle_seed=shuffle_seed, shuffle_buffer=SHUFFLE_BUFFER
    )
    print(f"  dataset: {dataset_source} (streaming=True, "
          f"{'shuffled seed=%d' % shuffle_seed if shuffle_seed is not None else 'unshuffled'})")

    # results[name][precision]       -> SAE FVU metrics dict / status string.
    # noise_results[name][precision] -> perturbation metrics dict / status string.
    results: dict[str, dict[str, object]] = {}
    noise_results: dict[str, dict[str, object]] = {}

    for model_cfg in MODEL_CONFIGS:
        name = model_cfg["model_name"]
        layer = model_cfg["layer"]
        release = model_cfg["sae_release"]
        available_layers = tuple(model_cfg["available_layers"])
        results.setdefault(name, {})
        noise_results.setdefault(name, {})

        print("\n" + "=" * 70)
        print(f"MODEL: {name}  (layer {layer})")
        print("=" * 70)

        # Load the SAE ONCE per model and reuse it across all its bit-widths. The
        # SAE is loaded in ALL modes (the noise-floor control also needs it for
        # the reconstruction-residual comparison), never gated on MEASURE_SAE.
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
                noise_results[name][precision] = "SAE-FAIL"
            continue

        try:
            # --- Measurement 1: SAE reconstruction FVU/L0 (existing path) ----
            if MEASURE_SAE:
                for precision in model_cfg["run_bitwidths"]:
                    print(f"\n--- [SAE] {name} @ {precision} ---")
                    try:
                        metrics = run_condition(loaded_sae, name, precision, layer, dataset)
                        results[name][precision] = metrics if metrics is not None else "MISMATCH"
                    except Exception as exc:  # noqa: BLE001 — keep the matrix alive
                        if _is_oom(exc):
                            print(f"\n  SKIPPED (out of memory): {name} @ {precision}")
                            results[name][precision] = "N/A (OOM)"
                        else:
                            print(f"\n  FAILED: {type(exc).__name__}: {exc}")
                            results[name][precision] = "FAIL"
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            # --- Measurement 2: noise-floor / activation perturbation -------
            # Separable layer: only runs when enabled, and is NOT woven into the
            # FVU path above. bf16 is the reference; each quantized model is
            # compared against it on the SAME deterministic tokens.
            if MEASURE_NOISE_FLOOR:
                _run_noise_floor(loaded_sae, name, layer, dataset, noise_results[name])
        finally:
            # Free the SAE when moving to the next model.
            del loaded_sae
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if MEASURE_SAE:
        _print_stability_tables(results)
        _print_gradient_tables(results)
    if MEASURE_NOISE_FLOOR:
        _print_noise_floor_tables(noise_results)
    print("\nExperiment complete.")


def _run_noise_floor(loaded_sae, name: str, layer: int, dataset, out: dict) -> None:
    """Measure activation perturbation of each quantized model vs bf16 reference.

    Loads the bf16 reference ONCE, then for 8bit and 4bit loads the quantized
    model, streams the SAME deterministic tokens (re-iterating `dataset`),
    captures both models' residual streams per chunk, and accumulates the
    perturbation sums plus the SAE residual over identical tokens. Results (or a
    status string on OOM/failure) are written into `out[precision]`. The bf16
    reference and each quantized model are freed promptly; only running sums are
    held, never a full-corpus activation cache.
    """
    ref_model = None
    try:
        print(f"\n--- [NOISE-FLOOR] {name}: loading bf16 reference ---")
        ref_model = load_model(name, precision="bf16", compute_dtype=torch.bfloat16)
        print("      bf16 reference loaded.")
    except Exception as exc:  # noqa: BLE001 — keep the matrix alive
        status = "N/A (OOM)" if _is_oom(exc) else "FAIL"
        print(f"  bf16 reference load {status}: {type(exc).__name__}: {exc}")
        for precision in ("8bit", "4bit"):
            out[precision] = status
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    try:
        for precision in ("8bit", "4bit"):
            print(f"\n--- [NOISE-FLOOR] {name} @ {precision} vs bf16 ---")
            test_model = None
            try:
                test_model = load_model(
                    name, precision=precision, compute_dtype=torch.bfloat16
                )
                metrics = measure_perturbation(
                    loaded_sae, ref_model, test_model, dataset, layer,
                    CHUNK_LEN, BATCH_SIZE, N_TOKENS,
                )
                out[precision] = metrics
                print(
                    f"  {precision}: activation perturbation (rel L2) = {metrics['pert_rel_l2']:.4f} ; "
                    f"SAE reconstruction residual (rel) = {metrics['sae_rel_residual']:.4f} ; "
                    f"perturbation/residual ratio = {metrics['ratio']:.3f}"
                )
                print(
                    f"        mean per-token L2 = {metrics['mean_token_l2']:.4f} ; "
                    f"pert MSE = {metrics['pert_mse']:.6f} ; SAE MSE = {metrics['sae_mse']:.6f} ; "
                    f"tokens = {metrics['tokens']}"
                )
            except Exception as exc:  # noqa: BLE001
                status = "N/A (OOM)" if _is_oom(exc) else "FAIL"
                print(f"  {precision} noise-floor {status}: {type(exc).__name__}: {exc}")
                out[precision] = status
            finally:
                del test_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        del ref_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cell(entry: object, key: str) -> str:
    """Format one table cell: a metric value, or a status string as-is."""
    if isinstance(entry, dict):
        return f"{entry[key]:.4f}" if key == "fvu" else f"{entry[key]:.1f}"
    if entry is None:
        return "-"
    return str(entry)  # "N/A (OOM)", "FAIL", "MISMATCH", "SAE-FAIL"


def _fmt_tokens(n: int) -> str:
    """Compact token-count label: 5000 -> '5k', 100000 -> '100k'."""
    return f"{n // 1000}k" if n % 1000 == 0 and n >= 1000 else str(n)


def _print_stability_tables(results: dict) -> None:
    """Print a per-model 'FVU vs token-count' table (rows = bit-widths).

    This is the stability evidence: read each row left-to-right and FVU should
    stop moving as the token count grows, proving the N_TOKENS sample is large
    enough for a believable reconstruction number. Columns are the configured
    checkpoints within the MAX_TOKENS budget; cells are FVU at that many tokens
    (or a status string if the condition didn't produce numbers).
    """
    checkpoints = _active_checkpoints()
    col_w = 10

    print("\n" + "=" * 70)
    print("STABILITY — FVU vs token-count (should flatten as tokens grow)")
    print("=" * 70)

    header = "  {:<8}".format("bits") + "".join(
        "{:>{w}}".format(_fmt_tokens(c), w=col_w) for c in checkpoints
    )
    for model_cfg in MODEL_CONFIGS:
        name = model_cfg["model_name"]
        print(f"\n{name.replace('google/', '')}")
        print(header)
        for bw in DISPLAY_BITWIDTHS:
            entry = results.get(name, {}).get(bw)
            row = "  {:<8}".format(bw)
            if isinstance(entry, dict):
                by_target = {s["target"]: s for s in entry.get("stability", [])}
                for c in checkpoints:
                    snap = by_target.get(c)
                    cell = f"{snap['fvu']:.4f}" if snap else "-"
                    row += "{:>{w}}".format(cell, w=col_w)
            else:
                # Status string (OOM / FAIL / MISMATCH / SAE-FAIL) or missing.
                status = "-" if entry is None else str(entry)
                for _ in checkpoints:
                    row += "{:>{w}}".format(status, w=col_w)
            print(row)


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


def _print_noise_floor_tables(noise_results: dict) -> None:
    """Print the activation-perturbation control vs the SAE reconstruction residual.

    For each (model x quantized bit-width) prints the relative-L2 activation
    perturbation, mean per-token L2, the SAE relative reconstruction residual,
    and their ratio — all over identical tokens. A per-model summary line states
    whether the quantization perturbation is LARGER or SMALLER than the SAE
    residual, i.e. whether the flat-FVU result is informative (perturbation rises
    to / above the instrument's resolution) or below it.
    """
    quant_bitwidths = ("8bit", "4bit")
    col_w = 14

    print("\n" + "=" * 70)
    print("NOISE FLOOR — activation perturbation vs SAE reconstruction residual")
    print("=" * 70)
    print("  (perturbation = how much quantization moves the residual stream;")
    print("   ratio = perturbation / SAE residual. ratio >= ~1 -> flat FVU is")
    print("   informative; ratio << 1 -> change is below the SAE's resolution.)")

    header = "  {:<8}".format("bits") + "".join(
        "{:>{w}}".format(h, w=col_w)
        for h in ("pert relL2", "tok L2", "SAE relL2", "pert/SAE")
    )
    for model_cfg in MODEL_CONFIGS:
        name = model_cfg["model_name"]
        print(f"\n{name.replace('google/', '')}")
        print(header)
        ratios = []
        for bw in quant_bitwidths:
            entry = noise_results.get(name, {}).get(bw)
            row = "  {:<8}".format(bw)
            if isinstance(entry, dict):
                cells = (
                    f"{entry['pert_rel_l2']:.4f}",
                    f"{entry['mean_token_l2']:.4f}",
                    f"{entry['sae_rel_residual']:.4f}",
                    f"{entry['ratio']:.3f}",
                )
                ratios.append(entry["ratio"])
            else:
                status = "-" if entry is None else str(entry)
                cells = (status, status, status, status)
            for c in cells:
                row += "{:>{w}}".format(c, w=col_w)
            print(row)

        # Per-model verdict from whichever quantized conditions produced numbers.
        if ratios:
            min_ratio, max_ratio = min(ratios), max(ratios)
            if min_ratio >= 1.0:
                verdict = (
                    "perturbation >= SAE residual -> flat FVU is INFORMATIVE "
                    "(representation is genuinely perturbed, SAE view survives)"
                )
            elif max_ratio < 0.1:
                verdict = (
                    "perturbation << SAE residual -> flat FVU is UNINFORMATIVE "
                    "(change is below the SAE's resolution)"
                )
            else:
                verdict = (
                    "perturbation is COMPARABLE TO / below SAE residual "
                    f"(ratio {min_ratio:.2f}-{max_ratio:.2f}) -> interpret with care"
                )
            print(f"  => {verdict}")


if __name__ == "__main__":
    main()
