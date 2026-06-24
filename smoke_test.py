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
from datetime import datetime
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch  # noqa: E402

from interp.corpus import iter_token_batches, load_pile_stream  # noqa: E402
from interp.model_loading import capture_resid_from_input_ids, load_model  # noqa: E402
from interp.noise_floor import measure_perturbation  # noqa: E402
from interp.performance import cross_entropy_sums, finalize_cross_entropy  # noqa: E402
from interp.results_io import build_results_document, write_results  # noqa: E402
from interp.sae_loading import encode_decode_sums, load_residual_sae  # noqa: E402


_TRUE_STRINGS = ("1", "true", "yes", "on")
_FALSE_STRINGS = ("0", "false", "no", "off")


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from the environment, else `default`.

    Accepts only explicit truthy/falsey spellings (case-insensitive); an
    unrecognized value fails fast rather than silently being treated as False.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUE_STRINGS:
        return True
    if value in _FALSE_STRINGS:
        return False
    raise ValueError(
        f"Invalid boolean for {name}={raw!r}; expected one of "
        f"{_TRUE_STRINGS + _FALSE_STRINGS}"
    )


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, else `default`.

    A non-integer value fails fast with a message naming the variable, rather
    than surfacing a bare `int()` ValueError.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f"Invalid integer for {name}={raw!r}") from None


# --- Measurement toggles (set in the first Colab cell, or via env var) -----
# Each measurement runs only if enabled. With MEASURE_NOISE_FLOOR off the behavior
# is exactly the original FVU path: no extra compute, no bf16-reference passes.
MEASURE_SAE = _env_bool("MEASURE_SAE", True)
MEASURE_NOISE_FLOOR = _env_bool("MEASURE_NOISE_FLOOR", True)
# Model-performance proxy: next-token cross-entropy / perplexity of each
# (model x bit-width) on the SAME streamed tokens. Reuses the activation-capture
# forward pass (no separate forward) and runs INDEPENDENTLY of MEASURE_SAE.
MEASURE_MODEL_PERFORMANCE = _env_bool("MEASURE_MODEL_PERFORMANCE", True)
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


def _headline_target() -> int:
    """Token count the HEADLINE metrics are reported at.

    This is N_TOKENS, capped to the MAX_TOKENS budget: if the configured headline
    exceeds the budget, the most we can stream is MAX_TOKENS, so that becomes the
    headline. Always a reachable, positive target.
    """
    return min(N_TOKENS, MAX_TOKENS)


def _active_checkpoints() -> list[int]:
    """Sorted stability checkpoints within the MAX_TOKENS budget.

    Always includes the headline target (see `_headline_target`) so the headline
    snapshot is recorded even when N_TOKENS is not one of the fixed
    STABILITY_CHECKPOINTS or the budget is smaller than the smallest checkpoint.
    Guaranteed non-empty.
    """
    pts = {c for c in STABILITY_CHECKPOINTS if c <= MAX_TOKENS}
    pts.add(_headline_target())
    return sorted(pts)

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


def _finalize_sae_result(
    measure_sae: bool, sae_status, snapshots: list, acc: dict,
    headline_target: int, name: str, precision: str,
):
    """Reduce the SAE accumulators into the headline result (or a status string)."""
    if not measure_sae:
        return None
    if sae_status is not None:  # e.g. "MISMATCH"
        return sae_status
    if acc["n_tokens"] == 0:
        print(f"  [!] No SAE tokens streamed for {name} @ {precision}.")
        return "NO-DATA"

    # Headline = the snapshot at the headline target (it is one of the
    # checkpoints, so normally recorded during the loop). Fall back to the metrics
    # at the most tokens we got if the stream ran dry — never assume a snapshot.
    headline = next((s for s in snapshots if s["target"] == headline_target), None)
    if headline is None:
        headline = _metrics_from_sums(acc)
        headline["target"] = headline_target
        print(f"  [!] SAE stream reached only {acc['n_tokens']} tokens "
              f"(< headline {headline_target}); reporting at that count.")
    print(f"  --- SAE headline @ {headline['tokens']} tokens ---")
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


def _finalize_perf_result(
    measure_perf: bool, perf_headline, perf_acc: dict,
    headline_target: int, name: str, precision: str,
):
    """Reduce the cross-entropy accumulators into the result (or a status string)."""
    if not measure_perf:
        return None
    if perf_acc["n_tokens"] == 0:
        print(f"  [!] No performance tokens streamed for {name} @ {precision}.")
        return "NO-DATA"
    result = perf_headline
    if result is None:
        # Stream ran dry before the headline target; report at what we got.
        result = finalize_cross_entropy(perf_acc["sum_nll"], perf_acc["n_tokens"])
        print(f"  [!] Perf stream reached only {perf_acc['n_tokens']} predicted "
              f"tokens (< headline {headline_target}); reporting at that count.")
    print(f"  --- performance @ {result['tokens']} predicted tokens ---")
    print(f"  cross-entropy (nats/tok):   {result['cross_entropy']:.4f}")
    print(f"  perplexity:                 {result['perplexity']:.3f}")
    return result


def run_condition(
    loaded_sae,
    name: str,
    precision: str,
    layer: int,
    dataset,
    measure_sae: bool,
    measure_perf: bool,
) -> dict:
    """Run one (model x bit-width) condition over the streamed corpus.

    A SINGLE forward pass per chunk serves both enabled measurements — there is no
    separate forward for performance:
      * SAE reconstruction FVU/L0 (when `measure_sae`): accumulated as a ratio of
        SUMS (numerator = total squared error, denominator = total variance sum),
        divided ONCE at the end; FVU/L0 snapshotted at each stability checkpoint.
      * Model performance (when `measure_perf`): next-token cross-entropy read from
        the SAME forward's output logits, accumulated TOKEN-WEIGHTED (total nll /
        total predicted tokens), divided ONCE at the end.

    The forward runs whenever EITHER measurement is enabled, so performance works
    even with `measure_sae` False — it is never gated behind the SAE. A d_in
    mismatch disables only the SAE half; performance still completes. Each chunk's
    activations and logits are freed right after their sums are accumulated, so
    memory stays flat regardless of token count.

    Returns {"sae": <result>, "performance": <result>} where each <result> is a
    metrics dict, a status string ("MISMATCH" / "NO-DATA"), or None if that
    measurement was disabled. The model is always freed in the finally block;
    OOM/other exceptions propagate after cleanup.
    """
    loaded = None
    try:
        # --- load model at this precision --------------------------------
        print(f"[1/2] Loading {name} at {precision}...")
        loaded = load_model(name, precision=precision, compute_dtype=torch.bfloat16)
        print("      done.")

        d_in = loaded_sae.sae.cfg.d_in if measure_sae else None
        checkpoints = _active_checkpoints()  # honors the MAX_TOKENS budget
        headline_target = _headline_target()
        # SAE needs the full stability sweep (up to the largest checkpoint);
        # performance only needs the headline budget. Stream to whichever the
        # enabled measurements require.
        sae_target = min(MAX_TOKENS, checkpoints[-1])
        target_tokens = sae_target if measure_sae else headline_target

        print(
            f"[2/2] Streaming layer {layer} in {BATCH_SIZE}x{CHUNK_LEN}-token chunks "
            f"up to {target_tokens} tokens  (SAE={measure_sae}, perf={measure_perf})..."
        )

        # SAE accumulators (numerator/denominator blocks; never average per-chunk
        # FVUs). All reduced in float64 inside the chunk helper, summed here.
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
        sae_active = measure_sae
        sae_status = None       # set to "MISMATCH" if the d_in check fails
        checked_d_in = False

        # Performance accumulators (token-weighted nll). `perf_headline` is the
        # snapshot taken when cumulative input tokens reach the headline target.
        perf_acc = {"sum_nll": 0.0, "n_tokens": 0}
        perf_headline = None
        total_input_tokens = 0

        batches = iter_token_batches(
            loaded.tokenizer, dataset, CHUNK_LEN, BATCH_SIZE, target_tokens
        )
        for input_ids in batches:
            if measure_perf:
                acts, logits = capture_resid_from_input_ids(
                    loaded, input_ids, layer, return_logits=True
                )
            else:
                acts = capture_resid_from_input_ids(loaded, input_ids, layer)
                logits = None
            total_input_tokens += int(input_ids.numel())

            # SAE d_in sanity check on the first chunk: a mismatch means the SAE
            # can't be applied — skip the SAE half, but DO NOT abort performance.
            if sae_active and not checked_d_in:
                d_model = acts.shape[-1]
                match = d_model == d_in
                print(f"      first chunk shape: {tuple(input_ids.shape)} -> "
                      f"resid {tuple(acts.shape)} (d_model={d_model})")
                print(f"      SAE sae_id={loaded_sae.sae_id}  d_in={d_in}  match={match}")
                if not match:
                    print(f"  [!] MISMATCH: model d_model={d_model} != SAE d_in={d_in}. "
                          f"Skipping SAE for {name} @ {precision}.")
                    sae_active = False
                    sae_status = "MISMATCH"
                checked_d_in = True

            # SAE: accumulate this chunk's sums.
            if sae_active:
                sums = encode_decode_sums(loaded_sae.sae, acts)
                for key in acc:
                    acc[key] += sums[key]
            del acts

            # Performance: accumulate token-weighted nll from the SAME forward.
            if measure_perf:
                ce = cross_entropy_sums(logits, input_ids)
                del logits
                perf_acc["sum_nll"] += ce["sum_nll"]
                perf_acc["n_tokens"] += ce["n_tokens"]
                if perf_headline is None and total_input_tokens >= headline_target:
                    perf_headline = finalize_cross_entropy(
                        perf_acc["sum_nll"], perf_acc["n_tokens"]
                    )

            # SAE stability snapshots at each crossed checkpoint.
            if sae_active:
                while next_ckpt < len(checkpoints) and acc["n_tokens"] >= checkpoints[next_ckpt]:
                    snap = _metrics_from_sums(acc)
                    snap["target"] = checkpoints[next_ckpt]
                    snapshots.append(snap)
                    print(f"      ~{checkpoints[next_ckpt]:>6} tok (actual {snap['tokens']:>6}): "
                          f"FVU={snap['fvu']:.4f}  L0={snap['l0']:.1f}")
                    next_ckpt += 1

            # Stop once every enabled measurement has what it needs.
            sae_done = (not sae_active) or (next_ckpt >= len(checkpoints))
            perf_done = (not measure_perf) or (perf_headline is not None)
            if sae_done and perf_done:
                break

        return {
            "sae": _finalize_sae_result(
                measure_sae, sae_status, snapshots, acc, headline_target, name, precision
            ),
            "performance": _finalize_perf_result(
                measure_perf, perf_headline, perf_acc, headline_target, name, precision
            ),
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


def _validate_config() -> None:
    """Reject token-budget settings that would silently produce misleading runs.

    N_TOKENS / MAX_TOKENS are user-configurable (env or top-of-file); a zero or
    negative value must fail fast with a clear message rather than yield a
    no-token or empty-checkpoint result. CHUNK_LEN / BATCH_SIZE must be positive
    so each forward actually carries tokens.
    """
    problems = []
    if N_TOKENS <= 0:
        problems.append(f"N_TOKENS must be > 0 (got {N_TOKENS})")
    if MAX_TOKENS <= 0:
        problems.append(f"MAX_TOKENS must be > 0 (got {MAX_TOKENS})")
    if CHUNK_LEN <= 0:
        problems.append(f"CHUNK_LEN must be > 0 (got {CHUNK_LEN})")
    if BATCH_SIZE <= 0:
        problems.append(f"BATCH_SIZE must be > 0 (got {BATCH_SIZE})")
    if problems:
        raise ValueError("Invalid token-budget configuration: " + "; ".join(problems))


def main() -> None:
    _validate_config()
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
    print(f"  measure:   SAE={MEASURE_SAE}  NOISE_FLOOR={MEASURE_NOISE_FLOOR}  "
          f"MODEL_PERFORMANCE={MEASURE_MODEL_PERFORMANCE}")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_name = None
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
    # perf_results[name][precision]  -> cross-entropy metrics dict / status string.
    results: dict[str, dict[str, object]] = {}
    noise_results: dict[str, dict[str, object]] = {}
    perf_results: dict[str, dict[str, object]] = {}

    for model_cfg in MODEL_CONFIGS:
        name = model_cfg["model_name"]
        layer = model_cfg["layer"]
        release = model_cfg["sae_release"]
        available_layers = tuple(model_cfg["available_layers"])
        results.setdefault(name, {})
        noise_results.setdefault(name, {})
        perf_results.setdefault(name, {})

        print("\n" + "=" * 70)
        print(f"MODEL: {name}  (layer {layer})")
        print("=" * 70)

        # Load the SAE ONCE per model and reuse it across all its bit-widths. The
        # SAE is loaded whenever it could be needed (SAE measurement or the
        # noise-floor control); a load failure disables only those — performance
        # does NOT depend on the SAE and still runs.
        loaded_sae = None
        sae_load_failed = False
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
            sae_load_failed = True
            if MEASURE_SAE:
                for precision in DISPLAY_BITWIDTHS:
                    results[name][precision] = "SAE-FAIL"
            if MEASURE_NOISE_FLOOR:
                for precision in ("8bit", "4bit"):
                    noise_results[name][precision] = "SAE-FAIL"

        # SAE measurement only runs if the SAE actually loaded.
        eff_measure_sae = MEASURE_SAE and not sae_load_failed

        try:
            # --- Forward-based measurements: SAE recon + model performance ----
            # ONE forward per chunk serves both. Runs if either is enabled; the
            # forward (and performance) is never gated behind the SAE.
            if eff_measure_sae or MEASURE_MODEL_PERFORMANCE:
                for precision in model_cfg["run_bitwidths"]:
                    print(f"\n--- [FWD] {name} @ {precision} ---")
                    try:
                        res = run_condition(
                            loaded_sae, name, precision, layer, dataset,
                            measure_sae=eff_measure_sae,
                            measure_perf=MEASURE_MODEL_PERFORMANCE,
                        )
                        if eff_measure_sae:
                            results[name][precision] = res["sae"]
                        if MEASURE_MODEL_PERFORMANCE:
                            perf_results[name][precision] = res["performance"]
                    except Exception as exc:  # noqa: BLE001 — keep the matrix alive
                        status = "N/A (OOM)" if _is_oom(exc) else "FAIL"
                        if _is_oom(exc):
                            print(f"\n  SKIPPED (out of memory): {name} @ {precision}")
                        else:
                            print(f"\n  FAILED: {type(exc).__name__}: {exc}")
                        if eff_measure_sae:
                            results[name][precision] = status
                        if MEASURE_MODEL_PERFORMANCE:
                            perf_results[name][precision] = status
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                # Deltas vs each model's bf16 reference (bf16 ran first).
                if MEASURE_MODEL_PERFORMANCE:
                    _fill_perf_deltas(perf_results[name])

            # --- Noise-floor / activation perturbation (needs the SAE) -------
            # Separable layer: only runs when enabled and the SAE loaded. bf16 is
            # the reference; each quantized model is compared on the SAME tokens.
            if MEASURE_NOISE_FLOOR and not sae_load_failed:
                _run_noise_floor(loaded_sae, name, layer, dataset, noise_results[name])
        finally:
            # Free the SAE when moving to the next model.
            del loaded_sae
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Console summaries (additive; unchanged when a measurement is off) ---
    if MEASURE_SAE:
        _print_stability_tables(results)
        _print_gradient_tables(results)
    if MEASURE_NOISE_FLOOR:
        _print_noise_floor_tables(noise_results)
    if MEASURE_MODEL_PERFORMANCE:
        _print_performance_tables(perf_results)

    # --- Persist all enabled measurements to disk (additive) -----------------
    metadata = {
        "timestamp": timestamp,
        "models": [m["model_name"] for m in MODEL_CONFIGS],
        "bit_widths": list(DISPLAY_BITWIDTHS),
        "n_tokens": N_TOKENS,
        "max_tokens": MAX_TOKENS,
        "headline_target": _headline_target(),
        "shuffle_seed": shuffle_seed,
        "shuffle_buffer": SHUFFLE_BUFFER,
        "dataset": dataset_source,
        "gpu": gpu_name,
        "measurements": {
            "sae": MEASURE_SAE,
            "noise_floor": MEASURE_NOISE_FLOOR,
            "model_performance": MEASURE_MODEL_PERFORMANCE,
        },
    }
    document = build_results_document(
        metadata,
        results if MEASURE_SAE else None,
        noise_results if MEASURE_NOISE_FLOOR else None,
        perf_results if MEASURE_MODEL_PERFORMANCE else None,
        MODEL_CONFIGS,
        DISPLAY_BITWIDTHS,
    )
    run_path, latest_path = write_results(document, timestamp)
    print(f"\nResults written to:\n  {run_path}\n  {latest_path}")

    print("\nExperiment complete.")


def _fill_perf_deltas(perf_by_bitwidth: dict) -> None:
    """Fill delta-vs-bf16 fields on each performance record (in place).

    bf16 is the reference precision: its delta is 0.0; each quantized condition's
    delta is its cross-entropy minus the bf16 cross-entropy (absolute and percent).
    Records that are status strings, or any condition when bf16 is unavailable, get
    None deltas.
    """
    bf16 = perf_by_bitwidth.get("bf16")
    base_ce = bf16["cross_entropy"] if isinstance(bf16, dict) else None
    for bw, entry in perf_by_bitwidth.items():
        if not isinstance(entry, dict):
            continue
        if bw == "bf16":
            entry["delta_vs_bf16_abs"] = 0.0
            entry["delta_vs_bf16_pct"] = 0.0
        elif base_ce is None:
            entry["delta_vs_bf16_abs"] = None
            entry["delta_vs_bf16_pct"] = None
        else:
            delta = entry["cross_entropy"] - base_ce
            entry["delta_vs_bf16_abs"] = delta
            entry["delta_vs_bf16_pct"] = delta / base_ce * 100.0


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
                # Use the SAME effective budget as the SAE headline (N_TOKENS
                # capped to MAX_TOKENS) so both measurements cover identical
                # tokens and a capped run stays cheap.
                metrics = measure_perturbation(
                    loaded_sae, ref_model, test_model, dataset, layer,
                    CHUNK_LEN, BATCH_SIZE, _headline_target(),
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


def _print_performance_tables(perf_results: dict) -> None:
    """Print model-performance tables: cross-entropy, perplexity, and delta vs bf16.

    Cross-entropy is the token-weighted mean (total nll / total predicted tokens);
    perplexity = exp(cross_entropy). The delta-vs-bf16 table is the interpretable
    quantity: how much compression raised the model's loss (absolute nats and %).
    """
    quant_bitwidths = ("8bit", "4bit")

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE — cross-entropy / perplexity (lower = better)")
    print("=" * 70)

    def _metric_cell(entry: object, key: str, fmt: str) -> str:
        if isinstance(entry, dict):
            value = entry.get(key)
            return format(value, fmt) if value is not None else "-"
        return "-" if entry is None else str(entry)

    col_w = 12
    header = "  {:<16}".format("model") + "".join(
        "{:>{w}}".format(bw, w=col_w) for bw in DISPLAY_BITWIDTHS
    )

    print("\ncross-entropy (nats/token) by (model x bit-width)")
    print(header)
    for model_cfg in MODEL_CONFIGS:
        name = model_cfg["model_name"]
        row = "  {:<16}".format(name.replace("google/", ""))
        for bw in DISPLAY_BITWIDTHS:
            row += "{:>{w}}".format(
                _metric_cell(perf_results.get(name, {}).get(bw), "cross_entropy", ".4f"), w=col_w
            )
        print(row)

    print("\nperplexity by (model x bit-width)")
    print(header)
    for model_cfg in MODEL_CONFIGS:
        name = model_cfg["model_name"]
        row = "  {:<16}".format(name.replace("google/", ""))
        for bw in DISPLAY_BITWIDTHS:
            row += "{:>{w}}".format(
                _metric_cell(perf_results.get(name, {}).get(bw), "perplexity", ".3f"), w=col_w
            )
        print(row)

    print("\ndelta vs bf16 — cross-entropy rise under compression (abs nats / %)")
    dheader = "  {:<16}".format("model") + "".join(
        "{:>{w}}".format(bw, w=col_w) for bw in quant_bitwidths
    )
    print(dheader)
    for model_cfg in MODEL_CONFIGS:
        name = model_cfg["model_name"]
        row = "  {:<16}".format(name.replace("google/", ""))
        for bw in quant_bitwidths:
            entry = perf_results.get(name, {}).get(bw)
            if isinstance(entry, dict) and entry.get("delta_vs_bf16_abs") is not None:
                cell = f"{entry['delta_vs_bf16_abs']:+.4f}/{entry['delta_vs_bf16_pct']:+.2f}%"
            elif isinstance(entry, dict):
                cell = "n/a"  # no bf16 reference for this model
            else:
                cell = "-" if entry is None else str(entry)
            row += "{:>{w}}".format(cell, w=col_w + 6)
        print(row)


if __name__ == "__main__":
    main()
