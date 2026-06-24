"""Activation-perturbation (noise-floor) control — independent of the SAE.

The compression-gradient result is that SAE reconstruction FVU is FLAT across
bf16 / 8bit / 4bit. Before trusting that, we must rule out the trivial
explanation: maybe quantization barely moved the residual-stream activations at
all, so a flat FVU just measures "nothing happened" rather than "the SAE view
survives a real perturbation".

This control measures HOW MUCH quantization perturbs the residual stream itself,
on the SAME tokens, INDEPENDENT of the SAE, and compares it to the SAE's own
reconstruction residual:

  * If the quantization perturbation is COMPARABLE TO / LARGER THAN the SAE
    reconstruction residual and FVU still doesn't change -> strong result.
  * If the perturbation is MUCH SMALLER THAN the SAE residual -> flat FVU is
    uninformative (the change is below the instrument's resolution).

Everything is accumulated as SUMS over identical tokens and divided ONCE at the
end (never averaging per-chunk ratios, never comparing mismatched tokens):

    rel L2 perturbation = sqrt( sum (x_quant - x_bf16)^2 / sum (x_bf16)^2 )
    mean per-token L2   = sum_t ||x_quant - x_bf16|| / n_tokens
    SAE rel residual    = sqrt( sum (x_quant - x_hat)^2 / sum (x_quant)^2 )
    ratio               = rel L2 perturbation / SAE rel residual

bf16 is the REFERENCE. The reference and quantized activations are captured from
identical input_ids (the chunk is generated once and fed to both models), so the
per-token comparison is valid. Activation chunks are freed immediately after
their sums are accumulated, so only running scalar sums are ever held in memory —
no full-corpus activation cache, regardless of token budget.
"""

from __future__ import annotations

import gc
import math

import torch

from .corpus import iter_token_batches
from .model_loading import capture_resid_from_input_ids
from .sae_loading import encode_decode_sums


@torch.no_grad()
def perturbation_sums(ref_acts: torch.Tensor, test_acts: torch.Tensor) -> dict:
    """Per-chunk SUMS for the activation perturbation of `test` vs `ref`.

    `ref_acts` (bf16 reference) and `test_acts` (quantized) must be the same
    shape, captured from the SAME tokens. Reductions are done in float64 so the
    energy sums stay stable across hundreds of millions of elements. Returns:
      - pert_energy  = sum over elements of (x_test - x_ref)^2
      - ref_energy   = sum over elements of (x_ref)^2
      - sum_token_l2 = sum over tokens of ||x_test - x_ref||  (for mean per-token L2)
      - n_elements, n_tokens
    """
    d_model = ref_acts.shape[-1]
    ref = ref_acts.reshape(-1, d_model).float()
    test = test_acts.reshape(-1, d_model).float()
    diff = test - ref

    return {
        "pert_energy": diff.pow(2).double().sum().item(),
        "ref_energy": ref.double().pow(2).sum().item(),
        "sum_token_l2": diff.norm(dim=-1).double().sum().item(),
        "n_elements": ref.shape[0] * ref.shape[1],
        "n_tokens": ref.shape[0],
    }


@torch.no_grad()
def measure_perturbation(
    loaded_sae,
    ref_model,
    test_model,
    dataset,
    layer: int,
    chunk_len: int,
    batch_size: int,
    max_tokens: int,
) -> dict:
    """Accumulate the activation perturbation of `test_model` vs `ref_model`.

    For each streamed chunk, captures the residual stream at `layer` from BOTH
    models on the SAME input_ids, accumulates the perturbation sums, and ALSO
    runs the SAE on the quantized activations to accumulate the SAE's own
    reconstruction residual over the very same tokens. Each chunk's activations
    are freed before the next chunk. Divides the sums ONCE at the end.

    Returns a dict with the relative-L2 perturbation, mean per-token L2, the SAE
    relative residual, their ratio, the MSEs, and the token count — or raises on
    OOM/other errors (the caller keeps the matrix alive).
    """
    acc = {
        "pert_energy": 0.0,
        "ref_energy": 0.0,
        "sum_token_l2": 0.0,
        "n_elements": 0,
        "n_tokens": 0,
        # SAE reconstruction residual on the quantized activations (same tokens).
        "sse_sae": 0.0,
        "sumx2_sae": 0.0,
        "n_elem_sae": 0,
    }

    batches = iter_token_batches(
        test_model.tokenizer, dataset, chunk_len, batch_size, max_tokens
    )
    for input_ids in batches:
        # Capture both models on IDENTICAL tokens (chunk generated once).
        ref_acts = capture_resid_from_input_ids(ref_model, input_ids, layer)
        test_acts = capture_resid_from_input_ids(test_model, input_ids, layer)

        ps = perturbation_sums(ref_acts, test_acts)
        # SAE reconstruction residual for the quantized model on these tokens.
        ss = encode_decode_sums(loaded_sae.sae, test_acts)

        del ref_acts, test_acts

        for key in ("pert_energy", "ref_energy", "sum_token_l2", "n_elements", "n_tokens"):
            acc[key] += ps[key]
        acc["sse_sae"] += ss["sse"]
        acc["sumx2_sae"] += ss["sum_x2"]
        acc["n_elem_sae"] += ss["n_elements"]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if acc["n_tokens"] == 0:
        raise RuntimeError("no tokens streamed for the noise-floor measurement")

    pert_rel_l2 = math.sqrt(acc["pert_energy"] / acc["ref_energy"])
    sae_rel_residual = math.sqrt(acc["sse_sae"] / acc["sumx2_sae"])
    return {
        "pert_rel_l2": pert_rel_l2,
        "mean_token_l2": acc["sum_token_l2"] / acc["n_tokens"],
        "pert_mse": acc["pert_energy"] / acc["n_elements"],
        "sae_rel_residual": sae_rel_residual,
        "sae_mse": acc["sse_sae"] / acc["n_elem_sae"],
        "ratio": pert_rel_l2 / sae_rel_residual,
        "tokens": acc["n_tokens"],
    }
