"""SAE loading helpers (Gemma Scope 2 via SAELens).

The Gemma Scope 2 suite ships residual-stream SAEs for the Gemma 3 family. For
google/gemma-3-4b-pt the relevant SAELens release is `gemma-scope-2-4b-pt-res`
(repo google/gemma-scope-2-4b-pt, hook point `resid_post`). SAEs are available
at layers 9, 17, 22, 29 in widths 16k / 65k / 262k / 1m.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sae_lens import SAE

from .config import get_hf_token

# Default release for gemma-3-4b-pt residual-stream SAEs.
GEMMA_SCOPE_2_4B_RES_RELEASE = "gemma-scope-2-4b-pt-res"
# Layers that actually have a trained residual SAE in this release.
AVAILABLE_LAYERS = (9, 17, 22, 29)


def sae_id_for_layer(
    layer: int,
    width: str = "16k",
    l0: str = "medium",
    available_layers: tuple[int, ...] = AVAILABLE_LAYERS,
    release: str = GEMMA_SCOPE_2_4B_RES_RELEASE,
) -> str:
    """Build the SAELens sae_id for a residual SAE at `layer`.

    width in {16k, 65k, 262k, 1m}; l0 in {small, medium, big}. These map onto
    the naming used in the gemma-scope-2-*-pt-res releases, e.g.
    `layer_17_width_16k_l0_medium`.

    `available_layers` is the registry-verified set of layers that actually have
    a trained SAE in `release`. This differs per model (4B: 9/17/22/29, 12B:
    12/24/31/41, 27B: 16/31/40/53) — pass the correct set so a bad layer fails
    loudly here rather than 404-ing on download.
    """
    if layer not in available_layers:
        raise ValueError(
            f"No residual SAE for layer {layer} in {release}; "
            f"available layers: {available_layers}"
        )
    return f"layer_{layer}_width_{width}_l0_{l0}"


@dataclass
class LoadedSAE:
    sae: SAE
    release: str
    sae_id: str
    layer: int


def load_residual_sae(
    layer: int,
    width: str = "16k",
    l0: str = "medium",
    device: str = "cpu",
    dtype: str = "float32",
    release: str = GEMMA_SCOPE_2_4B_RES_RELEASE,
    available_layers: tuple[int, ...] = AVAILABLE_LAYERS,
) -> LoadedSAE:
    """Load a Gemma Scope 2 residual SAE at `layer` from `release`.

    Defaults target gemma-3-4b-pt. For other model sizes pass the
    registry-verified `release` (e.g. `gemma-scope-2-12b-pt-res`) and its
    `available_layers` set so the layer is validated against the right model.

    Loads onto `device` (the SAE is small enough to live on CPU even when the
    model is on GPU). Returns the SAE plus its resolved coordinates.
    """
    # Ensure the HF token is in the environment for the gated repo download.
    get_hf_token(required=True)

    sae_id = sae_id_for_layer(
        layer, width=width, l0=l0, available_layers=available_layers, release=release
    )
    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
        dtype=dtype,
    )
    return LoadedSAE(
        sae=sae,
        release=release,
        sae_id=sae_id,
        layer=layer,
    )


@torch.no_grad()
def encode_decode(sae: SAE, activations: torch.Tensor) -> dict:
    """Encode activations through the SAE and measure reconstruction quality.

    `activations` is [..., d_model]. It is cast to the SAE's dtype/device first.
    Returns feature acts, reconstruction, and several error metrics:
      - mse: mean squared error over all elements
      - l2: mean per-token L2 distance ||x - x_hat||
      - fvu: fraction of variance unexplained (1 - R^2), the standard headline
        reconstruction metric for SAEs
      - fvu_denominator: the variance used as the FVU denominator (the same
        scalar that feeds `fvu = mse / var`); reported so callers can display it
        without recomputing it
      - l0: mean number of active (non-zero) features per token
    """
    sae_param = next(sae.parameters())
    acts = activations.to(device=sae_param.device, dtype=sae_param.dtype)

    feature_acts = sae.encode(acts)
    recon = sae.decode(feature_acts)

    # Do the metric math in float32 over the flattened (num_tokens, d_model)
    # tensor. The model runs in 4-bit, so cast up to avoid dtype noise.
    d_model = acts.shape[-1]
    x = acts.reshape(-1, d_model).float()
    x_hat = recon.reshape(-1, d_model).float()
    diff = x - x_hat

    mse = diff.pow(2).mean()
    per_token_l2 = diff.norm(dim=-1).mean()
    # FVU = MSE / Var(activations). The denominator is the variance of the whole
    # activation tensor as a single scalar (unbiased=False) — NOT a per-token or
    # per-dim variance, which can collapse to ~0 and blow FVU up to ~1e19.
    var = x.var(unbiased=False)
    fvu = mse / var
    l0 = (feature_acts != 0).float().flatten(0, -2).sum(-1).mean()

    return {
        "feature_acts": feature_acts,
        "recon": recon,
        "feature_acts_shape": tuple(feature_acts.shape),
        "mse": mse.item(),
        "l2": per_token_l2.item(),
        "fvu": fvu.item(),
        "fvu_denominator": var.item(),
        "l0": l0.item(),
    }
