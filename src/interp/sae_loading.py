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
) -> str:
    """Build the SAELens sae_id for a residual SAE at `layer`.

    width in {16k, 65k, 262k, 1m}; l0 in {small, medium, big}. These map onto
    the naming used in the gemma-scope-2-4b-pt-res release, e.g.
    `layer_17_width_16k_l0_medium`.
    """
    if layer not in AVAILABLE_LAYERS:
        raise ValueError(
            f"No residual SAE for layer {layer} in {GEMMA_SCOPE_2_4B_RES_RELEASE}; "
            f"available layers: {AVAILABLE_LAYERS}"
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
) -> LoadedSAE:
    """Load the Gemma Scope 2 residual SAE matching gemma-3-4b-pt at `layer`.

    Loads onto `device` (the SAE is small enough to live on CPU even when the
    model is on GPU). Returns the SAE plus its resolved coordinates.
    """
    # Ensure the HF token is in the environment for the gated repo download.
    get_hf_token(required=True)

    sae_id = sae_id_for_layer(layer, width=width, l0=l0)
    sae = SAE.from_pretrained(
        release=GEMMA_SCOPE_2_4B_RES_RELEASE,
        sae_id=sae_id,
        device=device,
        dtype=dtype,
    )
    return LoadedSAE(
        sae=sae,
        release=GEMMA_SCOPE_2_4B_RES_RELEASE,
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
      - l0: mean number of active (non-zero) features per token
    """
    sae_param = next(sae.parameters())
    acts = activations.to(device=sae_param.device, dtype=sae_param.dtype)

    feature_acts = sae.encode(acts)
    recon = sae.decode(feature_acts)

    diff = recon - acts
    mse = diff.pow(2).mean()
    per_token_l2 = diff.flatten(0, -2).norm(dim=-1).mean()
    # FVU: residual variance over total variance (centred per the whole tensor).
    total_var = (acts - acts.mean(0, keepdim=True)).pow(2).sum()
    fvu = diff.pow(2).sum() / total_var.clamp_min(1e-12)
    l0 = (feature_acts != 0).float().flatten(0, -2).sum(-1).mean()

    return {
        "feature_acts": feature_acts,
        "recon": recon,
        "feature_acts_shape": tuple(feature_acts.shape),
        "mse": mse.item(),
        "l2": per_token_l2.item(),
        "fvu": fvu.item(),
        "l0": l0.item(),
    }
