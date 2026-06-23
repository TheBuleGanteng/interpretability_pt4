"""Project configuration.

Centralises environment-derived settings (notably the Hugging Face token) and
the model / SAE coordinates used by the smoke test. Keep this module free of
heavy imports (no torch / transformers) so it stays cheap to import from
anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------
# We deliberately avoid a hard dependency on python-dotenv. If it is installed
# we use it; otherwise we fall back to a tiny inline parser so `HF_TOKEN` can
# live in `.env.local` without extra packages.

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv_files() -> None:
    """Load `.env` then `.env.local` (the latter wins) into os.environ.

    Existing process environment variables always take precedence over file
    values, matching the usual dotenv convention.
    """
    for filename in (".env", ".env.local"):
        path = _REPO_ROOT / filename
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Don't clobber a value already present in the real environment.
            os.environ.setdefault(key, value)


_load_dotenv_files()


# The repo's .env.local was written as `HF_Token`; the task asked for `HF_TOKEN`.
# Accept the canonical name plus the common casings/aliases so neither spelling
# silently fails.
_HF_TOKEN_KEYS = (
    "HF_TOKEN",
    "HF_Token",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)


def get_hf_token(required: bool = True) -> str | None:
    """Return the Hugging Face access token from the environment.

    Looks up `HF_TOKEN` first, then a few common alternate spellings. When
    `required` is True and nothing is found, raises with a clear message.
    """
    for key in _HF_TOKEN_KEYS:
        value = os.environ.get(key)
        if value:
            # Normalise so downstream libs that only read HF_TOKEN see it too.
            os.environ.setdefault("HF_TOKEN", value)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", value)
            return value
    if required:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN (or HF_Token) in your "
            "environment or in .env.local at the repo root. A token with access "
            "to the gated Gemma repos is required."
        )
    return None


@dataclass(frozen=True)
class ExperimentConfig:
    """Coordinates for the model + SAE under test.

    These defaults target the smoke test. `sae_release` / `sae_id` are filled
    in once we confirm the matching Gemma Scope release in the SAELens registry
    (see smoke_test.py), and may be overridden there.
    """

    model_name: str = "google/gemma-3-4b-pt"
    # A middle layer with a trained Gemma Scope 2 residual SAE. The release
    # gemma-scope-2-4b-pt-res has SAEs at layers 9, 17, 22, 29; 17 sits in the
    # middle of gemma-3-4b's ~34 layers. resid_post of this layer is the hook
    # point the SAE was trained on (== the forward-hook output we capture).
    layer: int = 17
    # Confirmed against the SAELens pretrained-SAE registry (see smoke_test.py).
    sae_release: str = "gemma-scope-2-4b-pt-res"
    sae_width: str = "16k"
    sae_l0: str = "medium"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 0  # smoke test only needs a forward pass


# A single shared default instance for convenience.
CONFIG = ExperimentConfig()
