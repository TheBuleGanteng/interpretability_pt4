"""Model loading helpers.

Loads a Gemma model in 4-bit (via bitsandbytes) and provides a small utility to
capture the residual stream at a chosen layer using a forward hook.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import get_hf_token


@dataclass
class LoadedModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    name: str


def load_model_4bit(
    model_name: str,
    device_map: str = "auto",
    compute_dtype: torch.dtype = torch.bfloat16,
) -> LoadedModel:
    """Load `model_name` quantised to 4-bit (NF4) for inference.

    Uses the HF token from the environment (see config.get_hf_token) so gated
    Gemma repos resolve. The model is returned in eval mode.
    """
    token = get_hf_token(required=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=compute_dtype,
        token=token,
    )
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, name=model_name)


def _decoder_layers(model) -> torch.nn.ModuleList:
    """Return the list of transformer decoder layers across HF Gemma variants.

    Gemma-3 text models expose `model.model.layers`; multimodal wrappers nest it
    under `model.model.language_model.layers`. We probe both.
    """
    inner = getattr(model, "model", model)
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "language_model"):
        lm = inner.language_model
        return getattr(lm, "model", lm).layers
    raise AttributeError(
        "Could not locate decoder layers on this model; inspect its structure."
    )


@torch.no_grad()
def capture_residual_stream(
    loaded: LoadedModel,
    text: str,
    layer: int,
) -> tuple[torch.Tensor, dict]:
    """Run `text` through the model and capture the residual stream at `layer`.

    Captures the *output* hidden state of decoder block `layer` — i.e. the
    residual stream after that block — which is the activation point Gemma Scope
    residual-stream SAEs are trained on.

    Returns (activations, info) where `activations` has shape
    [batch, seq_len, d_model] and `info` records token / shape metadata.
    """
    layers = _decoder_layers(loaded.model)
    if not (0 <= layer < len(layers)):
        raise IndexError(f"layer {layer} out of range (model has {len(layers)} layers)")

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        # Decoder layers return a tuple; the first element is the hidden state.
        hidden = output[0] if isinstance(output, tuple) else output
        captured["resid"] = hidden.detach()

    handle = layers[layer].register_forward_hook(hook)
    try:
        inputs = loaded.tokenizer(text, return_tensors="pt").to(loaded.model.device)
        loaded.model(**inputs)
    finally:
        handle.remove()

    acts = captured["resid"]
    info = {
        "layer": layer,
        "n_layers": len(layers),
        "input_ids_shape": tuple(inputs["input_ids"].shape),
        "resid_shape": tuple(acts.shape),
        "d_model": acts.shape[-1],
    }
    return acts, info
