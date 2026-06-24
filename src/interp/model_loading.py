"""Model loading helpers.

Loads a Gemma model at a chosen precision (bf16 / 8-bit / 4-bit via bitsandbytes)
and provides a small utility to capture the residual stream at a chosen layer
using a forward hook.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import get_hf_token

# Supported precisions for the compression-gradient experiment.
PRECISIONS = ("bf16", "8bit", "4bit")


@dataclass
class LoadedModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    name: str
    precision: str = "4bit"


def load_model(
    model_name: str,
    precision: str = "4bit",
    device_map: str = "auto",
    compute_dtype: torch.dtype = torch.bfloat16,
) -> LoadedModel:
    """Load `model_name` at the requested `precision` for inference.

    precision:
      - "bf16": no quantization; weights in torch.bfloat16 (device_map="cuda").
      - "8bit": BitsAndBytesConfig(load_in_8bit=True).
      - "4bit": NF4 (load_in_4bit, double-quant, compute_dtype=bf16) — the
        original smoke-test path.

    The model precision is the experimental variable; the residual-stream capture
    and SAE encode/decode downstream are precision-agnostic (they cast to float32
    for the metric). Uses the HF token from the environment so gated Gemma repos
    resolve. Returned in eval mode.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision {precision!r}; expected one of {PRECISIONS}")

    token = get_hf_token(required=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    if precision == "bf16":
        # Uncompressed baseline. Pin to a single CUDA device per the task spec.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            token=token,
        )
    else:
        if precision == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:  # "4bit"
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device_map,
            torch_dtype=compute_dtype,
            token=token,
        )

    model.eval()
    return LoadedModel(
        model=model, tokenizer=tokenizer, name=model_name, precision=precision
    )


def load_model_4bit(
    model_name: str,
    device_map: str = "auto",
    compute_dtype: torch.dtype = torch.bfloat16,
) -> LoadedModel:
    """Load `model_name` quantised to 4-bit (NF4). Thin wrapper over load_model.

    Preserved for the existing call sites; delegates to load_model with
    precision="4bit" so the original behavior is unchanged.
    """
    return load_model(
        model_name,
        precision="4bit",
        device_map=device_map,
        compute_dtype=compute_dtype,
    )


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


@torch.no_grad()
def capture_resid_from_input_ids(
    loaded: LoadedModel,
    input_ids: torch.Tensor,
    layer: int,
    return_logits: bool = False,
):
    """Capture the residual stream at `layer` for a batch of pre-tokenized ids.

    Same capture point as `capture_residual_stream` (the *output* hidden state of
    decoder block `layer`), but takes a ready-made [batch, seq_len] LongTensor of
    token ids instead of raw text. This is the chunked-streaming entry point: the
    driver packs a Pile token stream into fixed-length sequences and feeds them
    here one batch at a time, so tens of thousands of tokens never have to be
    tokenized — or held in memory — all at once.

    By default returns just the activations tensor [batch, seq_len, d_model].
    When `return_logits` is True, returns (activations, logits) where `logits` is
    the model's [batch, seq_len, vocab] output from the SAME forward pass — so the
    cross-entropy / performance measurement reuses this forward rather than running
    a second one. The caller is responsible for freeing both tensors after
    accumulating their metrics.
    """
    layers = _decoder_layers(loaded.model)
    if not (0 <= layer < len(layers)):
        raise IndexError(f"layer {layer} out of range (model has {len(layers)} layers)")

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["resid"] = hidden.detach()

    handle = layers[layer].register_forward_hook(hook)
    try:
        input_ids = input_ids.to(loaded.model.device)
        # All sequences are exactly seq_len (the stream is packed, not padded),
        # so a default full-attention mask is correct — no attention_mask needed.
        # use_cache=False: we only need the layer's hidden state via the hook, so
        # skip allocating a past_key_values KV-cache for every chunk.
        out = loaded.model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()

    acts = captured["resid"]
    if return_logits:
        return acts, out.logits.detach()
    return acts
