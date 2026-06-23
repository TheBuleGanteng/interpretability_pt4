# interpretability_pt4

Mechanistic-interpretability experiment scaffold. Logic lives in importable
modules under `src/interp/`; scripts orchestrate them. This stage is a
feasibility check, not the full harness.

## Layout

```
src/interp/
  config.py         # env / HF_TOKEN loading + experiment coordinates
  model_loading.py  # 4-bit Gemma load + residual-stream capture hook
  sae_loading.py    # Gemma Scope SAE loading via SAELens
smoke_test.py       # standalone end-to-end feasibility check
requirements.txt    # frozen venv stack
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

Put your Hugging Face token (with access to the gated Gemma repos) in
`.env.local` at the repo root:

```
HF_TOKEN=hf_xxx
```

`config.py` also accepts the alternate spellings `HF_Token` /
`HUGGING_FACE_HUB_TOKEN`.

## Smoke test

```bash
.venv/bin/python smoke_test.py
```

It loads the model in 4-bit, captures the residual stream at one middle layer,
loads the matching Gemma Scope residual-stream SAE for that layer via SAELens,
encodes the activations, and prints activation shapes plus the SAE's
reconstruction error. A CUDA GPU is required for the 4-bit load.
