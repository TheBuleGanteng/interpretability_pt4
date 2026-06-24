"""Model-performance (cross-entropy / perplexity) measurement.

Measures each (model x bit-width)'s next-token cross-entropy on the SAME streamed
Pile tokens the SAE and noise-floor measurements use — the standard
quantization-quality proxy ("did compression raise the model's loss?").

Cross-entropy is read from the model's output logits of the forward pass that
ALREADY runs for activation capture; there is no separate forward. The driver
(smoke_test.run_condition) feeds each chunk's logits and the input ids here.

Accumulation is TOKEN-WEIGHTED, the same discipline as FVU: sum the total
negative log-likelihood over all predicted tokens and the predicted-token count
separately across chunks, then divide ONCE at the end. Averaging per-chunk mean
losses would be subtly wrong, because chunks can carry different valid-token
counts (the same class of bug as averaging per-chunk FVU).

Next-token shift: within each chunk the logits at positions 0..T-2 predict the
tokens at positions 1..T-1 (the standard shifted-logits/labels alignment). The
final position of each chunk has no in-chunk label and is dropped — a slight
undercount at chunk boundaries (documented; the alternative of carrying the label
across chunk boundaries is not worth the complexity at these chunk sizes).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def cross_entropy_sums(logits: torch.Tensor, input_ids: torch.Tensor) -> dict:
    """Token-weighted next-token cross-entropy SUMS for one chunk.

    Args:
      logits: [B, T, V] model output logits for the chunk.
      input_ids: [B, T] the token ids that were fed in (labels come from these).

    Returns {"sum_nll": float, "n_tokens": int} where `sum_nll` is the TOTAL
    negative log-likelihood (nats) over the B*(T-1) predicted positions and
    `n_tokens` is that predicted-token count. The caller accumulates both across
    chunks and divides once: cross_entropy = sum_nll / n_tokens.

    The next-token shift drops the last position of each chunk (no in-chunk
    label). Loss is computed in float32 for a numerically stable log-softmax over
    the large (~262k) Gemma vocabulary.
    """
    # Shift so logits[:, t] is scored against the token at position t+1.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:].to(shift_logits.device)
    vocab = shift_logits.size(-1)

    # reduction="sum" -> TOTAL nll over predicted tokens (not a per-chunk mean),
    # so the driver can accumulate and divide once for the token-weighted mean.
    sum_nll = F.cross_entropy(
        shift_logits.reshape(-1, vocab).float(),
        shift_labels.reshape(-1),
        reduction="sum",
    )
    return {"sum_nll": sum_nll.item(), "n_tokens": int(shift_labels.numel())}


def finalize_cross_entropy(sum_nll: float, n_tokens: int) -> dict:
    """Form the token-weighted mean cross-entropy and perplexity from sums.

    The single division point: cross_entropy = sum_nll / n_tokens (nats/token),
    perplexity = exp(cross_entropy). `n_tokens` is the predicted-token count.
    """
    cross_entropy = sum_nll / n_tokens
    return {
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(cross_entropy),
        "tokens": n_tokens,
    }
