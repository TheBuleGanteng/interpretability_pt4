"""Streamed Pile corpus for SAE-reconstruction evaluation.

The Gemma Scope 2 SAEs report `dataset_path: monology/pile-uncopyrighted` in
their cfg metadata, so reconstruction should be measured on text from that same
distribution rather than a handful of hand-written sentences. This module
STREAMS a Pile slice (no full download) and packs it into fixed-length token
sequences the driver can feed through the model one batch at a time.

Preferred source is `NeelNanda/pile-10k` (a 10k-document Pile slice packaged for
interpretability work — small, fast, in-distribution). If it can't be reached we
fall back to streaming `monology/pile-uncopyrighted` directly. Both are opened
with `streaming=True`.
"""

from __future__ import annotations

from typing import Iterator

import torch

PILE_PRIMARY = "NeelNanda/pile-10k"
PILE_FALLBACK = "monology/pile-uncopyrighted"


def load_pile_stream(
    shuffle_seed: int | None = None,
    shuffle_buffer: int = 10_000,
) -> tuple[object, str]:
    """Open a streamed Pile dataset; return (dataset, source_name).

    Tries `NeelNanda/pile-10k` first and validates it is reachable by peeking at
    the first document (streaming datasets restart cleanly on re-iteration, so
    the peek consumes nothing). Falls back to `monology/pile-uncopyrighted` on
    any failure. `datasets` is imported lazily so importing this module stays
    cheap and does not require the package until a run actually streams data.

    If `shuffle_seed` is not None, the streaming iterable is shuffled with that
    FIXED seed and a `shuffle_buffer`-sized reservoir. This averages token
    difficulty over the corpus rather than reading it in raw stream order, while
    staying deterministic — re-iterating yields the SAME order every time, so
    every (model x bit-width) condition sees identical tokens (required for the
    noise-floor per-token comparison).
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(PILE_PRIMARY, split="train", streaming=True)
        _ = next(iter(ds))  # validate the stream is actually reachable
        source = PILE_PRIMARY
    except Exception:  # noqa: BLE001 — any failure means fall back
        ds = load_dataset(PILE_FALLBACK, split="train", streaming=True)
        source = PILE_FALLBACK

    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    return ds, source


def iter_token_batches(
    tokenizer,
    dataset,
    chunk_len: int,
    batch_size: int,
    max_tokens: int,
) -> Iterator[torch.Tensor]:
    """Yield [batch_size, chunk_len] LongTensors of packed Pile tokens.

    Documents are tokenized (BOS included at each document start) and their token
    ids concatenated into a rolling buffer, which is sliced into fixed-length
    `chunk_len` sequences — i.e. packed, not padded, so every position is a real
    token and a default full-attention forward is correct. Sequences are grouped
    into batches of `batch_size`. Streaming stops once `max_tokens` complete-chunk
    tokens have been produced (the final batch may be smaller). Re-iterating the
    same `dataset` restarts the stream, so every (model x bit-width) condition
    sees the SAME tokens, keeping the conditions directly comparable.
    """
    buffer: list[int] = []
    produced = 0
    batch: list[list[int]] = []

    for doc in dataset:
        text = doc.get("text") if isinstance(doc, dict) else None
        if not text:
            continue
        buffer.extend(tokenizer(text, add_special_tokens=True).input_ids)

        while len(buffer) >= chunk_len:
            batch.append(buffer[:chunk_len])
            del buffer[:chunk_len]
            produced += chunk_len

            if len(batch) == batch_size:
                yield torch.tensor(batch, dtype=torch.long)
                batch = []

            if produced >= max_tokens:
                if batch:
                    yield torch.tensor(batch, dtype=torch.long)
                return

    # Dataset exhausted before reaching max_tokens — emit whatever is buffered.
    if batch:
        yield torch.tensor(batch, dtype=torch.long)
