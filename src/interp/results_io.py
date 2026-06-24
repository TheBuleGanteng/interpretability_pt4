"""Persist all measurement results to a structured JSON file.

After a run, every enabled measurement (SAE reconstruction, noise-floor control,
model-performance cross-entropy) is written to one timestamped run file plus a
stable `latest.json`, so a separate presentation cell can render tables/graphs
without re-running the GPU work. Only the measurements that actually ran are
included for each condition (the toggles are respected) — a missing key means
"not measured", a {"status": ...} value means "attempted but skipped/failed".

JSON is used deliberately: no extra dependencies, and the presentation layer can
load it with `load_latest` (or plain `json.load`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _serialize_sae(entry: object) -> dict:
    """SAE record in the documented shape, or a status wrapper."""
    if not isinstance(entry, dict):
        return {"status": entry}
    return {
        "fvu": entry["fvu"],
        "l0": entry["l0"],
        "mse": entry["mse"],
        "fvu_denominator": entry["fvu_denominator"],
        "tokens": entry["tokens"],
        "stability": [
            {"tokens": s["tokens"], "fvu": s["fvu"], "l0": s["l0"]}
            for s in entry.get("stability", [])
        ],
    }


def _serialize_noise(entry: object) -> dict:
    """Noise-floor record in the documented shape, or a status wrapper."""
    if not isinstance(entry, dict):
        return {"status": entry}
    return {
        "pert_rel_l2": entry["pert_rel_l2"],
        "mean_token_l2": entry["mean_token_l2"],
        "sae_residual_rel": entry["sae_rel_residual"],
        "ratio": entry["ratio"],
        "pert_mse": entry["pert_mse"],
        "sae_mse": entry["sae_mse"],
        "tokens": entry["tokens"],
    }


def _serialize_perf(entry: object) -> dict:
    """Performance record in the documented shape, or a status wrapper."""
    if not isinstance(entry, dict):
        return {"status": entry}
    return {
        "cross_entropy": entry["cross_entropy"],
        "perplexity": entry["perplexity"],
        "delta_vs_bf16_abs": entry.get("delta_vs_bf16_abs"),
        "delta_vs_bf16_pct": entry.get("delta_vs_bf16_pct"),
        "tokens": entry["tokens"],
    }


def build_results_document(
    metadata: dict,
    sae_results: dict | None,
    noise_results: dict | None,
    perf_results: dict | None,
    model_configs: list,
    display_bitwidths: tuple,
) -> dict:
    """Assemble the nested results document: metadata + per-condition records.

    `*_results` are keyed [model_name][bit_width]; pass None for a measurement
    that was disabled. Each (model x bit-width) record contains only the
    measurement keys that were actually computed for that condition.
    """
    conditions: dict = {}
    for cfg in model_configs:
        name = cfg["model_name"]
        per_bw: dict = {}
        for bw in display_bitwidths:
            record: dict = {}
            sae = sae_results.get(name, {}).get(bw) if sae_results else None
            nf = noise_results.get(name, {}).get(bw) if noise_results else None
            perf = perf_results.get(name, {}).get(bw) if perf_results else None
            if sae is not None:
                record["sae"] = _serialize_sae(sae)
            if nf is not None:
                record["noise_floor"] = _serialize_noise(nf)
            if perf is not None:
                record["performance"] = _serialize_perf(perf)
            if record:
                per_bw[bw] = record
        conditions[name] = per_bw
    return {"metadata": metadata, "conditions": conditions}


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically (write to a temp file, then os.replace).

    Avoids leaving a half-written / truncated JSON file if the process is
    interrupted mid-write — the presentation layer always sees a complete file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)  # atomic on the same filesystem


def write_results(
    document: dict,
    timestamp: str,
    results_dir: str = "results",
) -> tuple[Path, Path]:
    """Write `document` to results/run_<timestamp>.json and results/latest.json.

    Creates `results_dir` if missing. Both files are written atomically. Returns
    (run_path, latest_path).
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / f"run_{timestamp}.json"
    latest_path = out_dir / "latest.json"
    text = json.dumps(document, indent=2)
    _atomic_write_text(run_path, text)
    _atomic_write_text(latest_path, text)
    return run_path, latest_path


def load_latest(results_dir: str = "results") -> dict:
    """Load the most recent results document (for the presentation layer)."""
    return json.loads((Path(results_dir) / "latest.json").read_text())
