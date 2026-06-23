"""Confound decomposition estimands and per-indication spectra."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _auroc_safe(metrics: Dict[str, Any]) -> Optional[float]:
    oof = metrics.get("oof_metrics", {})
    if isinstance(oof, dict) and "auroc" in oof:
        return float(oof["auroc"])
    if "mean_auroc" in metrics and metrics["mean_auroc"] is not None:
        return float(metrics["mean_auroc"])
    return None


def decompose_auroc_components(
    task_auroc: float,
    baselines: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Approximate decomposition of apparent AUROC into metadata-explained and residual.

    Components above chance (0.5) are allocated proportionally to explain
    min(task, metadata_combined) - 0.5; residual biological = max(0, task - meta_ceiling).
    """
    comp_names = ["indication", "portal", "compositional", "provenance"]
    single = {
        n: _auroc_safe(baselines.get(n, {}))
        for n in comp_names
        if not baselines.get(n, {}).get("skipped")
    }
    meta_ceiling = _auroc_safe(baselines.get("metadata_combined", {}))
    block_log = _auroc_safe(baselines.get("block_logistic", {}))
    for n in comp_names:
        single.setdefault(n, None)

    above = {k: max(0.0, (v if v is not None else 0.5) - 0.5) for k in comp_names for v in [single.get(k)]}
    total_above = sum(above.values()) + 1e-12
    explainable = max(0.0, min(task_auroc, meta_ceiling or task_auroc) - 0.5)
    shares = {k: (above[k] / total_above) * explainable for k in comp_names}
    residual_bio = max(0.0, task_auroc - (meta_ceiling or 0.5))
    unattributed = max(0.0, task_auroc - 0.5 - sum(shares.values()) - residual_bio)

    return {
        "task_auroc": float(task_auroc),
        "metadata_ceiling_auroc": meta_ceiling,
        "block_logistic_auroc": block_log,
        "component_aurocs": single,
        "explained_shares_above_chance": shares,
        "residual_biological": float(residual_bio),
        "unattributed": float(unattributed),
        "confound_fraction": float(
            (explainable) / max(task_auroc - 0.5, 1e-12) if task_auroc > 0.5 else 1.0
        ),
    }


def per_indication_decomposition(
    df,
    y: np.ndarray,
    oof_scores: np.ndarray,
    project_col: str = "project_id",
    min_n: int = 10,
) -> List[Dict[str, Any]]:
    """Within-indication apparent AUROC where both classes present."""
    from sklearn.metrics import roc_auc_score

    rows: List[Dict[str, Any]] = []
    df_reset = df.reset_index(drop=True)
    for proj, ix in df_reset.groupby(project_col).groups.items():
        idx = np.array(list(ix), dtype=int)
        if len(idx) < min_n:
            continue
        yt = y[idx]
        if np.unique(yt).size < 2:
            rows.append({
                "indication": str(proj),
                "n": int(len(idx)),
                "n_positive": int((yt > 0.5).sum()),
                "apparent_auroc": None,
                "note": "single class — confound-saturated stratum",
            })
            continue
        try:
            auc = float(roc_auc_score(yt, oof_scores[idx]))
        except Exception:
            auc = None
        rows.append({
            "indication": str(proj),
            "n": int(len(idx)),
            "n_positive": int((yt > 0.5).sum()),
            "apparent_auroc": auc,
        })
    return rows
