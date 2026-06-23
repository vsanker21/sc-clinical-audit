"""Published-style signature catalog and confound-aware re-evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.sc_clinical_audit.evaluation import _oof_logistic_cv, load_pack_tensors
from scripts.train_metastasis_supervised import _groups_for_cv

_CATALOG_PATH = Path(__file__).resolve().parents[2] / "data/sc_clinical_audit/published_signatures.json"


def load_signature_catalog(catalog_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = catalog_path or _CATALOG_PATH
    if not path.is_file():
        # Minimal fallback if catalog not built yet
        return [
            {"id": "emt_core", "name": "EMT core", "type": "gene_list", "genes": ["VIM", "SNAI1", "TWIST1", "ZEB1", "CDH1"], "source": "fallback"},
        ]
    obj = json.loads(path.read_text(encoding="utf-8"))
    return list(obj.get("signatures", []))


def _score_cell_type_signature(masks: np.ndarray, types: List[str], cell_types: List[str]) -> np.ndarray:
    idx = [i for i, t in enumerate(types) if any(ct.lower() in t.lower() for ct in cell_types)]
    if not idx:
        return np.zeros(masks.shape[0])
    return masks[:, idx].astype(float).mean(axis=1)


def _score_gene_list_proxy(
    feats,
    masks,
    types: List[str],
    dim_per: Dict[str, int],
    sup_ix: np.ndarray,
    genes: List[str],
) -> np.ndarray:
    """Proxy: mean block expression (gene-aligned indices unavailable in L1 blocks)."""
    from src.sc_clinical_audit.evaluation import block_input_features
    X = block_input_features(feats, masks, types, dim_per, sup_ix)
    return X.mean(axis=1)


# Primary (manuscript) criteria: conservative confound-correction survival.
STRICT_CORRECTED_MIN_AUROC = 0.60
STRICT_MAX_NAIVE_DELTA = 0.10
# Exploratory criteria (legacy sensitivity; not headline).
EXPLORATORY_CORRECTED_MIN_AUROC = 0.55
EXPLORATORY_MAX_NAIVE_DELTA = 0.15


def _survives_correction(
    naive_auc: Optional[float],
    corr_auc: Optional[float],
    *,
    corrected_min: float,
    max_delta: float,
) -> bool:
    if corr_auc is None or naive_auc is None:
        return False
    return bool(corr_auc >= corrected_min and (naive_auc - corr_auc) < max_delta)


def score_signatures_on_pack(
    data_dir: Path,
    *,
    k: int = 5,
    seed: int = 42,
    catalog: Optional[List[Dict[str, Any]]] = None,
    max_signatures: int = 0,
) -> List[Dict[str, Any]]:
    catalog = catalog or load_signature_catalog()
    if max_signatures > 0:
        catalog = catalog[:max_signatures]

    feats, masks, df, y_all, m_all, sup_ix = load_pack_tensors(data_dir)
    y = y_all[sup_ix]
    if np.unique(y).size < 2:
        return []
    groups = _groups_for_cv(df, sup_ix, "donor_id")
    from src.training.l1_block_features import load_l1_schema

    types, dim_per, _ = load_l1_schema(data_dir / "layer1_celltype_schema.json")
    m_np = masks[sup_ix].float().numpy()
    results: List[Dict[str, Any]] = []

    for sig in catalog:
        stype = sig.get("type", "gene_list")
        scores: Optional[np.ndarray] = None

        if stype == "cell_type_composition":
            scores = _score_cell_type_signature(m_np, types, sig.get("cell_types", []))
        elif stype == "gene_list":
            scores = _score_gene_list_proxy(feats, masks, types, dim_per, sup_ix, sig.get("genes", []))
        elif stype == "layer3_de":
            import torch
            integ = torch.load(data_dir / "integrated_dataset.pt", map_location="cpu", weights_only=False)
            if "layer3_features" in integ:
                l3 = integ["layer3_features"].float().numpy()[sup_ix]
                scores = l3.mean(axis=1)
        elif stype == "block_mean_comparator":
            from src.sc_clinical_audit.evaluation import block_input_features
            X = block_input_features(feats, masks, types, dim_per, sup_ix)
            scores = X.mean(axis=1)
        else:
            continue

        if scores is None or np.std(scores) < 1e-12:
            results.append({
                "signature_id": sig["id"],
                "name": sig.get("name", sig["id"]),
                "source": sig.get("source", ""),
                "skipped": True,
                "reason": "no score variance",
            })
            continue

        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        naive = _oof_logistic_cv(scores.reshape(-1, 1), y, groups, k=k, seed=seed)
        proj = df.iloc[sup_ix]["project_id"].fillna("unknown").astype(str)
        best_proj, best_n = None, 0
        for p in proj.unique():
            m = (proj == p).values
            if np.unique(y[m]).size == 2 and m.sum() > best_n:
                best_proj, best_n = p, int(m.sum())
        corrected: Dict[str, Any] = {"skipped": True}
        if best_proj and best_n >= 10:
            m = (proj == best_proj).values
            corrected = _oof_logistic_cv(
                scores[m].reshape(-1, 1), y[m], groups[m],
                k=min(k, max(2, len(np.unique(groups[m])))),
                seed=seed,
            )
            corrected["within_indication"] = best_proj
            corrected["n"] = int(m.sum())

        naive_auc = naive.get("oof_metrics", {}).get("auroc")
        corr_auc = corrected.get("oof_metrics", {}).get("auroc")
        results.append({
            "signature_id": sig["id"],
            "name": sig.get("name", sig["id"]),
            "source": sig.get("source", ""),
            "type": stype,
            "naive_oof_auroc": naive_auc,
            "corrected_within_indication_auroc": corr_auc,
            "corrected_indication": corrected.get("within_indication"),
            "naive_minus_corrected": (
                float(naive_auc - corr_auc) if naive_auc is not None and corr_auc is not None else None
            ),
            "survives_correction_strict": _survives_correction(
                naive_auc, corr_auc,
                corrected_min=STRICT_CORRECTED_MIN_AUROC,
                max_delta=STRICT_MAX_NAIVE_DELTA,
            ),
            "survives_correction_exploratory": _survives_correction(
                naive_auc, corr_auc,
                corrected_min=EXPLORATORY_CORRECTED_MIN_AUROC,
                max_delta=EXPLORATORY_MAX_NAIVE_DELTA,
            ),
            # Headline column uses strict criteria (manuscript primary).
            "survives_correction": _survives_correction(
                naive_auc, corr_auc,
                corrected_min=STRICT_CORRECTED_MIN_AUROC,
                max_delta=STRICT_MAX_NAIVE_DELTA,
            ),
        })
    return results
