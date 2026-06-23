"""Donor-grouped CV evaluation for task and metadata baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import scripts.train_metastasis_supervised as tms
from scripts.train_metastasis_supervised import BlockMLP, _groups_for_cv, _train_block_mlp_fold
from src.evaluation.preregistered_study import bootstrap_auroc_ci, extended_metrics
from src.training.l1_block_features import block_mean_matrix, load_l1_schema
from src.training.metastasis_from_manifest import metastasis_y_mask_arrays


def _oof_logistic_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    k: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    gkf = GroupKFold(n_splits=min(k, len(np.unique(groups))))
    oof = np.full(len(y), np.nan)
    fold_aurocs: List[float] = []
    for tr, va in gkf.split(X, y, groups):
        if np.unique(y[tr]).size < 2:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        clf.fit(Xtr, y[tr].astype(int))
        prob = clf.predict_proba(Xva)[:, 1]
        oof[va] = prob
        met = extended_metrics(y[va], prob)
        if "auroc" in met:
            fold_aurocs.append(met["auroc"])
    if not fold_aurocs:
        return {"skipped": True, "reason": "no valid CV folds (single-class training folds)"}
    out: Dict[str, Any] = {
        "mean_auroc": float(np.mean(fold_aurocs)) if fold_aurocs else None,
        "oof_metrics": extended_metrics(y, oof) if np.isfinite(oof).all() else {},
    }
    if np.isfinite(oof).all():
        out["oof_scores"] = oof.tolist()
        out["oof_bootstrap_auroc_ci"] = bootstrap_auroc_ci(y, oof)
    return out


def load_pack_tensors(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    types, dim_per, _ = load_l1_schema(data_dir / "layer1_celltype_schema.json")
    tms.types_cache = types
    tms.dim_cache = dim_per
    integ = torch.load(data_dir / "integrated_dataset.pt", map_location="cpu", weights_only=False)
    feats = integ["layer1_features"]
    masks = integ["layer1_celltype_mask"]
    n_graphs = int(feats.shape[0])
    df = pd.read_csv(data_dir / "layer1_cohort_manifest_aligned.csv", low_memory=False).iloc[:n_graphs]
    y_all, m_all = metastasis_y_mask_arrays(df, n_graphs, infer_local_primary_for_unlabeled=False)
    sup_ix = np.where(m_all > 0.5)[0].astype(np.int64)
    return feats, masks, df, y_all, m_all, sup_ix


def composition_features(masks: torch.Tensor, sup_ix: np.ndarray) -> np.ndarray:
    """Cell-type presence proportions (L1 norm normalized mask)."""
    m = masks[sup_ix].float().numpy()
    denom = m.sum(axis=1, keepdims=True) + 1e-8
    return m / denom


def block_input_features(
    feats: torch.Tensor,
    masks: torch.Tensor,
    types: List[str],
    dim_per: Dict[str, int],
    sup_ix: np.ndarray,
) -> np.ndarray:
    bm = block_mean_matrix(feats[sup_ix], masks[sup_ix], types, dim_per).numpy()
    mk = masks[sup_ix].float().numpy()
    return np.concatenate([np.log1p(np.abs(bm)), mk], axis=1)


def metadata_design_matrix(df: pd.DataFrame, sup_ix: np.ndarray, columns: List[str]) -> np.ndarray:
    cols = [c for c in columns if c in df.columns]
    sub = df.iloc[sup_ix][cols].fillna("unknown").astype(str)
    return pd.get_dummies(sub, drop_first=True).values.astype(np.float64)


def run_block_mlp_oof(
    data_dir: Path,
    *,
    k: int = 5,
    epochs: int = 40,
    seed: int = 42,
    gpu: bool = False,
) -> Dict[str, Any]:
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    types, dim_per, _ = load_l1_schema(data_dir / "layer1_celltype_schema.json")
    tms.types_cache = types
    tms.dim_cache = dim_per
    n_types = len(types)
    feats, masks, df, y_all, m_all, sup_ix = load_pack_tensors(data_dir)
    y_sup = y_all[sup_ix]
    if np.unique(y_sup).size < 2:
        return {"skipped": True, "reason": "single class in supervised set"}
    pos_weight = float((y_sup < 0.5).sum() / max((y_sup > 0.5).sum(), 1))
    groups = _groups_for_cv(df, sup_ix, "donor_id")
    gkf = GroupKFold(n_splits=min(k, len(np.unique(groups))))
    oof_prob = np.full(len(sup_ix), np.nan)
    fold_metrics: List[Dict[str, Any]] = []
    for fold, (tr_rel, va_rel) in enumerate(gkf.split(sup_ix, y_sup, groups)):
        tr_g, va_g = sup_ix[tr_rel], sup_ix[va_rel]
        model, _ = _train_block_mlp_fold(
            feats[tr_g], masks[tr_g],
            torch.from_numpy(y_all[tr_g].astype(np.float32)),
            feats[va_g], masks[va_g],
            torch.from_numpy(y_all[va_g].astype(np.float32)),
            n_types=n_types, device=device, epochs=epochs, pos_weight=pos_weight,
        )
        model.eval()
        with torch.no_grad():
            bm_va = block_mean_matrix(feats[va_g], masks[va_g], types, dim_per)
            prob = torch.sigmoid(model(bm_va.to(device), masks[va_g].to(device))).cpu().numpy()
        oof_prob[va_rel] = prob
        met = extended_metrics(y_all[va_g], prob)
        met["fold"] = fold
        fold_metrics.append(met)
    aurocs = [m["auroc"] for m in fold_metrics if "auroc" in m]
    summary = {
        "model": "block_mlp",
        "n_supervised": int(len(sup_ix)),
        "n_positive": int((y_sup > 0.5).sum()),
        "fold_metrics": fold_metrics,
        "mean_auroc": float(np.mean(aurocs)) if aurocs else None,
    }
    if np.isfinite(oof_prob).all():
        summary["oof_metrics"] = extended_metrics(y_sup, oof_prob)
        summary["oof_scores"] = oof_prob.tolist()
        summary["oof_bootstrap_auroc_ci"] = bootstrap_auroc_ci(y_sup, oof_prob)
    return summary


def run_metadata_baselines(
    data_dir: Path,
    *,
    k: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    feats, masks, df, y_all, m_all, sup_ix = load_pack_tensors(data_dir)
    y = y_all[sup_ix]
    if np.unique(y).size < 2:
        return {"skipped": True, "reason": "single class"}
    groups = _groups_for_cv(df, sup_ix, "donor_id")
    types, dim_per, _ = load_l1_schema(data_dir / "layer1_celltype_schema.json")

    baselines: Dict[str, Any] = {}
    feature_sets = {
        "indication": ["project_id"],
        "portal": ["portal_proxy"],
        "provenance": ["label_source_field", "label_source_value"],
        "metadata_combined": ["project_id", "portal_proxy", "label_source_field", "label_source_value", "disease_label"],
    }
    # inject portal proxy column
    df2 = df.copy()
    from scripts.build_auditable_provenance_tables import _infer_portal, _label_field_and_value

    portals, fields, vals = [], [], []
    for _, row in df2.iterrows():
        portals.append(_infer_portal(row))
        f, v, _ = _label_field_and_value(row)
        fields.append(f)
        vals.append(v)
    df2["portal_proxy"] = portals
    df2["label_source_field"] = fields
    df2["label_source_value"] = vals

    for name, cols in feature_sets.items():
        X = metadata_design_matrix(df2, sup_ix, cols)
        if X.shape[1] == 0:
            baselines[name] = {"skipped": True, "reason": "no features"}
            continue
        baselines[name] = _oof_logistic_cv(X, y, groups, k=k, seed=seed)
        baselines[name]["features"] = cols

    comp = composition_features(masks, sup_ix)
    baselines["compositional"] = _oof_logistic_cv(comp, y, groups, k=k, seed=seed)
    baselines["compositional"]["features"] = ["cell_type_proportions"]

    blk = block_input_features(feats, masks, types, dim_per, sup_ix)
    baselines["block_logistic"] = _oof_logistic_cv(blk, y, groups, k=k, seed=seed)
    baselines["block_logistic"]["features"] = ["log1p_block_means", "presence_mask"]

    return baselines
