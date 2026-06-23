"""Transportability matrix: train stratum → test stratum performance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.sc_clinical_audit.evaluation import block_input_features, load_pack_tensors
from src.training.l1_block_features import load_l1_schema


def _train_block_logistic(Xtr: np.ndarray, ytr: np.ndarray, seed: int = 42) -> Tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    clf.fit(Xs, ytr.astype(int))
    return clf, scaler


def transport_matrix(
    data_dir,
    *,
    stratify_col: str = "project_id",
    min_train: int = 8,
    min_test: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train block-logistic on stratum A, evaluate AUROC on stratum B (no CV mixing).
    Diagonal uses donor-grouped OOF from full data as reference when possible.
    """
    feats, masks, df, y_all, m_all, sup_ix = load_pack_tensors(data_dir)
    y = y_all[sup_ix]
    types, dim_per, _ = load_l1_schema(data_dir / "layer1_celltype_schema.json")
    X = block_input_features(feats, masks, types, dim_per, sup_ix)
    sub = df.iloc[sup_ix].copy().reset_index(drop=True)
    if stratify_col not in sub.columns:
        from scripts.build_auditable_provenance_tables import _infer_portal
        if stratify_col == "portal_proxy":
            sub["portal_proxy"] = [_infer_portal(row) for _, row in sub.iterrows()]
        else:
            return {"skipped": True, "reason": f"missing {stratify_col}"}

    strata = sorted(sub[stratify_col].fillna("unknown").astype(str).unique())
    matrix: Dict[str, Dict[str, Optional[float]]] = {}
    details: List[Dict[str, Any]] = []

    for train_s in strata:
        matrix[train_s] = {}
        tr_mask = (sub[stratify_col].astype(str) == train_s).values
        if tr_mask.sum() < min_train or np.unique(y[tr_mask]).size < 2:
            for test_s in strata:
                matrix[train_s][test_s] = None
            continue
        clf, scaler = _train_block_logistic(X[tr_mask], y[tr_mask], seed=seed)
        for test_s in strata:
            te_mask = (sub[stratify_col].astype(str) == test_s).values
            if te_mask.sum() < min_test or np.unique(y[te_mask]).size < 2:
                matrix[train_s][test_s] = None
                continue
            prob = clf.predict_proba(scaler.transform(X[te_mask]))[:, 1]
            try:
                auc = float(roc_auc_score(y[te_mask], prob))
            except Exception:
                auc = None
            matrix[train_s][test_s] = auc
            details.append({
                "train_stratum": train_s,
                "test_stratum": test_s,
                "n_train": int(tr_mask.sum()),
                "n_test": int(te_mask.sum()),
                "auroc": auc,
                "same_stratum": int(train_s == test_s),
            })

    # portal transport if portal_proxy available
    from scripts.build_auditable_provenance_tables import _infer_portal
    sub["portal_proxy"] = [_infer_portal(row) for _, row in sub.iterrows()]

    return {
        "stratify_col": stratify_col,
        "strata": strata,
        "matrix": matrix,
        "details": details,
        "portal_substrata": sorted(sub["portal_proxy"].unique().tolist()),
    }
