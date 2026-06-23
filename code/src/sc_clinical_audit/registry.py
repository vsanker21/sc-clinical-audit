"""Multi-drive specimen registry for Archetype A benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from scripts.build_auditable_provenance_tables import _infer_portal, _label_field_and_value, _project_id_rule
from src.training.metastasis_from_manifest import metastasis_label_provenance


def _leakage_mode(row: pd.Series) -> str:
    proj = str(row.get("project_id_tag", row.get("project_id", ""))).strip()
    portal = str(row.get("portal", ""))
    pos = int(float(row.get("is_metastatic", 0)) > 0.5)
    if pos and proj in ("TCGA-SKCM",) and "7/7" in str(row.get("notes", "")):
        return "indication_saturated"
    if pos and "HTAPP" in portal and proj == "TCGA-BRCA":
        return "portal_indication_coupled"
    if str(row.get("label_provenance_tag", "")).startswith("specimen"):
        return "specimen_grounded"
    if str(row.get("label_provenance_tag", "")).startswith("inferred"):
        return "inferred_primary"
    if float(row.get("metastasis_supervision_mask", 0)) < 0.5:
        return "unlabeled"
    if pos:
        return "metastatic_labeled"
    return "primary_labeled"


def _normalize_manifest_row(row: pd.Series, source_pack: str, source_drive: str) -> Dict[str, Any]:
    disease = str(row.get("disease_label", ""))
    suggested_proj, rule = _project_id_rule(disease)
    actual_proj = str(row.get("project_id", "")).strip()
    label_field, label_val, prov_tag = _label_field_and_value(row)
    portal = _infer_portal(row)
    donor = str(row.get("donor_id", "")).strip()
    sup_mask = float(row.get("metastasis_supervision_mask", 0))
    is_meta = int(float(row.get("is_metastatic", 0)) > 0.5)
    out = {
        "graph_file": str(row.get("graph_file", "")),
        "h5ad_stem": str(row.get("h5ad_stem", "")),
        "portal": portal,
        "donor_id": donor,
        "study_id": str(row.get("study_id", "")),
        "disease_label": disease,
        "project_id_tag": actual_proj or suggested_proj,
        "project_id_derivation": rule,
        "label_source_field": label_field,
        "label_source_value": label_val,
        "label_provenance_tag": prov_tag,
        "is_metastatic": is_meta,
        "metastasis_supervision_mask": sup_mask,
        "source_pack": source_pack,
        "source_drive": source_drive,
    }
    out["leakage_mode"] = _leakage_mode(pd.Series(out))
    # F943 bulk graphs: graph_file is the unique specimen key (943 graphs, 207 h5ad stems)
    if source_pack.startswith("f943") and out["graph_file"]:
        out["specimen_key"] = out["graph_file"]
    else:
        out["specimen_key"] = out["h5ad_stem"] or out["graph_file"] or donor
    return out


def _load_manifest(path: Path, source_pack: str, source_drive: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    rows = [_normalize_manifest_row(row, source_pack, source_drive) for _, row in df.iterrows()]
    return pd.DataFrame(rows)


def manifest_sources(data_root: Path) -> List[Tuple[str, str, Path]]:
    """Return (drive, pack_name, manifest_path) for all known multi-drive manifests."""
    e_overflow = Path(r"E:/MetastaticEvolution")
    c_root = Path(r"C:/MetastaticEvolution")
    sources: List[Tuple[str, str, Path]] = [
        ("D", "specimen_authority", data_root / "data/integrated_specimen_authority/layer1_cohort_manifest_aligned.csv"),
        ("D", "graphs_manifest_287", data_root / "artifacts/layer1/graphs_manifest.csv"),
        ("D", "cohort_l1_l2", data_root / "data/integrated_l1_314_l2_cohort_project/layer1_cohort_manifest_aligned.csv"),
        ("D", "htan_bridge", data_root / "data/integrated_htan_bridge_20260519/layer1_cohort_manifest_aligned.csv"),
        ("D", "integrated_287", data_root / "data/integrated/layer1_cohort_manifest_aligned.csv"),
    ]
    if e_overflow.is_dir():
        sources.extend([
            ("E", "f943_merged", e_overflow / "artifacts/layer1/f943_d_merged/graphs_manifest.csv"),
            ("E", "f943_labeled207", e_overflow / "artifacts/layer1/f943_d_merged/graphs_manifest_d_labeled207.csv"),
            ("E", "f142_d_parity", e_overflow / "data/integrated_f142_d_parity/layer1_cohort_manifest_aligned.csv"),
            ("E", "f207_base", e_overflow / "data/integrated_f207_base_f/layer1_cohort_manifest_aligned.csv"),
            ("E", "f943_lr_merged", e_overflow / "data/integrated_f943_lr_merged/layer1_cohort_manifest_aligned.csv"),
        ])
    if c_root.is_dir():
        sources.append(("C", "lr_repair_207", c_root / "artifacts/layer1/lr_repair_manifest_0_656.csv"))
    return sources


def integrated_pack_dirs(data_root: Path) -> List[Tuple[str, str, Path]]:
    """Packs with integrated_dataset.pt suitable for CV decomposition."""
    packs: List[Tuple[str, str, Path]] = [
        ("D", "specimen_authority", data_root / "data/integrated_specimen_authority"),
        ("D", "cohort_l1_l2", data_root / "data/integrated_l1_314_l2_cohort_project"),
        ("D", "htan_bridge", data_root / "data/integrated_htan_bridge_20260519"),
        ("D", "integrated_287", data_root / "data/integrated"),
    ]
    e_root = Path(r"E:/MetastaticEvolution/data")
    if e_root.is_dir():
        # Headline scale proxy: D-graph symlink parity (142/33) — preferred over F943 rebuild
        parity = e_root / "integrated_f142_d_parity"
        if (parity / "integrated_dataset.pt").is_file():
            packs.insert(0, ("E", "f142_d_parity_headline_scale", parity))
        for name in ("integrated_f207_base_f",):
            p = e_root / name
            if (p / "integrated_dataset.pt").is_file():
                packs.append(("E", name, p))
    return [(d, n, p) for d, n, p in packs if (p / "integrated_dataset.pt").is_file()]


def build_multi_drive_registry(data_root: Path) -> Dict[str, Any]:
    """Merge all manifests; dedupe by specimen_key with specimen_authority priority."""
    frames: List[pd.DataFrame] = []
    for drive, pack, path in manifest_sources(data_root):
        sub = _load_manifest(path, pack, drive)
        if not sub.empty:
            frames.append(sub)

    if not frames:
        raise FileNotFoundError("No manifests found for registry build")

    # F943 bulk graphs: include unlabeled specimens for field-breadth accounting
    f943_path = Path(r"E:/MetastaticEvolution/artifacts/layer1/f943_d_merged/graphs_manifest.csv")
    if f943_path.is_file():
        f943 = _load_manifest(f943_path, "f943_all", "F")
        if not f943.empty:
            frames.append(f943)

    all_df = pd.concat(frames, ignore_index=True)
    priority = {
        "specimen_authority": 0,
        "htan_bridge": 1,
        "graphs_manifest_287": 2,
        "cohort_l1_l2": 3,
        "f142_d_parity": 4,
        "f207_base": 5,
        "f943_all": 6,
        "f943_merged": 7,
        "f943_labeled207": 8,
        "lr_repair_207": 8,
        "f943_lr_merged": 9,
        "integrated_287": 10,
    }
    all_df["pack_priority"] = all_df["source_pack"].map(priority).fillna(99)
    all_df = all_df.sort_values(["pack_priority", "metastasis_supervision_mask"], ascending=[True, False])
    deduped = all_df.drop_duplicates(subset=["specimen_key"], keep="first").copy()

    # CELLxGENE catalog indication taxonomy (metadata-only breadth)
    cxg_path = data_root / "data/czi_cellxgene/metadata/cellxgene_curated_manifest_full.tsv"
    cxg_stats: Dict[str, Any] = {}
    if cxg_path.is_file():
        cxg = pd.read_csv(cxg_path, sep="\t", low_memory=False)
        cxg_stats = {
            "n_datasets": int(len(cxg)),
            "n_unique_diseases": int(cxg["disease"].nunique()) if "disease" in cxg.columns else None,
        }

    supervised = deduped[deduped["metastasis_supervision_mask"] > 0.5]
    pos = supervised[supervised["is_metastatic"] == 1]
    canonical = deduped[deduped["source_pack"] == "specimen_authority"]
    can_sup = canonical[canonical["metastasis_supervision_mask"] > 0.5]
    summary = {
        "n_unique_specimens": int(len(deduped)),
        "n_supervised": int(len(supervised)),
        "n_metastatic_positive": int(len(pos)),
        "canonical_specimen_authority_supervised": int(len(can_sup)),
        "canonical_metastatic_positive": int(can_sup["is_metastatic"].sum()),
        "f943_graph_specimens": int((deduped["source_pack"] == "f943_all").sum()),
        "n_portals": supervised["portal"].value_counts().to_dict(),
        "n_indications_supervised": supervised["project_id_tag"].value_counts().to_dict(),
        "n_indications_with_positive": pos["project_id_tag"].value_counts().to_dict(),
        "leakage_mode_counts": deduped["leakage_mode"].value_counts().to_dict(),
        "source_pack_counts": deduped["source_pack"].value_counts().to_dict(),
        "cellxgene_catalog": cxg_stats,
        "manifest_rows_ingested": int(len(all_df)),
    }
    return {"registry": deduped, "summary": summary, "all_rows": all_df}
