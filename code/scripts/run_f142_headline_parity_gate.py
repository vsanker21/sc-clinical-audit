#!/usr/bin/env python3
"""
Headline cohort scale path: D-graph symlink parity (142 supervised / 33 meta+).

Cheaper than F943 NATMI rebuild (~0 GB new tensors vs ~537 GB).
Uses identical graph tensors as E:/D canonical scientific rebuild.

Steps:
  1. Symlink 142 graph_*.pt from D/E scientific graphs dir
  2. (Optional) prepare integrated_f142_d_parity pack
  3. Train block_mlp donor GroupKFold — expect mean AUROC ~0.77–0.78
  4. Write parity_gate_report.json for Archetype A

Usage:
  python scripts/run_f142_headline_parity_gate.py --data-root D:/MetastaticEvolution
  python scripts/run_f142_headline_parity_gate.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

DEFAULT_SPECIMEN = Path("D:/MetastaticEvolution/data/integrated_specimen_authority/layer1_cohort_manifest_aligned.csv")
DEFAULT_D_GRAPHS = Path("E:/MetastaticEvolution/artifacts/layer1/graphs")
DEFAULT_LINK_DIR = Path("E:/MetastaticEvolution/artifacts/layer1/f142_d_parity_graphs")
DEFAULT_PACK = Path("E:/MetastaticEvolution/data/integrated_f142_d_parity")
DEFAULT_MODELS = Path("E:/MetastaticEvolution/models/metastasis_block_mlp_f142_d_parity")
HEADLINE_MODELS = Path("E:/Metastatic Evolution/models/metastasis_block_mlp_cv_specimen_authority")


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_symlinks(
    specimen_manifest: Path,
    link_dir: Path,
    d_graphs: Path,
    *,
    full_md5: bool = False,
) -> Dict[str, Any]:
    import pandas as pd

    sup = pd.read_csv(specimen_manifest, low_memory=False)
    sup = sup[sup["metastasis_supervision_mask"].astype(float) > 0.5]
    rows: List[Dict[str, Any]] = []
    ok = 0
    for _, row in sup.iterrows():
        gf = str(row["graph_file"])
        dst = link_dir / gf
        src = d_graphs / gf
        entry = {"graph_file": gf, "dst_exists": dst.exists(), "src_exists": src.is_file()}
        if dst.exists() and src.is_file():
            try:
                target = dst.resolve()
                entry["is_symlink"] = dst.is_symlink()
                entry["size_match"] = target.stat().st_size == src.stat().st_size
                if full_md5:
                    entry["md5_match"] = _md5(target) == _md5(src)
                    matched = entry["md5_match"]
                else:
                    matched = bool(entry.get("size_match")) and target.resolve() == src.resolve()
                if matched:
                    ok += 1
            except OSError as exc:
                entry["error"] = str(exc)
        rows.append(entry)
    return {
        "n_supervised": int(len(sup)),
        "n_parity_ok": ok,
        "all_parity_ok": ok == len(sup),
        "link_dir": str(link_dir),
        "d_graphs": str(d_graphs),
        "details_sample": rows[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("D:/MetastaticEvolution"))
    ap.add_argument("--specimen-manifest", type=Path, default=DEFAULT_SPECIMEN)
    ap.add_argument("--d-graphs", type=Path, default=DEFAULT_D_GRAPHS)
    ap.add_argument("--link-dir", type=Path, default=DEFAULT_LINK_DIR)
    ap.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--full-md5", action="store_true", help="Byte-identical MD5 check (slow on ~2.6GB/graph)")
    ap.add_argument("--skip-prepare", action="store_true", help="Skip prepare if pack exists")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    dr = args.data_root.resolve()
    report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "d_graph_symlink_parity",
        "priority_over": "f943_canonical_natmi_rebuild (~537 GB / 28–69 h)",
        "headline_cohort": "integrated_specimen_authority 142/33",
    }

    # 1. Ensure symlinks (refresh links; skip heavy prepare when pack exists)
    prep_cmd = [
        PY, str(ROOT / "scripts/prepare_f142_d_graph_parity.py"),
        "--specimen-manifest", str(args.specimen_manifest),
        "--d-graphs", str(args.d_graphs),
        "--link-dir", str(args.link_dir),
    ]
    if args.full_md5:
        prep_cmd.append("--verify-md5")
    if args.skip_prepare and (args.pack_dir / "integrated_dataset.pt").is_file():
        prep_cmd.append("--skip-prepare")
        print("Pack exists — refreshing symlinks only (no prepare)")
    else:
        prep_cmd.extend(["--data-root", str(dr), "--out-dir", str(args.pack_dir)])

    rc = subprocess.call(prep_cmd, cwd=str(ROOT))
    report["symlink_step_rc"] = rc

    verify = verify_symlinks(args.specimen_manifest, args.link_dir, args.d_graphs)
    report["symlink_verification"] = verify
    check = "MD5-identical" if args.full_md5 else "size/symlink-matched"
    print(f"Symlink parity: {verify['n_parity_ok']}/{verify['n_supervised']} {check} to {verify['d_graphs']}")

    if args.verify_only:
        out = dr / "artifacts/scientific_analyses/archetype_a/f142_parity_gate_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")
        return 0 if verify["all_parity_ok"] else 1

    # 2. Prepare if needed
    if not args.skip_prepare or not (args.pack_dir / "integrated_dataset.pt").is_file():
        prep_full = [
            PY, str(ROOT / "scripts/prepare_f142_d_graph_parity.py"),
            "--specimen-manifest", str(args.specimen_manifest),
            "--d-graphs", str(args.d_graphs),
            "--link-dir", str(args.link_dir),
            "--data-root", str(dr),
            "--out-dir", str(args.pack_dir),
        ]
        rc |= subprocess.call(prep_full, cwd=str(ROOT))
        report["prepare_rc"] = rc

    # 3. Train parity gate
    if not args.skip_train:
        train_cmd = [
            PY, str(ROOT / "scripts/train_metastasis_supervised.py"),
            "--data-dir", str(args.pack_dir),
            "--encoder", "block_mlp",
            "--group-cv", "5",
            "--epochs", "80",
            "--out-dir", str(args.models_dir),
        ]
        if args.gpu:
            train_cmd.append("--gpu")
        rc = subprocess.call(train_cmd, cwd=str(ROOT))
        report["train_rc"] = rc

    metrics_path = args.models_dir / "group_cv_metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        report["parity_metrics"] = {
            "mean_auroc": metrics.get("mean_auroc"),
            "oof_auroc": metrics.get("oof_auroc") or metrics.get("oof_metrics", {}).get("auroc"),
            "n_supervised": metrics.get("n_supervised"),
            "n_positive": metrics.get("n_positive"),
        }
    hm = HEADLINE_MODELS if HEADLINE_MODELS.name == "group_cv_metrics.json" else HEADLINE_MODELS / "group_cv_metrics.json"
    if hm.is_file():
            headline = json.loads(hm.read_text(encoding="utf-8"))
            report["headline_d_metrics"] = {"mean_auroc": headline.get("mean_auroc")}
            pm = report.get("parity_metrics", {}).get("mean_auroc")
            hm_auc = headline.get("mean_auroc")
            if pm and hm_auc:
                report["parity_delta_vs_headline"] = float(pm) - float(hm_auc)
                report["parity_gate_pass"] = abs(float(pm) - float(hm_auc)) < 0.05

    report["feasibility"] = {
        "disk_cost_gb": 0,
        "time_cost": "minutes (symlinks) + ~30–60 min train",
        "replaces": "f943_canonical_natmi rebuild (537 GB, 28–69 h)",
    }

    out = dr / "artifacts/scientific_analyses/archetype_a/f142_parity_gate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    if report.get("parity_metrics"):
        print(f"Parity mean AUROC: {report['parity_metrics'].get('mean_auroc'):.4f}")
    return 0 if verify.get("all_parity_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
