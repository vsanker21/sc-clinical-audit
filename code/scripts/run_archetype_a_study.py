#!/usr/bin/env python3
"""
Execute Archetype A: scClinicalAudit field benchmark across C/D/E/F integrated assets.

Outputs: <data-root>/artifacts/scientific_analyses/archetype_a/
  specimen_registry.parquet
  registry_summary.json
  archetype_a_report.json
  decomposition_by_pack.csv
  transport_matrix.csv
  signature_graveyard.csv
  figures/FigA1–A6

Usage:
  python scripts/run_archetype_a_study.py --data-root D:/MetastaticEvolution
  python scripts/run_archetype_a_study.py --data-root D:/MetastaticEvolution --figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sc_clinical_audit.audit import ClinicalAudit


def _export_tables(report: dict, out: Path) -> None:
    # decomposition by pack
    rows = []
    for p in report.get("pack_decompositions", []):
        if p.get("skipped"):
            continue
        d = p.get("decomposition", {})
        rows.append({
            "pack": p.get("pack"),
            "drive": p.get("drive"),
            "task_auroc": d.get("task_auroc"),
            "metadata_ceiling": d.get("metadata_ceiling_auroc"),
            "indication_auroc": d.get("component_aurocs", {}).get("indication"),
            "portal_auroc": d.get("component_aurocs", {}).get("portal"),
            "compositional_auroc": d.get("component_aurocs", {}).get("compositional"),
            "provenance_auroc": d.get("component_aurocs", {}).get("provenance"),
            "residual_biological": d.get("residual_biological"),
            "confound_fraction": d.get("confound_fraction"),
        })
    if rows:
        pd.DataFrame(rows).to_csv(out / "decomposition_by_pack.csv", index=False)

    # transport
    t = report.get("transport_indication", {})
    if t.get("matrix"):
        strata = t["strata"]
        mat = t["matrix"]
        tdf = pd.DataFrame(
            [[mat.get(a, {}).get(b) for b in strata] for a in strata],
            index=strata,
            columns=strata,
        )
        tdf.to_csv(out / "transport_matrix_indication.csv")

    # signatures
    sigs = report.get("signature_reevaluation", [])
    if sigs:
        pd.DataFrame(sigs).to_csv(out / "signature_graveyard.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("D:/MetastaticEvolution"))
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no-mlp", action="store_true", help="Use block-logistic only (faster)")
    ap.add_argument("--figures", action="store_true", help="Build Archetype A figures after analysis")
    ap.add_argument("--osf-bundle", action="store_true", help="Build OSF upload bundle after analysis")
    args = ap.parse_args()

    audit = ClinicalAudit(args.data_root.resolve())
    report = audit.run_full_study(gpu=args.gpu, canonical_mlp=not args.no_mlp)
    out = audit.out_dir
    _export_tables(report, out)
    print(f"Wrote {out / 'archetype_a_report.json'}")
    print(f"Registry: {report['registry_summary']['n_unique_specimens']} unique specimens, "
          f"{report['registry_summary']['n_supervised']} supervised")
    hc = report.get("headline_conclusions", {})
    print(f"Canonical confound fraction: {hc.get('canonical_confound_fraction', 'n/a')}")
    print(f"Signatures surviving correction: {hc.get('signatures_surviving_correction', 0)}/"
          f"{hc.get('signatures_tested', 0)}")

    if args.figures:
        from scripts.build_archetype_a_figures import main as fig_main
        import sys as _sys
        _sys.argv = ["build_archetype_a_figures.py", "--data-root", str(args.data_root)]
        fig_main()

    if args.osf_bundle:
        from scripts.build_archetype_a_v2_osf_bundle import main as bundle_main
        import sys as _sys
        _sys.argv = ["build_archetype_a_v2_osf_bundle.py", "--data-root", str(args.data_root)]
        bundle_main()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
