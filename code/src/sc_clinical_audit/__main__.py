"""CLI: python -m sc_clinical_audit run --data-root D:/MetastaticEvolution"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ap = argparse.ArgumentParser(prog="sc-clinical-audit", description="Run scClinicalAudit Archetype A field benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Execute full Archetype A study")
    run.add_argument("--data-root", type=Path, default=Path("D:/MetastaticEvolution"))
    run.add_argument("--figures", action="store_true")
    run.add_argument("--gpu", action="store_true")
    run.add_argument("--no-mlp", action="store_true")

    info = sub.add_parser("info", help="Print package version and estimands")
    info.add_argument("--data-root", type=Path, default=Path("D:/MetastaticEvolution"))

    args = ap.parse_args(argv)

    if args.cmd == "info":
        from src.sc_clinical_audit import ClinicalAudit

        out = Path(args.data_root) / "artifacts/scientific_analyses/archetype_a/headline_conclusions.json"
        print("scClinicalAudit Archetype A v2.0")
        if out.is_file():
            print(json.dumps(json.loads(out.read_text(encoding="utf-8")), indent=2))
        else:
            print("No headline_conclusions.json — run: python -m sc_clinical_audit run --figures")
        return 0

    from src.sc_clinical_audit import ClinicalAudit

    audit = ClinicalAudit(args.data_root.resolve())
    report = audit.run_full_study(gpu=args.gpu, canonical_mlp=not args.no_mlp)
    print(json.dumps(report.get("headline_conclusions", {}), indent=2))
    if args.figures:
        from scripts.build_archetype_a_figures import main as fig_main

        sys.argv = ["build_archetype_a_figures.py", "--data-root", str(args.data_root)]
        fig_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
