#!/usr/bin/env python3
"""
Build OSF-ready Archetype A v2 pre-registration bundle (internal snapshot + upload instructions).

Outputs: <data-root>/artifacts/scientific_analyses/archetype_a_v2_osf/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PREREG_V2 = {
    "study_id": "scClinicalAudit_ArchetypeA_v2",
    "version": "2.0",
    "registered_utc": None,  # filled at build time
    "registry": "OSF (target) — https://osf.io/",
    "status": "internal_snapshot_pending_osf_upload",
    "title": "scClinicalAudit: A causal evaluation framework for single-cell clinical prediction under cancer-type and portal confounding",
    "authors": ["Sanker, V. (TN Omics)"],
    "primary_estimands": [
        {
            "id": "E1_confound_fraction",
            "definition": "Fraction of (task_OOF_AUROC - 0.5) explained by metadata ceiling AUROC",
            "estimator": "decompose_auroc_components() in sc_clinical_audit",
        },
        {
            "id": "E2_transportability",
            "definition": "Train-stratum → test-stratum AUROC minus test-stratum metadata ceiling",
            "estimator": "transport_matrix() donor-grouped block-logistic",
        },
        {
            "id": "E3_signature_survival",
            "definition": "Proportion of curated signatures with corrected_within_indication_AUROC within 0.15 of naive and >= 0.55",
            "estimator": "score_signatures_on_pack() graveyard protocol",
        },
        {
            "id": "E4_positive_control",
            "definition": "Balanced TCGA bulk within-indication OOF AUROC >= 0.65 (BRCA/THCA arms)",
            "estimator": "case-grouped logistic on expression_sample_features",
        },
    ],
    "exclusion_criteria": [
        "Bare disease-string metastasis labels (specimen-authority policy)",
        "HTAPP holdout with >90% prevalence as external validation",
        "F943 graphs without H5AD stems (736 unpaired rows)",
        "Non-donor-grouped CV for headline metrics",
    ],
    "analysis_sequence": [
        "0. Headline scale: f142 D-graph symlink parity gate (142/33; 0 GB)",
        "1. Build multi-drive specimen registry (specimen_registry.csv)",
        "2. Decomposition + metadata ceilings per integrated pack",
        "3. Transport matrices (indication × indication; portal × portal)",
        "4. Signature graveyard (≥30 curated signatures)",
        "5. TCGA bulk positive controls (balanced arms only)",
        "6. Archetype A figures FigA1–A6",
    ],
    "headline_scale_path": {
        "strategy": "f142_d_parity_symlink_parity",
        "script": "scripts/run_f142_headline_parity_gate.py",
        "pack": "integrated_f142_d_parity",
        "note": "Preferred over F943 NATMI rebuild (~537 GB); symlinks E canonical graph_*.pt",
    },
    "data_roots": {
        "primary": "D:/MetastaticEvolution",
        "graphs_canonical": "E:/MetastaticEvolution/artifacts/layer1/graphs",
        "f_scale": "F:/Metastatic Evolution/artifacts/layer1/graphs",
        "lr_repair_staging": "C:/MetastaticEvolution/artifacts/layer1/graphs_with_lr_repair",
    },
    "software": {
        "package": "src/sc_clinical_audit",
        "orchestrator": "scripts/run_archetype_a_study.py",
        "signature_catalog": "data/sc_clinical_audit/published_signatures.json",
    },
    "locked_constants": {
        "cv": "GroupKFold k=5 grouped by donor_id",
        "headline_model": "block_mlp 256-128-1",
        "canonical_pack": "integrated_specimen_authority",
        "metadata_ceiling_features": ["project_id", "portal", "label_source_field", "label_source_value"],
    },
}


STUDY_PROTOCOL_MD = """# scClinicalAudit Archetype A v2 — Study Protocol

## Registration status
This bundle is an **internal pre-registration snapshot** prepared for OSF upload.
Upload to https://osf.io/ **before** scaling analyses or manuscript submission claiming registration.

## Hypothesis
Apparent single-cell clinical prediction in public atlases is systematically confounded by
indication, portal/batch, compositional structure, and label-provenance leakage; a formal
decomposition and transportability benchmark quantifies this as a field-level failure mode.

## Primary estimands
1. **Confound fraction** — share of apparent AUROC attributable to metadata ceilings
2. **Transportability** — train/test indication and portal matrices
3. **Signature survival** — graveyard re-evaluation of ≥30 published/internal signatures
4. **Positive control** — balanced TCGA bulk within-indication discrimination

## Data (locked at v2)
- Canonical: `integrated_specimen_authority` (142 supervised / 33 meta+)
- Headline scale proxy: `integrated_f142_d_parity` (symlink parity to E canonical graphs)
- Registry breadth: multi-drive C/D/E/F manifests + F943 graph inventory
- Signatures: `data/sc_clinical_audit/published_signatures.json`

## Analysis code (frozen paths)
```bash
python scripts/run_f142_headline_parity_gate.py --data-root D:/MetastaticEvolution --skip-prepare --skip-train
python scripts/build_signature_catalog.py --data-root D:/MetastaticEvolution
python scripts/run_archetype_a_study.py --data-root D:/MetastaticEvolution --figures
python scripts/build_archetype_a_v2_osf_bundle.py --data-root D:/MetastaticEvolution
```

## Success criteria (pre-specified)
| Gate | Threshold |
|------|-----------|
| Registry supervised | ≥142 canonical; aspirational ≥1,500 multi-portal |
| Confound fraction (canonical) | report with CI; expect ≥0.5 |
| Signatures tested | ≥30 |
| Signatures surviving correction | report count (exploratory) |
| TCGA positive control | ≥1 balanced arm with OOF AUROC ≥0.65 |

## Deviations log
Record any post-registration changes in `deviations_log.md` in this bundle.
"""


OSF_UPLOAD_MD = """# OSF upload checklist — Archetype A v2

1. Create OSF project: **scClinicalAudit Archetype A v2**
2. Add component **Pre-registration** (upload before running scaled analyses):
   - `preregistration_protocol.json`
   - `STUDY_PROTOCOL.md`
   - `analysis_plan.md`
3. Add component **Code snapshot** (tag release in Git when public repo exists):
   - `src/sc_clinical_audit/`
   - `scripts/run_archetype_a_study.py`
   - `scripts/build_signature_catalog.py`
4. Add component **Data inventory** (no PHI):
   - `registry_summary.json`
   - `f943_rebuild_estimate.json`
5. Add component **Results** (post-execution snapshot):
   - `execution_summary.json`
   - `headline_conclusions.json`
   - `decomposition_by_pack.csv`
   - `signature_graveyard.csv`
   - `f142_parity_gate_report.json`
   - `figures/` (FigA1–A6)
6. Register DOI; paste OSF URL into manuscript Data Availability.

**Do not** claim public pre-registration until OSF DOI is live.
"""


DEVIATIONS_LOG = """# Deviations log

| Date | Deviation | Rationale |
|------|-----------|-----------|
| 2026-06-06 | F943 canonical NATMI rebuild deferred | ~537 GB / 28–69 h; C: drive lacks headroom; parity path validates 142/33 at ~0.78 AUROC with 0 GB |
| 2026-06-06 | Headline scale uses symlink parity pack (`f142_d_parity_headline_scale`) | Tensor identity to E canonical scientific rebuild; 142/142 symlinks verified |
| 2026-06-06 | Signature survival criteria exploratory | 31/40 pass loose Δ<0.15 & corrected≥0.55; manuscript reports graveyard not discovery |
"""


def _build_execution_summary(dr: Path) -> dict:
    report_path = dr / "artifacts/scientific_analyses/archetype_a/archetype_a_report.json"
    if not report_path.is_file():
        return {"status": "missing_report", "path": str(report_path)}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    hc = report.get("headline_conclusions", {})
    reg = report.get("registry_summary", {})
    gates = {
        "canonical_supervised_ge_142": {
            "threshold": 142,
            "value": reg.get("canonical_specimen_authority_supervised"),
            "pass": (reg.get("canonical_specimen_authority_supervised") or 0) >= 142,
        },
        "confound_fraction_ge_0_5": {
            "threshold": 0.5,
            "value": hc.get("canonical_confound_fraction"),
            "pass": (hc.get("canonical_confound_fraction") or 0) >= 0.5,
        },
        "signatures_tested_ge_30": {
            "threshold": 30,
            "value": hc.get("signatures_tested"),
            "pass": (hc.get("signatures_tested") or 0) >= 30,
        },
        "tcga_positive_control_ge_0_65": {
            "threshold": 0.65,
            "value": hc.get("tcga_bulk_positive_control"),
            "pass": (hc.get("tcga_bulk_positive_control") or 0) >= 0.65,
        },
        "f142_parity_gate": {
            "threshold": True,
            "value": hc.get("headline_scale_parity_gate_pass"),
            "pass": bool(hc.get("headline_scale_parity_gate_pass")),
        },
    }
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "study_report_utc": report.get("generated_utc"),
        "framework": report.get("framework"),
        "success_gates": gates,
        "all_primary_gates_pass": all(g["pass"] for g in gates.values()),
        "headline_conclusions": hc,
        "exploratory": {
            "signatures_surviving_correction": hc.get("signatures_surviving_correction"),
            "headline_scale_task_auroc": hc.get("headline_scale_task_auroc"),
            "registry_unique_specimens": hc.get("registry_unique_specimens"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("D:/MetastaticEvolution"))
    args = ap.parse_args()
    dr = args.data_root.resolve()
    out = dr / "artifacts/scientific_analyses/archetype_a_v2_osf"
    out.mkdir(parents=True, exist_ok=True)

    prereg = dict(PREREG_V2)
    prereg["registered_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "preregistration_protocol.json").write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")
    (out / "STUDY_PROTOCOL.md").write_text(STUDY_PROTOCOL_MD, encoding="utf-8")
    (out / "OSF_UPLOAD_INSTRUCTIONS.md").write_text(OSF_UPLOAD_MD, encoding="utf-8")
    (out / "analysis_plan.md").write_text(
        "## Analysis plan\n\n" + "\n".join(f"- {s}" for s in prereg["analysis_sequence"]) + "\n",
        encoding="utf-8",
    )
    (out / "deviations_log.md").write_text(DEVIATIONS_LOG, encoding="utf-8")

    aa = dr / "artifacts/scientific_analyses/archetype_a"
    copy_map = [
        (aa / "registry_summary.json", out / "registry_summary.json"),
        (aa / "f943_rebuild_estimate.json", out / "f943_rebuild_estimate.json"),
        (aa / "f142_parity_gate_report.json", out / "f142_parity_gate_report.json"),
        (aa / "decomposition_by_pack.csv", out / "decomposition_by_pack.csv"),
        (aa / "signature_graveyard.csv", out / "signature_graveyard.csv"),
        (aa / "transport_matrix_indication.csv", out / "transport_matrix_indication.csv"),
    ]
    for src, dst in copy_map:
        if src.is_file():
            shutil.copy2(src, dst)

    report_path = aa / "archetype_a_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        hc = report.get("headline_conclusions", {})
        (out / "headline_conclusions.json").write_text(json.dumps(hc, indent=2) + "\n", encoding="utf-8")

    exec_summary = _build_execution_summary(dr)
    lineage = {
        "canonical_arm": "confound_audit (specimen_authority 142/33)",
        "generalization": "archetype_a (multi-drive registry + transport + signatures + TCGA)",
        "confound_audit_report": str(dr / "artifacts/scientific_analyses/confound_audit/confound_audit_report.json"),
        "archetype_a_report": str(aa / "archetype_a_report.json"),
        "manuscript_word": str(dr / "artifacts/scientific_analyses/manuscript_word/manuscript_main.docx"),
        "relation": "Archetype A embeds confound_audit OOF estimand as canonical pack; adds field-breadth registry, scale probes, and signature graveyard.",
    }
    (aa / "study_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    (out / "study_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")

    (out / "execution_summary.json").write_text(json.dumps(exec_summary, indent=2) + "\n", encoding="utf-8")
    (aa / "execution_summary.json").write_text(json.dumps(exec_summary, indent=2) + "\n", encoding="utf-8")

    hc = exec_summary.get("headline_conclusions", {})
    md = [
        "# Archetype A — execution summary",
        "",
        f"Generated: {exec_summary.get('generated_utc', 'n/a')}",
        f"Primary gates pass: **{exec_summary.get('all_primary_gates_pass')}**",
        "",
        "## Headline estimands",
        f"- Registry: {hc.get('registry_unique_specimens')} unique specimens",
        f"- Canonical supervised: OOF AUROC **{hc.get('canonical_task_auroc', 0):.3f}**, confound fraction **{hc.get('canonical_confound_fraction')}**",
        f"- Headline scale (f142 parity): AUROC **{hc.get('headline_scale_task_auroc', 0):.3f}**, gate pass **{hc.get('headline_scale_parity_gate_pass')}**",
        f"- Signatures: {hc.get('signatures_surviving_correction')}/{hc.get('signatures_tested')} survive correction",
        f"- TCGA bulk positive control: **{hc.get('tcga_bulk_positive_control', 0):.3f}**",
        "",
        "## Next step",
        "Upload `archetype_a_v2_osf/` to OSF and register DOI before claiming pre-registration in manuscript.",
        "",
    ]
    summary_md = "\n".join(md)
    (out / "EXECUTION_SUMMARY.md").write_text(summary_md, encoding="utf-8")
    (aa / "EXECUTION_SUMMARY.md").write_text(summary_md, encoding="utf-8")

    fig_src = aa / "figures"
    fig_dst = out / "figures"
    if fig_src.is_dir():
        if fig_dst.exists():
            shutil.rmtree(fig_dst)
        shutil.copytree(fig_src, fig_dst)

    sig_src = ROOT / "data/sc_clinical_audit/published_signatures.json"
    if sig_src.is_file():
        shutil.copy2(sig_src, out / "published_signatures.json")

    manifest = {
        "bundle": "archetype_a_v2_osf",
        "generated_utc": prereg["registered_utc"],
        "out_dir": str(out),
        "execution_complete": exec_summary.get("all_primary_gates_pass"),
        "files": [p.name for p in sorted(out.iterdir()) if p.is_file()],
        "subdirs": [p.name for p in sorted(out.iterdir()) if p.is_dir()],
        "next_step": "Upload to OSF per OSF_UPLOAD_INSTRUCTIONS.md; obtain DOI before manuscript claims registration",
    }
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote OSF bundle to {out}")
    print(f"Files: {', '.join(manifest['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
