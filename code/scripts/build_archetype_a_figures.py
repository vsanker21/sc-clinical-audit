#!/usr/bin/env python3
"""Archetype A publication figures from archetype_a_report.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {
    "indication": "#E69F00",
    "portal": "#56B4E9",
    "compositional": "#009E73",
    "provenance": "#CC79A7",
    "residual": "#0072B2",
    "neutral": "#999999",
    "chance": "#D55E00",
}


def _load_report(dr: Path) -> Dict[str, Any]:
    p = dr / "artifacts/scientific_analyses/archetype_a/archetype_a_report.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _save_fig(fig, out: Path, stem: str, *, pdf: bool = False) -> None:
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=300)
    if pdf:
        try:
            fig.savefig(out / f"{stem}.pdf")
        except OSError as exc:
            print(f"Warning: PDF save skipped for {stem}: {exc}")
    plt.close(fig)


def fig1_decomposition_spectrum(report: Dict[str, Any], out: Path) -> None:
    packs = [p for p in report.get("pack_decompositions", []) if not p.get("skipped")]
    if not packs:
        return
    names, stacks = [], []
    for p in packs:
        d = p.get("decomposition", {})
        shares = d.get("explained_shares_above_chance", {})
        names.append(f"{p['drive']}\n{p['pack'][:18]}")
        stacks.append([
            shares.get("indication", 0),
            shares.get("portal", 0),
            shares.get("compositional", 0),
            shares.get("provenance", 0),
            d.get("residual_biological", 0),
        ])
    stacks = np.array(stacks)
    base = np.ones(len(names)) * 0.5
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["Indication", "Portal", "Compositional", "Provenance", "Residual bio."]
    colors = [C["indication"], C["portal"], C["compositional"], C["provenance"], C["residual"]]
    left = base.copy()
    for i, (lab, col) in enumerate(zip(labels, colors)):
        ax.barh(range(len(names)), stacks[:, i], left=left, color=col, label=lab, height=0.6)
        left += stacks[:, i]
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("AUROC (stacked above chance = 0.5)")
    ax.set_xlim(0.5, min(1.05, left.max() + 0.05))
    ax.axvline(0.5, color=C["neutral"], ls=":", lw=1)
    ax.set_title("Confound decomposition spectrum across multi-drive integrated packs")
    ax.legend(loc="lower right", fontsize=8)
    _save_fig(fig, out, "FigA1_decomposition_spectrum")


def fig2_transport_matrix(report: Dict[str, Any], out: Path) -> None:
    t = report.get("transport_indication", {})
    mat = t.get("matrix")
    if not mat:
        return
    strata = t["strata"]
    arr = np.array([[mat.get(a, {}).get(b, np.nan) for b in strata] for a in strata], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(arr, vmin=0.4, vmax=1.0, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(len(strata)))
    ax.set_yticks(range(len(strata)))
    ax.set_xticklabels([s.replace("TCGA-", "") for s in strata], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([s.replace("TCGA-", "") for s in strata], fontsize=7)
    ax.set_xlabel("Test indication")
    ax.set_ylabel("Train indication")
    ax.set_title("Transportability matrix (train → test AUROC)")
    plt.colorbar(im, ax=ax, label="AUROC")
    for i in range(len(strata)):
        for j in range(len(strata)):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="black")
    _save_fig(fig, out, "FigA2_transportability_matrix")


def fig3_signature_graveyard(report: Dict[str, Any], out: Path) -> None:
    sigs = report.get("signature_reevaluation", [])
    if not sigs:
        return
    names = [s["name"][:32] for s in sigs][::-1]
    naive = [s.get("naive_oof_auroc") or 0.5 for s in sigs][::-1]
    corrected = [s.get("corrected_within_indication_auroc") or 0.5 for s in sigs][::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.35)))
    ax.scatter(naive, y, color=C["chance"], s=60, label="Naive pan-cohort", zorder=3)
    ax.scatter(corrected, y, color=C["residual"], s=60, label="Within-indication corrected", zorder=3)
    for i in range(len(names)):
        ax.plot([naive[i], corrected[i]], [y[i], y[i]], color=C["neutral"], lw=1, alpha=0.5)
    ax.axvline(0.5, color=C["neutral"], ls=":", lw=1)
    canon = next((p for p in report.get("pack_decompositions", []) if p.get("pack") == "specimen_authority"), {})
    ceiling = canon.get("decomposition", {}).get("metadata_ceiling_auroc", 0.83)
    ax.axvline(ceiling, color=C["indication"], ls="--", lw=1, label=f"Metadata ceiling ({ceiling:.2f})")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("OOF AUROC")
    ax.set_xlim(0.45, 1.0)
    ax.set_title("Signature graveyard: apparent vs confound-corrected performance")
    ax.legend(loc="lower right", fontsize=8)
    _save_fig(fig, out, "FigA3_signature_graveyard")


def fig4_leakage_taxonomy(report: Dict[str, Any], out: Path) -> None:
    summ = report.get("registry_summary", {})
    modes = summ.get("leakage_mode_counts", {})
    if not modes:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels, vals = zip(*sorted(modes.items(), key=lambda x: -x[1]))
    axes[0].barh(range(len(labels)), vals, color=C["portal"])
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Specimens")
    axes[0].set_title("Leakage / label-provenance taxonomy")
    ind = summ.get("n_indications_supervised", {})
    if ind:
        il, iv = zip(*sorted(ind.items(), key=lambda x: -x[1])[:12])
        axes[1].bar(range(len(il)), iv, color=C["indication"])
        axes[1].set_xticks(range(len(il)))
        axes[1].set_xticklabels([x.replace("TCGA-", "") for x in il], rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("Supervised n")
        axes[1].set_title("Supervised specimens by indication")
    fig.suptitle("Registry breadth across C/D/E/F manifests", fontweight="bold")
    _save_fig(fig, out, "FigA4_leakage_taxonomy")


def fig5_positive_control(report: Dict[str, Any], out: Path) -> None:
    tcga = report.get("tcga_bulk_positive_control", {})
    primary = tcga.get("primary_arm", {})
    if primary.get("skipped") or not primary.get("oof_auroc"):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    canon = next((p for p in report.get("pack_decompositions", []) if p.get("pack") == "specimen_authority"), {})
    sc_auc = canon.get("decomposition", {}).get("task_auroc", 0.67)
    brca_rows = [r for r in canon.get("per_indication", []) if "BRCA" in str(r.get("indication", ""))]
    brca_auc = brca_rows[0].get("apparent_auroc", 0.6) if brca_rows else 0.6
    bulk_auc = primary.get("oof_auroc", 0.5)
    bulk_lab = primary.get("within_indication", "TCGA bulk").replace("TCGA-", "")
    vals = [bulk_auc, sc_auc, brca_auc]
    colors = [C["residual"], C["chance"], C["portal"]]
    ax.bar([0, 1, 2], vals, color=colors, width=0.5)
    labels = [f"TCGA {bulk_lab} bulk\n(balanced pairs)", "Pan-cohort scRNA\n(confounded)", "Within-BRCA scRNA\n(underpowered)"]
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.5, color=C["neutral"], ls=":", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("OOF / holdout AUROC")
    ax.set_title("Positive control: bulk within-indication vs scRNA confound strata")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    _save_fig(fig, out, "FigA5_positive_control")


def fig6_metadata_ceiling_comparison(report: Dict[str, Any], out: Path) -> None:
    canon = next((p for p in report.get("pack_decompositions", []) if p.get("pack") == "specimen_authority"), {})
    baselines = canon.get("metadata_baselines", {})
    if not baselines:
        return
    names, vals = [], []
    mapping = [
        ("block_logistic", "TME blocks"),
        ("indication", "Indication only"),
        ("portal", "Portal only"),
        ("compositional", "Composition only"),
        ("provenance", "Provenance only"),
        ("metadata_combined", "Combined metadata"),
    ]
    for key, lab in mapping:
        m = baselines.get(key, {})
        auc = m.get("oof_metrics", {}).get("auroc")
        if auc is not None:
            names.append(lab)
            vals.append(auc)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    cols = [C["residual"], C["indication"], C["portal"], C["compositional"], C["provenance"], C["chance"]]
    ax.bar(x, vals, color=cols[: len(names)], width=0.55)
    ax.axhline(0.5, color=C["neutral"], ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("OOF AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Adversarial confound ceiling (canonical specimen-authority cohort)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    _save_fig(fig, out, "FigA6_confound_ceiling")


def figS1_portal_transport(report: Dict[str, Any], out: Path) -> None:
    t = report.get("transport_portal", {})
    mat = t.get("matrix")
    if not mat:
        return
    strata = t.get("strata", [])
    if len(strata) < 2:
        return
    arr = np.array([[mat.get(a, {}).get(b, np.nan) for b in strata] for a in strata], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, vmin=0.4, vmax=1.0, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(len(strata)))
    ax.set_yticks(range(len(strata)))
    labels = [s[:28] + "…" if len(s) > 28 else s for s in strata]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Test portal stratum")
    ax.set_ylabel("Train portal stratum")
    ax.set_title("Portal-stratified transportability (canonical cohort)")
    plt.colorbar(im, ax=ax, label="AUROC")
    _save_fig(fig, out, "FigS1_portal_transport")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("D:/MetastaticEvolution"))
    ap.add_argument("--out-dir", type=Path, default=None, help="Figure output (default: data-root/.../figures)")
    ap.add_argument("--pdf", action="store_true", help="Also write PDF (requires disk space)")
    args = ap.parse_args()
    dr = args.data_root.resolve()
    report = _load_report(dr)
    out = args.out_dir or (dr / "artifacts/scientific_analyses/archetype_a/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig1_decomposition_spectrum(report, out)
    fig2_transport_matrix(report, out)
    fig3_signature_graveyard(report, out)
    fig4_leakage_taxonomy(report, out)
    fig5_positive_control(report, out)
    fig6_metadata_ceiling_comparison(report, out)
    figS1_portal_transport(report, out)
    print(f"Wrote Archetype A figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
