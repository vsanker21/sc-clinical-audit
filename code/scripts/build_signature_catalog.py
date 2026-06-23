#!/usr/bin/env python3
"""
Curate 30+ published / literature / internal signatures for scClinicalAudit graveyard.

Outputs:
  data/sc_clinical_audit/published_signatures.json
  data/sc_clinical_audit/signature_catalog_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Literature / MSigDB-style metastasis & TME gene sets (curated from published programs)
LITERATURE_GENE_SETS: List[Dict[str, Any]] = [
    {"id": "emt_core", "name": "EMT core (VIM/SNAI/TWIST/ZEB/CDH)", "source": "Taube et al. 2010; Ye & Weinberg 2015", "genes": ["VIM", "SNAI1", "SNAI2", "TWIST1", "TWIST2", "ZEB1", "ZEB2", "CDH1", "CDH2", "FN1", "MMP2", "MMP9"]},
    {"id": "metastasis_seed", "name": "Metastasis seed & prep (PTGS2/MMP9/VCAN)", "source": "Gupta & Massagué 2006 review", "genes": ["PTGS2", "MMP1", "MMP9", "VCAN", "SPARC", "CTSB", "CTSL", "PLAUR", "TGFB1", "LOX"]},
    {"id": "hypoxia_hif", "name": "Hypoxia / HIF targets", "source": "MSigDB HALLMARK_HYPOXIA (subset)", "genes": ["VEGFA", "LDHA", "PGK1", "SLC2A1", "BNIP3", "CA9", "ADM", "ENO1", "PDK1", "HK2"]},
    {"id": "angiogenesis", "name": "Angiogenesis program", "source": "MSigDB HALLMARK_ANGIOGENESIS (subset)", "genes": ["VEGFA", "KDR", "FLT1", "PECAM1", "ANGPT2", "PDGFB", "FGF2", "HIF1A", "CXCL8", "MMP9"]},
    {"id": "inflammatory_response", "name": "Inflammatory response", "source": "MSigDB HALLMARK_INFLAMMATORY_RESPONSE", "genes": ["IL6", "IL1B", "CXCL8", "CCL2", "PTGS2", "NFKBIA", "ICAM1", "SELE", "TNF", "CSF3"]},
    {"id": "tnf_nfkb", "name": "TNF-α/NFκB signaling", "source": "MSigDB HALLMARK_TNFA_NFKB", "genes": ["NFKB1", "RELA", "TNF", "IL6", "CXCL1", "CXCL2", "CCL20", "NFKBIA", "BIRC3", "PTGS2"]},
    {"id": "interferon_gamma", "name": "IFN-γ response", "source": "MSigDB HALLMARK_INTERFERON_GAMMA", "genes": ["STAT1", "IRF1", "CXCL9", "CXCL10", "CXCL11", "GBP1", "HLA-DRA", "IDO1", "WARS1", "TAP1"]},
    {"id": "complement", "name": "Complement cascade", "source": "MSigDB HALLMARK_COMPLEMENT", "genes": ["C1QA", "C1QB", "C3", "C3AR1", "CFB", "CFD", "SERPING1", "ITGAM", "CR1", "C5AR1"]},
    {"id": "coagulation", "name": "Coagulation / platelet", "source": "MSigDB HALLMARK_COAGULATION", "genes": ["F2", "F3", "F5", "F8", "F10", "SERPINE1", "PLAT", "PLAU", "THBS1", "VWF"]},
    {"id": "epithelial_mesenchymal_msigdb", "name": "EMT MSigDB hallmark", "source": "MSigDB HALLMARK_EMT", "genes": ["CDH2", "FN1", "VIM", "SNAI2", "ZEB1", "ZEB2", "COL1A1", "COL3A1", "MMP2", "TGFB1", "SPARC", "ITGA5"]},
    {"id": "melanoma_metastasis", "name": "Melanoma metastasis (MITF/AXL switch)", "source": "Hugo et al. 2013; Tsoi et al. 2018", "genes": ["MITF", "AXL", "TGFB1", "WNT5A", "EGFR", "PDGFRB", "NGFR", "SOX10", "SOX9", "CDH1"]},
    {"id": "breast_metastasis_lung", "name": "Breast lung metastasis seeding", "source": "Minn et al. 2005 (subset)", "genes": ["CXCR4", "MMP1", "MMP2", "ERBB2", "PTGS2", "RHOC", "ID1", "IL13RA2", "COL1A1", "LGALS3"]},
    {"id": "tgf_beta", "name": "TGF-β signaling", "source": "KEGG TGF-beta", "genes": ["TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4", "SERPINE1", "THBS1", "CDKN2B"]},
    {"id": "wnt_beta_catenin", "name": "Wnt/β-catenin", "source": "KEGG Wnt", "genes": ["CTNNB1", "LEF1", "TCF7", "AXIN2", "MYC", "CCND1", "FZD1", "FZD7", "DKK1", "WNT5A"]},
    {"id": "jak_stat", "name": "JAK-STAT cytokine", "source": "KEGG JAK-STAT", "genes": ["JAK1", "JAK2", "STAT1", "STAT3", "SOCS1", "SOCS3", "IL6ST", "IL10", "IFNG", "BCL2L1"]},
]

TME_CELL_TYPE_SETS: List[Dict[str, Any]] = [
    {"id": "tme_t_cell", "name": "T cell infiltration", "source": "TME literature proxy", "cell_types": ["T cell", "CD8 T cell", "CD4 T cell"]},
    {"id": "tme_nk", "name": "NK cell infiltration", "source": "TME literature proxy", "cell_types": ["natural killer cell", "NK cell"]},
    {"id": "tme_treg", "name": "Regulatory T cells", "source": "TME suppression literature", "cell_types": ["regulatory T cell"]},
    {"id": "tme_macrophage", "name": "Macrophage / monocyte", "source": "TME literature proxy", "cell_types": ["macrophage", "monocyte"]},
    {"id": "tme_dc", "name": "Dendritic cells", "source": "TME antigen presentation", "cell_types": ["dendritic cell", "conventional dendritic cell"]},
    {"id": "tme_b_cell", "name": "B cell infiltration", "source": "TME literature proxy", "cell_types": ["B cell", "plasma cell"]},
    {"id": "tme_fibroblast", "name": "CAF / fibroblast stroma", "source": "Stromal literature proxy", "cell_types": ["fibroblast", "cancer-associated fibroblast", "stromal cell"]},
    {"id": "tme_endothelial", "name": "Endothelial / vascular", "source": "Angiogenesis TME proxy", "cell_types": ["endothelial cell", "vascular endothelial cell"]},
    {"id": "tme_epithelial", "name": "Epithelial tumor fraction", "source": "TME composition proxy", "cell_types": ["epithelial cell", "malignant cell", "tumor cell"]},
    {"id": "tme_myeloid", "name": "Myeloid compartment", "source": "TME literature proxy", "cell_types": ["macrophage", "monocyte", "neutrophil", "myeloid cell"]},
]


def _gsea_pathway_signatures(gsea_csv: Path, n: int = 12) -> List[Dict[str, Any]]:
    if not gsea_csv.is_file():
        return []
    df = pd.read_csv(gsea_csv)
    out: List[Dict[str, Any]] = []
    for i, row in df.head(n).iterrows():
        term = str(row.get("Term", f"pathway_{i}"))
        lead = str(row.get("Lead_genes", ""))
        genes = [g.strip() for g in lead.split(";") if g.strip()]
        sid = re.sub(r"[^a-z0-9]+", "_", term.lower())[:48]
        out.append({
            "id": f"gsea_kegg_{sid}",
            "name": f"GSEA KEGG: {term[:60]}",
            "source": f"Internal GSEA prerank (NES={float(row.get('NES', 0)):.2f})",
            "type": "gene_list",
            "genes": genes[:30],
        })
    return out


def _de_top_gene_panels(de_csv: Path, panels: List[int] = (25, 50, 100)) -> List[Dict[str, Any]]:
    if not de_csv.is_file():
        return []
    df = pd.read_csv(de_csv)
    gcol = "gene" if "gene" in df.columns else df.columns[0]
    lfc = "log_fold_change" if "log_fold_change" in df.columns else "log2FoldChange"
    df = df.sort_values(lfc, key=lambda s: s.abs(), ascending=False)
    out = []
    for n in panels:
        genes = df.head(n)[gcol].astype(str).tolist()
        out.append({
            "id": f"pan_de_top{n}",
            "name": f"Pan-cancer DE top-{n} (bulk paired)",
            "source": str(de_csv.name),
            "type": "gene_list",
            "genes": genes,
        })
    return out


def build_catalog(data_root: Path) -> Dict[str, Any]:
    sci = data_root / "artifacts/scientific_analyses"
    gsea_csv = sci / "preregistered_study/gsea/gsea_prerank_kegg_results.csv"
    de_csv = data_root / "artifacts/tcga_paired_transcriptomics/layer3/signatures/pan_metastatic_vs_primary_pancancer_v2.csv"

    entries: List[Dict[str, Any]] = []

    for g in LITERATURE_GENE_SETS:
        entries.append({**g, "type": "gene_list"})

    for c in TME_CELL_TYPE_SETS:
        entries.append({**c, "type": "cell_type_composition"})

    entries.extend(_gsea_pathway_signatures(gsea_csv, n=10))
    entries.extend(_de_top_gene_panels(de_csv, panels=[25, 50, 100]))

    # Layer-3 projection proxy
    entries.append({
        "id": "layer3_pan_projection",
        "name": "Layer-3 pan-cancer projection (mean)",
        "source": "Internal integrated layer3_features",
        "type": "layer3_de",
        "de_csv": "artifacts/tcga_paired_transcriptomics/layer3/signatures/pan_metastatic_vs_primary_pancancer_v2.csv",
    })

    # Block-mean full TME (baseline comparator)
    entries.append({
        "id": "block_mean_full_tme",
        "name": "Full block-mean TME vector (comparator)",
        "source": "Headline block_mlp input space",
        "type": "block_mean_comparator",
    })

    return {
        "version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_signatures": len(entries),
        "signatures": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("D:/MetastaticEvolution"))
    args = ap.parse_args()
    dr = args.data_root.resolve()
    out_dir = ROOT / "data/sc_clinical_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = build_catalog(dr)
    sig_path = out_dir / "published_signatures.json"
    sig_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "catalog_path": str(sig_path),
        "n_signatures": catalog["n_signatures"],
        "breakdown": {
            "literature_gene_sets": len(LITERATURE_GENE_SETS),
            "tme_cell_type": len(TME_CELL_TYPE_SETS),
            "gsea_kegg": sum(1 for s in catalog["signatures"] if str(s["id"]).startswith("gsea_")),
            "de_panels": sum(1 for s in catalog["signatures"] if str(s["id"]).startswith("pan_de_")),
        },
        "usage": "Loaded by src.sc_clinical_audit.signatures.load_signature_catalog()",
    }
    (out_dir / "signature_catalog_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {sig_path} ({catalog['n_signatures']} signatures)")
    print(json.dumps(manifest["breakdown"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
