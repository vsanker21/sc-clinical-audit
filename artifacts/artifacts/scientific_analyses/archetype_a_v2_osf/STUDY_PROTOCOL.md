# scClinicalAudit Archetype A v2 — Study Protocol

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
