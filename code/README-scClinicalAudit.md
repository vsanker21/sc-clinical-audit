# scClinicalAudit (Archetype A)

Field benchmark and reusable audit API for single-cell clinical-label prediction under indication, portal, and provenance confounding.

## Install

```bash
pip install -e .
# or
pip install -r requirements-scClinicalAudit.txt
```

## Quick start

```bash
python -m sc_clinical_audit run --data-root D:/MetastaticEvolution --figures
```

Requires local integrated pack `data/integrated_specimen_authority` (graphs on disk).

## Relation to Genome Biology companion paper

The specimen-authority confound audit is the **canonical arm** embedded here. See the companion audit paper (Zenodo `10.5281/zenodo.20610218`).
