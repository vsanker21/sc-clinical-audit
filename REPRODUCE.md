# Reproduce — scClinicalAudit Archetype A v2

## Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r code/requirements-scClinicalAudit.txt
pip install python-docx pytest
```

## Headline metrics (no retrain)
```bash
python -c "import json; print(json.load(open('artifacts/archetype_a/headline_conclusions.json')))"
```

## Full study (requires D:/MetastaticEvolution data packs on disk)
```bash
python code/scripts/run_archetype_a_study.py --data-root D:/MetastaticEvolution --figures --osf-bundle
python code/scripts/build_archetype_a_manuscript.py --data-root D:/MetastaticEvolution
```

## Companion Genome Biology audit
Zenodo: https://doi.org/10.5281/zenodo.20610218
GitHub: https://github.com/vsanker21/scrna-metastasis-confound-audit
