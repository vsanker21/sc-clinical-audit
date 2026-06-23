"""
scClinicalAudit — causal evaluation framework for single-cell clinical prediction.

Decomposes apparent predictive signal into indication, portal, compositional,
provenance, and residual-biological components under donor-grouped cross-validation.
"""

from src.sc_clinical_audit.audit import ClinicalAudit
from src.sc_clinical_audit.registry import build_multi_drive_registry

__all__ = ["ClinicalAudit", "build_multi_drive_registry"]
