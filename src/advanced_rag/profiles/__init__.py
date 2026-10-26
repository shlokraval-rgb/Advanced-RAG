"""One profile per document. get_profile('gpo' | 'gep' | 'imac')."""
from __future__ import annotations


def get_profile(doc_id: str):
    if doc_id == "gpo":
        from .gpo import GPOProfile
        return GPOProfile()
    if doc_id == "gep":
        from .gep import GEPProfile
        return GEPProfile()
    if doc_id == "imac":
        from .imac import IMacProfile
        return IMacProfile()
    raise SystemExit(f"Unknown document '{doc_id}'. Choose one of: gpo, gep, imac")
