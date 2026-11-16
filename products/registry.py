"""Versioned product packs. Auto is live; other lines are reserved skeletons."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from products.auto.rules import AUTO_PRODUCT_PACK


DEFAULT_PRODUCT_LINE = "auto"

# Reserved for later extension — packs exist as stubs only.
_HOME_STUB = {
    "product_line": "home",
    "label_fr": "Habitation",
    "status": "reserved",
    "version": "0.0.0-reserved",
    "required_documents": [
        "attestation_propriete_ou_bail",
        "descriptif_logement",
    ],
    "decision_types": [
        "Accept_Standard",
        "Accept_with_Surcharge",
        "Accept_with_Exclusion",
        "Reject",
        "Escalate_to_Human",
        "Request_More_Info",
    ],
    "notes": "Skeleton only — not enabled for scoring in the current pilot.",
}

_PRODUCT_PACKS: dict[str, dict[str, Any]] = {
    "auto": AUTO_PRODUCT_PACK,
    "home": _HOME_STUB,
}


def list_product_lines(*, enabled_only: bool = False) -> list[dict[str, str]]:
    lines = []
    for code, pack in _PRODUCT_PACKS.items():
        if enabled_only and pack.get("status") != "active":
            continue
        lines.append(
            {
                "code": code,
                "label_fr": pack.get("label_fr", code),
                "status": pack.get("status", "unknown"),
                "version": pack.get("version", ""),
            }
        )
    return lines


def validate_product_line(product_line: str, *, require_active: bool = True) -> str:
    code = (product_line or "").strip().lower()
    if code not in _PRODUCT_PACKS:
        supported = ", ".join(sorted(_PRODUCT_PACKS))
        raise ValueError(f"Unsupported product_line '{product_line}'. Supported: {supported}")
    if require_active and _PRODUCT_PACKS[code].get("status") != "active":
        raise ValueError(
            f"Product line '{code}' is reserved for a later extension and is not active yet"
        )
    return code


def get_product_pack(product_line: str | None = None) -> dict[str, Any]:
    code = validate_product_line(product_line or DEFAULT_PRODUCT_LINE, require_active=False)
    return deepcopy(_PRODUCT_PACKS[code])
