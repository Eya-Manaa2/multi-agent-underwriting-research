"""File-backed reference-data connector.

Reads a versioned local reference dataset and exposes its provenance (source,
licence, version, SHA-256 checksum, record count). This is the production-shaped
replacement for the former hardcoded simulation: the data is a real, inspectable
fixture, and the access path is the one a real insurer feed would use.

Behaviour is fail-closed: a missing, unreadable or invalid dataset yields
``available=False`` with a machine-readable reason, and callers must not invent a
signal. Override the directory with ``AURELIA_REFERENCE_DATA_DIR``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

# Logical source id -> fixture file. Agents depend on the source id, not the file.
SOURCE_FILES: dict[str, str] = {
    "geo_risk": "geographic_risk.json",
    "claims_history": "claims_history.json",
    "agira": "agira_fixture.json",
}


def reference_dir() -> Path:
    """Directory holding the reference datasets (env-overridable)."""
    override = os.environ.get("AURELIA_REFERENCE_DATA_DIR")
    return Path(override) if override else DEFAULT_REFERENCE_DIR


@dataclass
class ReferenceDataset:
    """A loaded reference dataset plus its provenance, or the reason it is missing."""

    source_id: str
    available: bool
    payload: dict[str, Any] = field(default_factory=dict)
    version: str | None = None
    source: str | None = None
    license: str | None = None
    checksum: str | None = None
    row_count: int = 0
    reason: str | None = None

    def descriptor(self) -> dict[str, Any]:
        """Audit-facing summary: what was used, from where, and its integrity."""
        return {
            "source_id": self.source_id,
            "available": self.available,
            "version": self.version,
            "source": self.source,
            "license": self.license,
            "checksum": self.checksum,
            "row_count": self.row_count,
            "reason": self.reason,
        }


def _count_records(payload: dict[str, Any]) -> int:
    records = payload.get("records")
    if isinstance(records, (list, dict)):
        return len(records)
    return 0


def load_reference_dataset(source_id: str) -> ReferenceDataset:
    """Load one reference dataset, never raising on missing/broken input."""
    filename = SOURCE_FILES.get(source_id)
    if filename is None:
        return ReferenceDataset(
            source_id=source_id, available=False, reason="Source inconnue."
        )

    path = reference_dir() / filename
    if not path.is_file():
        return ReferenceDataset(
            source_id=source_id,
            available=False,
            reason=f"Jeu de référence absent : {filename}.",
        )

    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        return ReferenceDataset(
            source_id=source_id,
            available=False,
            checksum=checksum,
            reason=f"Jeu de référence illisible : {error}",
        )
    if not isinstance(payload, dict):
        return ReferenceDataset(
            source_id=source_id,
            available=False,
            checksum=checksum,
            reason="Jeu de référence invalide (objet JSON attendu).",
        )

    meta = payload.get("_meta") or {}
    return ReferenceDataset(
        source_id=source_id,
        available=True,
        payload=payload,
        version=meta.get("version"),
        source=meta.get("source"),
        license=meta.get("license"),
        checksum=checksum,
        row_count=_count_records(payload),
    )
