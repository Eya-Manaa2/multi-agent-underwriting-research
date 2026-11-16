"""AGIRA connector: integration-ready, replay by default.

AGIRA (Association pour la Gestion de l'Information sur le Risque Automobile) is
the French inter-insurers body that operates the "Fichier des Véhicules Assurés"
(FVA) and the shared cancellation / claims-history files. It exposes **no public
API**: access is reserved to member insurers (and authorised authorities) under a
signed convention, with credentials issued by AGIRA.

This module reproduces the *access path* a real insurer feed would use, in three
explicit modes selected with the ``AGIRA_MODE`` environment variable:

* ``replay`` (default): answers from a local fixture at the AGIRA exchange shape
  (``data/reference/agira_fixture.json``). The fixture is FICTIONAL and its
  provenance says so: it must never be presented as real AGIRA data.
* ``live``: performs the real HTTPS exchange against ``AGIRA_API_URL`` using
  ``AGIRA_API_TOKEN``. The exact endpoint, authentication and payload of a real
  convention must be confirmed with AGIRA before switching this mode on.
* ``disabled``: no lookup at all.

Any missing configuration, unknown mode, network error or malformed answer yields
``available=False`` with a machine-readable reason: callers must not invent an
insurance status or a claims history.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from connectors.reference_data import load_reference_dataset

SOURCE_ID = "agira"
DEFAULT_MODE = "replay"
VALID_MODES = ("replay", "live", "disabled")
TIMEOUT_SECONDS = 5

_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def normalize_registration(value: Any) -> str:
    """Uppercase plate without separators, so ``AB-123-CD`` == ``ab 123 cd``."""
    if not isinstance(value, str):
        return ""
    return _NON_ALNUM.sub("", value.upper())


def agira_mode() -> str:
    """Active mode, read at call time from ``AGIRA_MODE`` (default ``replay``)."""
    return (os.environ.get("AGIRA_MODE") or DEFAULT_MODE).strip().lower()


@dataclass
class AgiraLookup:
    """One AGIRA lookup plus its provenance, or the reason it could not be done."""

    registration: str | None = None
    available: bool = False
    mode: str = DEFAULT_MODE
    status: str | None = None  # insured | not_insured | unknown
    insurer: str | None = None
    contract_number: str | None = None
    contract_valid_until: str | None = None
    vehicle: dict[str, Any] = field(default_factory=dict)
    resiliation: dict[str, Any] | None = None
    claims_history: list[Any] = field(default_factory=list)
    real_data: bool = False
    version: str | None = None
    source: str | None = None
    license: str | None = None
    checksum: str | None = None
    reason: str | None = None

    @property
    def prior_claims_count(self) -> int:
        return len(self.claims_history) if self.available else 0

    def descriptor(self) -> dict[str, Any]:
        """Audit-facing summary: what was used, from where, and whether it is real."""
        return {
            "source_id": SOURCE_ID,
            "available": self.available,
            "mode": self.mode,
            "real_data": self.real_data,
            "version": self.version,
            "source": self.source,
            "license": self.license,
            "checksum": self.checksum,
            "reason": self.reason,
        }


def _build(
    record: dict[str, Any],
    registration: str,
    *,
    mode: str,
    real_data: bool,
    version: str | None = None,
    source: str | None = None,
    license: str | None = None,
    checksum: str | None = None,
) -> AgiraLookup:
    if not isinstance(record, dict):
        return AgiraLookup(
            registration=registration,
            mode=mode,
            real_data=real_data,
            reason="Réponse AGIRA invalide (enregistrement attendu).",
        )
    return AgiraLookup(
        registration=record.get("registration") or registration,
        available=True,
        mode=mode,
        status=record.get("status") or "unknown",
        insurer=record.get("insurer"),
        contract_number=record.get("contract_number"),
        contract_valid_until=record.get("contract_valid_until"),
        vehicle=record.get("vehicle") or {},
        resiliation=record.get("resiliation"),
        claims_history=record.get("claims_history") or [],
        real_data=real_data,
        version=version,
        source=source,
        license=license,
        checksum=checksum,
    )


def _replay_lookup(registration: str, normalized: str) -> AgiraLookup:
    dataset = load_reference_dataset(SOURCE_ID)
    if not dataset.available:
        return AgiraLookup(registration=registration, mode="replay", reason=dataset.reason)
    records = dataset.payload.get("records") or []
    record = next(
        (
            item
            for item in records
            if normalize_registration(item.get("registration")) == normalized
        ),
        None,
    )
    provenance = {
        "version": dataset.version,
        "source": dataset.source,
        "license": dataset.license,
        "checksum": dataset.checksum,
    }
    if record is None:
        return AgiraLookup(
            registration=registration,
            available=True,
            mode="replay",
            status="unknown",
            real_data=False,
            reason="Immatriculation absente du référentiel AGIRA de démonstration.",
            **provenance,
        )
    return _build(record, registration, mode="replay", real_data=False, **provenance)


def _live_lookup(registration: str, normalized: str) -> AgiraLookup:
    url = (os.environ.get("AGIRA_API_URL") or "").strip()
    token = (os.environ.get("AGIRA_API_TOKEN") or "").strip()
    if not url or not token:
        return AgiraLookup(
            registration=registration,
            mode="live",
            reason=(
                "Accès AGIRA live non configuré : AGIRA_API_URL et AGIRA_API_TOKEN "
                "sont requis, sous convention AGIRA signée."
            ),
        )
    if not url.lower().startswith("https://"):
        return AgiraLookup(
            registration=registration,
            mode="live",
            reason="Endpoint AGIRA live refusé : HTTPS requis pour des données personnelles.",
        )
    request = urllib.request.Request(  # noqa: S310 - agreed endpoint, HTTPS enforced
        url,
        data=json.dumps({"registration": normalized}).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, UnicodeDecodeError) as error:
        return AgiraLookup(
            registration=registration,
            mode="live",
            reason=f"Échange AGIRA live en échec : {error}",
        )
    record = payload.get("record") if isinstance(payload, dict) else None
    return _build(record, registration, mode="live", real_data=True)


def lookup_vehicle(registration: Any) -> AgiraLookup:
    """Look up the insurance status and claims history of one vehicle.

    Never raises: a missing plate, an unconfigured live access, an unavailable
    fixture or a broken answer all yield ``available=False`` (fail closed).
    """
    mode = agira_mode()
    if mode == "disabled":
        return AgiraLookup(
            registration=registration,
            mode=mode,
            reason="Connecteur AGIRA désactivé (AGIRA_MODE=disabled).",
        )
    if mode not in VALID_MODES:
        return AgiraLookup(
            registration=registration,
            mode=mode,
            reason=f"Mode AGIRA inconnu : {mode}.",
        )
    normalized = normalize_registration(registration)
    if not normalized:
        return AgiraLookup(
            registration=registration,
            mode=mode,
            reason="Immatriculation absente : interrogation AGIRA impossible.",
        )
    if mode == "live":
        return _live_lookup(registration, normalized)
    return _replay_lookup(registration, normalized)
