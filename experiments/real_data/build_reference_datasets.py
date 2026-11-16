"""Derive the pilot reference datasets from the real freMTPL2 data (CC0).

The former reference fixtures were internal fiction. This script replaces their
content with values actually computed from the public freMTPL2 datasets so the
External and Claims agents read genuine records:

    data/reference/geographic_risk.json   real claim frequency per freMTPL2 Area (A-F)
    data/reference/claims_history.json     real prior claims of sampled freMTPL2 policies

Both files keep the ``_meta`` provenance block expected by
``connectors.reference_data`` (source, licence, version) so the agent results stay
auditable. Re-run after ``experiments/data/fetch_fremtpl2.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.actuarial_features import AREA_CODES  # noqa: E402

RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
REFERENCE_DIR = PROJECT_ROOT / 'data' / 'reference'
FREQ_FILE = RAW_DIR / 'freMTPL2freq.arff'
SEV_FILE = RAW_DIR / 'freMTPL2sev.arff'
MANIFEST_FILE = RAW_DIR / 'fremtpl2_manifest.json'

# Column order of freMTPL2freq (see the @attribute header of the ARFF file).
FREQ_IDPOL = 0
FREQ_CLAIM_NB = 1
FREQ_AREA = 3
SEV_IDPOL = 0
SEV_CLAIM_AMOUNT = 1

# Bounded sample of real policies kept in the claims-history reference.
MAX_HISTORY_POLICIES = 300

AREA_LABELS_FR = {
    'A': 'Rural / très faible densité',
    'B': 'Semi-rural',
    'C': 'Périurbain',
    'D': 'Urbain',
    'E': 'Urbain dense',
    'F': 'Grande métropole / très dense',
}


def _iter_rows(path: Path):
    """Yield data rows of an ARFF file without loading it into memory."""
    in_data = False
    with path.open(encoding='utf-8-sig', newline='') as source:
        for line in source:
            stripped = line.strip()
            if not stripped or stripped.startswith('%'):
                continue
            if not in_data:
                if stripped.lower() == '@data':
                    in_data = True
                continue
            yield next(csv.reader([line], skipinitialspace=True, quotechar="'"))


def _source_metadata() -> dict:
    manifest = {}
    if MANIFEST_FILE.is_file():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding='utf-8'))
    return {
        'source': 'freMTPL2 (OpenML) — valeurs dérivées des données réelles',
        'license': manifest.get('license', 'CC0 (as stated in OpenML dataset metadata)'),
        'citation': manifest.get(
            'citation',
            'Christophe Dutang and Arthur Charpentier (2018), CASdatasets: Insurance datasets.',
        ),
    }


def _read_frequency():
    """Return per-area aggregates and the IDpol -> area map from the real frequency file."""
    area_policies = defaultdict(int)
    area_claim_policies = defaultdict(int)
    policy_area = {}
    total = 0
    for row in _iter_rows(FREQ_FILE):
        if len(row) != 12:
            continue
        area = row[FREQ_AREA].strip().upper()
        if area not in AREA_CODES:
            continue
        policy_id = row[FREQ_IDPOL].strip()
        try:
            claim_nb = int(float(row[FREQ_CLAIM_NB]))
        except ValueError:
            continue
        area_policies[area] += 1
        if claim_nb >= 1:
            area_claim_policies[area] += 1
        if policy_id:
            policy_area[policy_id] = area
        total += 1
    return area_policies, area_claim_policies, policy_area, total


def _read_severity(policy_area):
    """Group real claim amounts by policy id and accumulate per-area totals."""
    claims_by_policy = defaultdict(list)
    area_claim_amounts = defaultdict(list)
    for row in _iter_rows(SEV_FILE):
        if len(row) != 2:
            continue
        policy_id = row[SEV_IDPOL].strip()
        try:
            amount = float(row[SEV_CLAIM_AMOUNT])
        except ValueError:
            continue
        claims_by_policy[policy_id].append(amount)
        area = policy_area.get(policy_id)
        if area is not None:
            area_claim_amounts[area].append(amount)
    return claims_by_policy, area_claim_amounts


def _build_geographic(area_policies, area_claim_policies, area_claim_amounts, source):
    records = []
    for area in AREA_CODES:
        policies = area_policies.get(area, 0)
        claim_policies = area_claim_policies.get(area, 0)
        amounts = area_claim_amounts.get(area, [])
        records.append(
            {
                'area_code': area,
                'label_fr': AREA_LABELS_FR[area],
                'policies': policies,
                'claim_policies': claim_policies,
                'zone_risk_rate': round(claim_policies / policies, 6) if policies else 0.0,
                'claims_observed': len(amounts),
                'avg_claim_amount': (
                    round(sum(amounts) / len(amounts), 2) if amounts else None
                ),
                'hazards': [],
            }
        )
    return {
        '_meta': {
            'version': '1.0.0-fremtpl2',
            'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'description': (
                'Taux de sinistralité réel par classe géographique freMTPL2 (Area A-F), '
                'dérivé de l’intégralité des polices réelles.'
            ),
            **source,
        },
        'records': records,
    }


def _build_claims_history(claims_by_policy, policy_area, source):
    sampled = [
        policy_id
        for policy_id in sorted(claims_by_policy, key=lambda key: (-len(claims_by_policy[key]), key))
        if policy_area.get(policy_id) is not None
    ][:MAX_HISTORY_POLICIES]

    records = {}
    for policy_id in sampled:
        amounts = claims_by_policy[policy_id]
        records[policy_id] = {
            'policy_id': policy_id,
            'area': policy_area.get(policy_id),
            'prior_claims': [
                {'claim_id': f'{policy_id}-{index + 1}', 'claim_amount': amount}
                for index, amount in enumerate(amounts)
            ],
        }
    return {
        '_meta': {
            'version': '1.0.0-fremtpl2',
            'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'description': (
                'Antécédents sinistres réels (montants freMTPL2sev) pour un échantillon '
                f'borné de {len(records)} polices réelles freMTPL2, indexés par IDpol (ID de police).'
            ),
            **source,
        },
        'records': records,
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f'{path.name}: {len(payload.get("records", {}))} enregistrement(s), sha256 {digest[:12]}...')


def build() -> None:
    if not FREQ_FILE.is_file() or not SEV_FILE.is_file():
        raise FileNotFoundError(
            'Raw freMTPL2 files are missing. Run experiments/data/fetch_fremtpl2.py first.'
        )

    print(f'Reading real data from {FREQ_FILE.name} and {SEV_FILE.name} ...')
    area_policies, area_claim_policies, policy_area, total = _read_frequency()
    claims_by_policy, area_claim_amounts = _read_severity(policy_area)
    print(
        f'{total:,} real policies · {len(claims_by_policy):,} policies with real claims · '
        f'{sum(len(v) for v in claims_by_policy.values()):,} claim amounts'
    )

    source = _source_metadata()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    _write(REFERENCE_DIR / 'geographic_risk.json', _build_geographic(
        area_policies, area_claim_policies, area_claim_amounts, source
    ))
    _write(
        REFERENCE_DIR / 'claims_history.json',
        _build_claims_history(claims_by_policy, policy_area, source),
    )


if __name__ == '__main__':
    build()
