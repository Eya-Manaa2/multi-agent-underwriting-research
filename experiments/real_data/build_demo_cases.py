"""Build a small demo set of real freMTPL2 policies for testing the pilot app.

Selects genuine policies from the public freMTPL2 frequency file (CC0) and writes
their real fields plus their real observed outcome, so the application can be
exercised on real data instead of invented profiles.

    data/demo/real_cases.json

Each case keeps the raw values carried by the real policy (driver age, bonus-malus,
vehicle age/power/fuel, density, area) and the real observed outcome (claim count and
exposure). The recommendation the app derives from them remains a documented research
convention, not an insurer decision: freMTPL2 contains no underwriting decisions.

Re-run after ``experiments/data/fetch_fremtpl2.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
REFERENCE_DIR = PROJECT_ROOT / 'data' / 'reference'
DEMO_DIR = PROJECT_ROOT / 'data' / 'demo'
FREQ_FILE = RAW_DIR / 'freMTPL2freq.arff'
SEV_FILE = RAW_DIR / 'freMTPL2sev.arff'
MANIFEST_FILE = RAW_DIR / 'fremtpl2_manifest.json'
CLAIMS_HISTORY_FILE = REFERENCE_DIR / 'claims_history.json'

# Column order of freMTPL2freq (see the @attribute header of the ARFF file).
COL_IDPOL = 0
COL_CLAIM_NB = 1
COL_EXPOSURE = 2
COL_AREA = 3
COL_VEH_POWER = 4
COL_VEH_AGE = 5
COL_DRIV_AGE = 6
COL_BONUS_MALUS = 7
COL_VEH_GAS = 9
COL_DENSITY = 10

# Deterministic demo buckets. The first matching real policies encountered in file
# order are kept, so the output is reproducible. The first bucket deliberately picks
# policies that also carry real prior claims, so the Claims agent demonstrates a real
# history when the case is analysed.
BUCKETS = (
    ('known_claims_history', lambda r: r['policy_id'] in _REFERENCE_IDS, 5),
    ('low_risk_rural', lambda r: r['area'] in {'A', 'B'} and r['claim_nb'] == 0
     and r['bonus_malus'] <= 60 and r['driver_age'] >= 45, 2),
    ('young_driver_claim', lambda r: r['driver_age'] <= 25 and r['claim_nb'] >= 1, 2),
    ('urban_malussed', lambda r: r['area'] in {'E', 'F'} and r['bonus_malus'] >= 150, 2),
    ('urban_no_claim', lambda r: r['area'] in {'E', 'F'} and r['claim_nb'] == 0
     and r['bonus_malus'] >= 100, 2),
    ('senior_multi_claim', lambda r: r['driver_age'] >= 60 and r['claim_nb'] >= 2, 2),
)


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


def _as_policy(row) -> dict | None:
    """Parse one real frequency row into a policy dict, or None if unusable."""
    if len(row) != 12:
        return None
    try:
        exposure = float(row[COL_EXPOSURE])
        if exposure <= 0:
            return None
        return {
            'policy_id': row[COL_IDPOL].strip(),
            'claim_nb': int(float(row[COL_CLAIM_NB])),
            'exposure': exposure,
            'area': row[COL_AREA].strip().upper(),
            'vehicle_power': int(float(row[COL_VEH_POWER])),
            'vehicle_age': int(float(row[COL_VEH_AGE])),
            'driver_age': int(float(row[COL_DRIV_AGE])),
            'bonus_malus': float(row[COL_BONUS_MALUS]),
            'vehicle_gas': row[COL_VEH_GAS].strip(),
            'density': float(row[COL_DENSITY]),
        }
    except (ValueError, IndexError):
        return None


def _select_policies() -> list[dict]:
    """Return one deterministic real policy per wanted slot, in bucket order."""
    chosen: list[dict] = []
    remaining = [wanted for _, _, wanted in BUCKETS]
    used: set[str] = set()
    for row in _iter_rows(FREQ_FILE):
        if not any(remaining):
            break
        policy = _as_policy(row)
        if policy is None or not policy['policy_id'] or policy['policy_id'] in used:
            continue
        for index, (_, predicate, _) in enumerate(BUCKETS):
            if remaining[index] > 0 and predicate(policy):
                chosen.append(policy)
                used.add(policy['policy_id'])
                remaining[index] -= 1
                break
    return chosen


def _real_claim_amounts(policy_ids: set[str]) -> dict[str, list[float]]:
    """Return the real claim amounts (freMTPL2sev) grouped by policy id."""
    amounts: dict[str, list[float]] = {policy_id: [] for policy_id in policy_ids}
    if not SEV_FILE.is_file():
        return amounts
    for row in _iter_rows(SEV_FILE):
        if len(row) != 2:
            continue
        policy_id = row[0].strip()
        if policy_id in amounts:
            try:
                amounts[policy_id].append(float(row[1]))
            except ValueError:
                continue
    return amounts


def _reference_ids() -> set[str]:
    if not CLAIMS_HISTORY_FILE.is_file():
        return set()
    payload = json.loads(CLAIMS_HISTORY_FILE.read_text(encoding='utf-8'))
    return set((payload.get('records') or {}).keys())


_REFERENCE_IDS = _reference_ids()


def _source_metadata() -> dict:
    manifest = {}
    if MANIFEST_FILE.is_file():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding='utf-8'))
    return {
        'source': 'freMTPL2 (OpenML) — polices réelles, champs et issue observée',
        'license': manifest.get('license', 'CC0 (as stated in OpenML dataset metadata)'),
        'citation': manifest.get(
            'citation',
            'Christophe Dutang and Arthur Charpentier (2018), CASdatasets: Insurance datasets.',
        ),
    }


def build() -> None:
    if not FREQ_FILE.is_file():
        raise FileNotFoundError(
            'Raw freMTPL2 file is missing. Run experiments/data/fetch_fremtpl2.py first.'
        )

    print(f'Reading real data from {FREQ_FILE.name} ...')
    policies = _select_policies()
    amounts = _real_claim_amounts({policy['policy_id'] for policy in policies})
    reference_ids = _reference_ids()

    cases = []
    for index, policy in enumerate(policies, start=1):
        policy_id = policy['policy_id']
        real_amounts = amounts.get(policy_id, [])
        cases.append(
            {
                'case_id': f'REAL-{index:02d}',
                'application_id': policy_id,
                'profile': {
                    'driver_age': policy['driver_age'],
                    'bonus_malus': policy['bonus_malus'],
                    'vehicle_age': policy['vehicle_age'],
                    'vehicle_power': policy['vehicle_power'],
                    'vehicle_gas': policy['vehicle_gas'],
                    'density': policy['density'],
                    'area': policy['area'],
                },
                'observed': {
                    'claim_count': policy['claim_nb'],
                    'claim_occurred': policy['claim_nb'] >= 1,
                    'exposure_years': round(policy['exposure'], 4),
                    'prior_claim_amounts_available': len(real_amounts),
                },
                'in_claims_history_reference': policy_id in reference_ids,
            }
        )

    payload = {
        '_meta': {
            'version': '1.0.0-fremtpl2',
            'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'description': (
                'Jeu de cas de démonstration : polices réelles freMTPL2 (champs réels + '
                'issue observée réelle), pour tester l’app sur des données réelles. La '
                'recommandation produite reste une convention de recherche, pas une '
                'décision d’assureur.'
            ),
            'usage': (
                'Recopier les champs de "profile" dans le formulaire de l’app ; mettre '
                '"application_id" pour retrouver, le cas échéant, les antécédents réels.'
            ),
            **_source_metadata(),
        },
        'cases': cases,
    }

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    output = DEMO_DIR / 'real_cases.json'
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f'{output}: {len(cases)} cas réels, sha256 {digest[:12]}...')
    print(
        f"  dont {sum(case['in_claims_history_reference'] for case in cases)} avec "
        'antécédents réels disponibles'
    )


if __name__ == '__main__':
    build()
