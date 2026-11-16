"""External Agent: geographic reference-risk signal and AGIRA vehicle status.

Two external sources are consulted:

* a real, versioned local reference dataset of per-area claim frequency derived
  from all 678,013 real freMTPL2 policies
  (``experiments/real_data/build_reference_datasets.py``), matched on the
  application's freMTPL2 ``area`` class (A, rural, to F, dense metropolis);
* the AGIRA connector (``connectors/agira.py``), which checks the vehicle's
  insurance status and claims history from the registration (plate) when one is
  available — supplied by the application or read from the uploaded documents.

Both sources attach their provenance (version, licence, checksum) for audit and
fail closed: when a source is unavailable the agent reports the missing source
instead of inventing a factor.
"""

import re

from connectors.agira import lookup_vehicle
from connectors.reference_data import load_reference_dataset
from utils.actuarial_features import SURCHARGE_THRESHOLD

# Above the freMTPL2 base claim rate the zone itself warrants a surcharge adjustment.
ZONE_RISK_SURCHARGE_THRESHOLD = SURCHARGE_THRESHOLD

# Current French registration format (AA-123-BB), used only to read a plate that the
# applicant already provided in an uploaded document.
REGISTRATION_PATTERN = re.compile(r'\b([A-Z]{2})[\s\-.]?([0-9]{3})[\s\-.]?([A-Z]{2})\b')


class ExternalAgent:
    """Retrieves and analyses external/reference information"""

    def __init__(self):
        print("[OK] External Agent initialized")

    def process(self, application, document_bundle=None):
        area = str(application.get('area', '')).strip().upper()
        dataset = load_reference_dataset('geo_risk')

        result = {
            'agent': 'External',
            'external_factors': [],
            'risk_adjustments': [],
            'data_sources': [],
            'source': dataset.descriptor(),
        }

        if not dataset.available:
            result['source_status'] = 'unavailable'
            result['external_factors'].append(
                f"Données de référence indisponibles ({dataset.reason})"
            )
            self._apply_agira(application, document_bundle, result)
            return result

        result['source_status'] = 'ok'
        result['data_sources'].append(f"{dataset.source} (v{dataset.version})")

        records = dataset.payload.get('records') or []
        record = next(
            (item for item in records if item.get('area_code') == area), None
        )
        if record is None:
            result['external_factors'].append(
                f"Zone « {area or 'inconnue'} » non référencée dans le jeu de référence"
            )
            self._apply_agira(application, document_bundle, result)
            return result

        result['data_sources'].append(f"Classe géographique freMTPL2 {record['area_code']}")

        zone_risk_rate = record.get('zone_risk_rate')
        if isinstance(zone_risk_rate, (int, float)) and not isinstance(zone_risk_rate, bool):
            result['external_factors'].append(
                f"Fréquence de sinistralité observée en zone {record['area_code']} : "
                f"{zone_risk_rate * 100:.2f}%"
            )
            if zone_risk_rate > ZONE_RISK_SURCHARGE_THRESHOLD:
                result['external_factors'].append(
                    f"Taux de risque de zone supérieur à la moyenne nationale "
                    f"({ZONE_RISK_SURCHARGE_THRESHOLD * 100:.1f}%)"
                )
                result['risk_adjustments'].append("Ajustement prime — risque de zone")

        avg_claim_amount = record.get('avg_claim_amount')
        if isinstance(avg_claim_amount, (int, float)) and not isinstance(avg_claim_amount, bool):
            result['external_factors'].append(
                f"Coût moyen de sinistre observé en zone : {avg_claim_amount:,.0f} €"
            )

        self._apply_agira(application, document_bundle, result)
        return result

    @staticmethod
    def _registration_from_inputs(application, document_bundle):
        """Plate declared by the applicant, else read from the uploaded documents."""
        for key in ('registration', 'immatriculation'):
            value = application.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if isinstance(document_bundle, dict):
            text = (document_bundle.get('combined_text') or '').upper()
            match = REGISTRATION_PATTERN.search(text)
            if match:
                return ''.join(match.groups())
        return None

    def _apply_agira(self, application, document_bundle, result):
        """Add the AGIRA vehicle status when a plate is available; never invent one."""
        lookup = lookup_vehicle(self._registration_from_inputs(application, document_bundle))
        result['agira'] = lookup.descriptor()
        if not lookup.available:
            return
        if lookup.status == 'unknown':
            result['external_factors'].append(
                'Statut d’assurance AGIRA : véhicule absent du référentiel de démonstration'
            )
            return

        source_label = 'AGIRA (données réelles)' if lookup.real_data else 'AGIRA (fixture de démonstration, non réelle)'
        result['data_sources'].append(source_label)
        if lookup.status == 'insured':
            result['external_factors'].append(
                f"Véhicule assuré au fichier AGIRA — contrat valable jusqu’au "
                f"{lookup.contract_valid_until or 'date inconnue'}"
            )
        elif lookup.status == 'not_insured':
            result['external_factors'].append(
                'Véhicule non assuré au fichier AGIRA — aucune période de validité'
            )
        if lookup.resiliation:
            result['external_factors'].append(
                f"Résiliation enregistrée : {lookup.resiliation.get('reason')} "
                f"({lookup.resiliation.get('date')})"
            )
        if lookup.claims_history:
            result['external_factors'].append(
                f"Antécédents AGIRA sur 5 ans : {lookup.prior_claims_count} sinistre(s)"
            )
