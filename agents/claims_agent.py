"""Claims Agent: real prior-claims history.

The former implementation mixed fabricated current-case fields (claim amount,
premium, fraud flag) with a fake reference file. It now relies solely on a real
claims-history reference derived from the public freMTPL2 dataset
(``experiments/real_data/build_reference_datasets.py``): real prior-claim amounts per
policy, read through the ``connectors`` layer. When the dataset is unavailable the
agent reports ``unavailable`` and does not fabricate a signal.
"""

from connectors.reference_data import load_reference_dataset

# Prior-claims severity buckets on the aggregate amount (below bound -> label).
SEVERITY_BUCKETS = (
    (5000, 'Low severity'),
    (15000, 'Medium severity'),
    (50000, 'High severity'),
)
SEVERITY_REFERENCE = 30000
MULTI_CLAIM_COUNT = 3
SIMILARITY_FLAG = 0.7


class ClaimsAgent:
    """Analyzes prior claims history and patterns"""

    def __init__(self):
        print("[OK] Claims Agent initialized")

    def process(self, application):
        application_id = application.get('application_id')

        result = {
            'agent': 'Claims',
            'claims_analysis': {},
            'fraud_indicators': [],
            'severity_assessment': 'No claims',
            'similarity_score': 0.0,
        }

        dataset = load_reference_dataset('claims_history')
        result['history_source'] = dataset.descriptor()
        if not dataset.available:
            result['history_source_status'] = 'unavailable'
            result['claims_analysis']['prior_claims_count'] = None
            return result

        result['history_source_status'] = 'ok'
        records = dataset.payload.get('records') or {}
        record = records.get(str(application_id)) if application_id else None
        if record is None:
            result['claims_analysis']['prior_claims_count'] = 0
            result['claims_analysis']['history_note'] = (
                "Aucun antécédent au fichier de référence"
            )
            return result

        prior_claims = record.get('prior_claims') or []
        total_amount = sum(
            float(item.get('claim_amount', 0) or 0) for item in prior_claims
        )
        result['claims_analysis']['prior_claims_count'] = len(prior_claims)
        result['claims_analysis']['prior_claims_total_amount'] = total_amount
        result['severity_assessment'] = self._severity(total_amount)

        similarity = self._similarity(prior_claims, total_amount)
        result['similarity_score'] = similarity

        if len(prior_claims) >= MULTI_CLAIM_COUNT:
            result['fraud_indicators'].append(
                f"Antécédents multiples ({len(prior_claims)}) au fichier de référence"
            )
        if similarity > SIMILARITY_FLAG:
            result['fraud_indicators'].append(
                f"Similarité élevée avec des cas à risque : {similarity:.2f}"
            )

        return result

    def _severity(self, total_amount):
        """Severity label for the aggregate prior-claim amount."""
        if total_amount <= 0:
            return 'No claims'
        for bound, label in SEVERITY_BUCKETS:
            if total_amount < bound:
                return label
        return 'Severe'

    def _similarity(self, prior_claims, total_amount):
        """Risk similarity derived from the number and severity of real prior claims."""
        frequency = min(len(prior_claims), 4) / 4
        severity = min(total_amount, SEVERITY_REFERENCE) / SEVERITY_REFERENCE
        return round(min(0.7 * frequency + 0.3 * severity, 1.0), 2)
