"""Uncertainty Agent: how far can the model's estimate be trusted?

The risk agent returns a claim probability. This agent turns it into an explicit
predictive-uncertainty signal (the normalized binary entropy of the model's
two-class distribution) and recommends whether the case is confident enough to be
proposed or must be escalated for human review. It never decides the underwriting
outcome: it only qualifies the reliability of the model estimate.

Entropy is normalized to [0, 1] (0 = the model is certain, 1 = maximum doubt at
p = 0.5):

* ``low``      (< 0.30): the estimate clearly leans to one side;
* ``moderate`` (< 0.50): usable but to be read with care;
* ``high``     (>= 0.50): too ambiguous to be proposed without human review.

The agent fails closed: without a usable probability it cannot establish any
reliability, so it recommends a human review instead of inventing certainty.
"""

import math

MODERATE_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.50


def binary_entropy(probability: float) -> float:
    """Normalized entropy of a Bernoulli claim probability, in [0, 1]."""
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise TypeError('probability must be a number')
    if not 0.0 <= probability <= 1.0:
        raise ValueError('probability must lie in [0, 1]')
    if probability in (0.0, 1.0):
        return 0.0
    return -(
        probability * math.log2(probability)
        + (1 - probability) * math.log2(1 - probability)
    )


class UncertaintyAgent:
    """Quantifies the predictive uncertainty of the risk model for one case."""

    def __init__(self):
        self.thresholds = {'moderate': MODERATE_THRESHOLD, 'high': HIGH_THRESHOLD}
        print("[OK] Uncertainty Agent initialized")

    def process(self, application, risk_result):
        """Assess the reliability of the risk model's claim probability."""
        result = {
            'agent': 'Uncertainty',
            'status': 'unavailable',
            'claim_probability': None,
            'predictive_uncertainty': None,
            'level': 'unknown',
            'escalation_recommended': True,  # fail closed: no estimate -> review
            'thresholds': dict(self.thresholds),
            'notes': [],
        }

        probability = (
            risk_result.get('claim_probability')
            if isinstance(risk_result, dict)
            else None
        )
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            result['notes'].append(
                'Probabilité du modèle indisponible : la fiabilité de l’estimation '
                'ne peut pas être établie, revue humaine recommandée.'
            )
            return result
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            result['notes'].append(
                'Probabilité du modèle hors bornes : estimation non fiable.'
            )
            return result

        uncertainty = binary_entropy(probability)
        if uncertainty >= HIGH_THRESHOLD:
            level = 'high'
        elif uncertainty >= MODERATE_THRESHOLD:
            level = 'moderate'
        else:
            level = 'low'

        result.update(
            status='evaluated',
            claim_probability=probability,
            predictive_uncertainty=uncertainty,
            level=level,
            escalation_recommended=level == 'high',
        )
        if level == 'high':
            result['notes'].append(
                f"Estimation trop ambiguë (incertitude {uncertainty:.2f} >= "
                f"{HIGH_THRESHOLD:.2f}) : revue humaine recommandée."
            )
        elif level == 'moderate':
            result['notes'].append(
                f"Estimation à interpréter avec prudence (incertitude {uncertainty:.2f})."
            )
        return result
