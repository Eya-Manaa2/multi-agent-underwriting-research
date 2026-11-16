"""Risk Agent: real freMTPL2 claim-occurrence model.

Loads the model trained on 678,013 real public French motor policies
(``experiments/real_data/train_fremtpl2_model.py``) and turns a validated
application into a predicted claim probability. The probability is mapped onto the
pilot recommendation vocabulary through the documented business thresholds, so the
encoding can never drift from training (both use ``utils.actuarial_features``).
"""

from pathlib import Path

import numpy as np
import xgboost as xgb

from agents.intake_agent import IntakeAgent
from utils.actuarial_features import (
    CLASS_LABELS,
    FEATURE_NAMES,
    REVIEW_THRESHOLD,
    SURCHARGE_THRESHOLD,
    decision_from_probability,
    feature_vector,
)

MODEL_FILE = (
    Path(__file__).resolve().parents[1] / 'models' / 'fremtpl2_underwriting_model.json'
)
NEGATIVE_LABEL = 'no_claim'
POSITIVE_LABEL = 'claim_occurred'


class RiskAgent:
    """Evaluates risk using the trained ML model"""

    def __init__(self):
        self.model = None
        self.feature_names = FEATURE_NAMES
        self._load_model()

    def _load_model(self):
        """Load the trained XGBoost claim model."""
        if not MODEL_FILE.is_file():
            raise FileNotFoundError(f'Trained model not found: {MODEL_FILE}')

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(MODEL_FILE))
        if self.model.n_features_in_ != len(FEATURE_NAMES):
            raise ValueError(
                f'Model expects {self.model.n_features_in_} features; '
                f'the application pipeline provides {len(FEATURE_NAMES)}'
            )

    @staticmethod
    def _risk_score(probability: float) -> float:
        """Map the real claim probability onto a 0-100 index aligned with the bands."""
        if probability >= REVIEW_THRESHOLD:
            return 70 + (probability - REVIEW_THRESHOLD) / (1 - REVIEW_THRESHOLD) * 30
        if probability >= SURCHARGE_THRESHOLD:
            return (
                40
                + (probability - SURCHARGE_THRESHOLD)
                / (REVIEW_THRESHOLD - SURCHARGE_THRESHOLD)
                * 30
            )
        return probability / SURCHARGE_THRESHOLD * 40

    def process(self, application, intake_result, customer_result):
        """Evaluate risk using the real claim-probability model."""
        validation = IntakeAgent().process(application)
        if validation['status'] != 'validated':
            raise ValueError(
                'Risk evaluation requires a valid application: '
                + '; '.join(validation['errors'])
            )

        features = np.asarray([feature_vector(application)], dtype=np.float32)
        probability = float(self.model.predict_proba(features)[0][1])
        decision = decision_from_probability(probability)
        confidence = max(probability, 1 - probability)

        return {
            'agent': 'Risk',
            'decision': decision,
            'risk_score': float(self._risk_score(probability)),
            'confidence': confidence,
            'claim_probability': probability,
            'ml_probabilities': {
                NEGATIVE_LABEL: 1 - probability,
                POSITIVE_LABEL: probability,
            },
        }
