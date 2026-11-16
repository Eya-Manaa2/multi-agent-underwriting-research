"""Tests for the real freMTPL2 pure-premium pricing model and its economic policy.

They cover the pure helpers (technical premium, economic decision, equivalent
claim probability, deviances), the generated artefacts, and the coherence of the
calibrated policy. Expected numbers are read from the artefacts, never typed by
hand, so a re-training cannot silently invalidate the suite.
"""

import json
import unittest
from pathlib import Path

import numpy as np

from experiments.pricing.pure_premium import (
    PRICING_METADATA_FILE,
    POLICY_FILE,
    FREQUENCY_MODEL_FILE,
    SEVERITY_MODEL_FILE,
    _gamma_deviance,
    _poisson_deviance,
    equivalent_claim_probability,
    recommend_from_pure_premium,
    technical_premium,
)

LOW_RISK = {
    'driver_age': 45,
    'bonus_malus': 60.0,
    'vehicle_age': 4,
    'vehicle_power': 6,
    'vehicle_gas': 'Regular',
    'density': 800.0,
    'area': 'B',
}


class TechnicalPremiumTests(unittest.TestCase):
    def test_technical_premium_applies_the_target_loss_ratio(self):
        self.assertAlmostEqual(technical_premium(70.0, 0.7), 100.0)

    def test_technical_premium_rejects_an_invalid_loss_ratio(self):
        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                technical_premium(10.0, bad)


class EconomicDecisionTests(unittest.TestCase):
    def test_below_the_surcharge_threshold_is_accepted_standard(self):
        self.assertEqual(
            recommend_from_pure_premium(50.0, 100.0, 300.0), 'Accept_Standard'
        )

    def test_at_the_surcharge_threshold_a_surcharge_is_applied(self):
        self.assertEqual(
            recommend_from_pure_premium(100.0, 100.0, 300.0),
            'Accept_with_Surcharge',
        )

    def test_at_the_reject_threshold_the_policy_is_rejected(self):
        self.assertEqual(recommend_from_pure_premium(300.0, 100.0, 300.0), 'Reject')

    def test_above_the_reject_threshold_is_rejected(self):
        self.assertEqual(recommend_from_pure_premium(500.0, 100.0, 300.0), 'Reject')

    def test_invalid_threshold_ordering_is_rejected(self):
        with self.assertRaises(ValueError):
            recommend_from_pure_premium(10.0, 300.0, 100.0)
        with self.assertRaises(ValueError):
            recommend_from_pure_premium(10.0, -1.0, 100.0)


class EquivalentProbabilityTests(unittest.TestCase):
    def test_equivalent_probability_is_money_over_mean_severity(self):
        self.assertAlmostEqual(equivalent_claim_probability(230.0, 2300.0), 0.1)

    def test_a_non_positive_severity_is_rejected(self):
        with self.assertRaises(ValueError):
            equivalent_claim_probability(10.0, 0.0)


class DevianceTests(unittest.TestCase):
    def test_a_perfect_poisson_prediction_has_no_deviance(self):
        observed = np.asarray([0.0, 1.0, 3.0])
        self.assertAlmostEqual(_poisson_deviance(observed, observed.copy()), 0.0)

    def test_a_perfect_gamma_prediction_has_no_deviance(self):
        observed = np.asarray([100.0, 250.0, 4000.0])
        self.assertAlmostEqual(_gamma_deviance(observed, observed.copy()), 0.0)


@unittest.skipUnless(
    PRICING_METADATA_FILE.is_file() and POLICY_FILE.is_file(),
    'pricing artefacts missing (run experiments/pricing/pure_premium.py)',
)
class PolicyArtefactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads(PRICING_METADATA_FILE.read_text(encoding='utf-8'))
        cls.policy = json.loads(POLICY_FILE.read_text(encoding='utf-8'))

    def test_the_policy_thresholds_are_calibrated_from_the_baseline(self):
        parameters = self.policy['parameters']
        thresholds = self.policy['calibrated_thresholds']
        baseline = thresholds['baseline_pure_premium_eur']
        self.assertAlmostEqual(
            thresholds['surcharge_threshold_eur'],
            parameters['surcharge_multiple_of_baseline'] * baseline,
            places=6,
        )
        self.assertAlmostEqual(
            thresholds['reject_threshold_eur'],
            parameters['reject_multiple_of_baseline'] * baseline,
            places=6,
        )

    def test_the_policy_thresholds_are_ordered(self):
        thresholds = self.policy['calibrated_thresholds']
        self.assertLessEqual(
            thresholds['surcharge_threshold_eur'], thresholds['reject_threshold_eur']
        )
        self.assertGreater(thresholds['baseline_pure_premium_eur'], 0.0)

    def test_the_equivalent_probabilities_reproduce_the_thresholds(self):
        thresholds = self.policy['calibrated_thresholds']
        mean_severity = self.metadata['metrics']['severity']['mean_claim_amount']
        self.assertAlmostEqual(
            equivalent_claim_probability(
                thresholds['surcharge_threshold_eur'], mean_severity
            ),
            thresholds['equivalent_surcharge_claim_probability'],
            places=9,
        )

    def test_the_decision_counts_cover_the_whole_holdout(self):
        counts = self.metadata['metrics']['decision_counts_on_holdout']
        test_policies = self.metadata['split']['test_policies']
        self.assertEqual(sum(counts.values()), test_policies)
        for label in ('Accept_Standard', 'Accept_with_Surcharge', 'Reject'):
            self.assertIn(label, counts)

    def test_the_frequency_prediction_is_calibrated_in_the_mean(self):
        frequency = self.metadata['metrics']['frequency']
        self.assertAlmostEqual(
            frequency['mean_predicted_count'] / frequency['mean_observed_count'],
            1.0,
            places=1,
        )

    def test_the_policy_carries_the_real_provenance(self):
        provenance = self.policy['provenance']
        self.assertEqual(provenance['frequency_dataset'], 'freMTPL2freq')
        self.assertEqual(provenance['severity_dataset'], 'freMTPL2sev')
        self.assertIn('CC0', provenance['license'])


@unittest.skipUnless(
    FREQUENCY_MODEL_FILE.is_file() and SEVERITY_MODEL_FILE.is_file(),
    'pricing models missing (run experiments/pricing/pure_premium.py)',
)
class PricingModelTests(unittest.TestCase):
    def test_the_frequency_model_scores_a_real_application(self):
        from utils.actuarial_features import FEATURE_NAMES, feature_vector
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(str(FREQUENCY_MODEL_FILE))
        self.assertEqual(model.n_features_in_, len(FEATURE_NAMES))
        vector = np.asarray([feature_vector(LOW_RISK)], dtype=np.float32)
        predicted = model.predict(vector)
        self.assertEqual(predicted.shape, (1,))
        self.assertGreaterEqual(float(predicted[0]), 0.0)

    def test_the_severity_model_returns_positive_amounts(self):
        from utils.actuarial_features import feature_vector
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(str(SEVERITY_MODEL_FILE))
        vector = np.asarray([feature_vector(LOW_RISK)], dtype=np.float32)
        predicted = float(model.predict(vector)[0])
        self.assertGreater(predicted, 0.0)


if __name__ == '__main__':
    unittest.main()
