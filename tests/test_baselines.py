"""Tests for the real freMTPL2 underwriting baselines (B1-B4)."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.baselines.baseline_b1_rules import RulesBaseline
from experiments.baselines.baseline_b3_llm_agent import (
    GeoRiskTool,
    SingleLLMAgentBaseline,
)
from experiments.baselines.baseline_b3_llm_agent import run as run_b3
from experiments.baselines.baseline_b4_pipeline import run as run_b4
from experiments.baselines.evaluation import (
    PILOT_DECISIONS,
    agreement_rate,
    decision_summary,
    score_metrics,
    stratified_sample,
)

RAW_FREMTPL2_FREQ = (
    Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'freMTPL2freq.arff'
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
HIGH_RISK = {
    'driver_age': 22,
    'bonus_malus': 150.0,
    'vehicle_age': 15,
    'vehicle_power': 12,
    'vehicle_gas': 'Diesel',
    'density': 20000.0,
    'area': 'F',
}


class RulesBaselineTests(unittest.TestCase):
    def test_low_risk_record_is_accepted_standard(self):
        baseline = RulesBaseline()
        self.assertEqual(baseline.decide(LOW_RISK), 'Accept_Standard')
        self.assertEqual(baseline.score(LOW_RISK), 0.0)

    def test_high_risk_record_is_rejected(self):
        baseline = RulesBaseline()
        self.assertEqual(baseline.decide(HIGH_RISK), 'Reject')
        self.assertGreater(baseline.score(HIGH_RISK), 0.5)

    def test_decisions_are_deterministic_and_in_vocabulary(self):
        baseline = RulesBaseline()
        first = baseline.evaluate([LOW_RISK, HIGH_RISK])
        second = baseline.evaluate([LOW_RISK, HIGH_RISK])
        self.assertEqual(first['decisions'], second['decisions'])
        self.assertIn('young_driver', first['rules_triggered'][1])
        for decision in first['decisions']:
            self.assertIn(decision, ('Accept_Standard', 'Accept_with_Surcharge', 'Reject'))
        for score in first['scores']:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class EvaluationHarnessTests(unittest.TestCase):
    def test_score_metrics_on_perfect_separation(self):
        metrics = score_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(metrics['roc_auc'], 1.0)
        self.assertAlmostEqual(metrics['accuracy_at_0_5'], 1.0)
        self.assertAlmostEqual(metrics['base_rate'], 0.5)

    def test_stratified_sample_keeps_claim_rate_and_size(self):
        indices = np.arange(100)
        targets = np.array([1] * 10 + [0] * 90)
        sample = stratified_sample(indices, targets, 20, seed=0)
        self.assertEqual(len(sample), 20)
        self.assertEqual(int(targets[sample].sum()), 2)
        self.assertEqual(len(set(sample.tolist())), 20)

    def test_decision_summary_counts_and_flags(self):
        summary = decision_summary(
            [1, 0, 1, 0],
            ['Accept_Standard', 'Accept_Standard', 'Accept_with_Surcharge', 'Reject'],
        )
        self.assertEqual(sum(summary['decision_counts'].values()), 4)
        self.assertAlmostEqual(summary['flagged_rate'], 0.5)
        # Two flagged records (indices 2 and 3); only index 2 is actually a claim.
        self.assertAlmostEqual(summary['flag_precision'], 0.5)
        # Two claims (indices 0 and 2); only index 2 was flagged.
        self.assertAlmostEqual(summary['flag_recall'], 0.5)

    def test_decision_summary_rejects_unknown_labels(self):
        with self.assertRaisesRegex(ValueError, 'unknown decision'):
            decision_summary([0], ['Not_A_Decision'])

    def test_agreement_rate(self):
        self.assertEqual(agreement_rate(['A', 'B'], ['A', 'B']), 1.0)
        self.assertEqual(agreement_rate(['A', 'B'], ['B', 'A']), 0.0)


class _BrokenCompletions:
    @staticmethod
    def create(**kwargs):
        raise RuntimeError('provider unreachable')


class _BrokenChat:
    completions = _BrokenCompletions()


class _BrokenClient:
    chat = _BrokenChat()


class SingleLLMBaselineTests(unittest.TestCase):
    def test_fails_closed_without_provider(self):
        cleared = {
            key: value
            for key, value in os.environ.items()
            if key not in {'LLM_PROVIDER', 'GEMINI_API_KEY', 'GROQ_API_KEY'}
        }
        with patch.dict(os.environ, cleared, clear=True):
            agent = SingleLLMAgentBaseline()
            available, reason = agent.available()
            self.assertFalse(available)
            self.assertIn('Aucun fournisseur', reason)

    def test_run_refuses_an_oversized_batch_without_confirmation(self):
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}):
            result = run_b3(sample_size=500, runs=3)
        self.assertEqual(result['status'], 'refused')
        self.assertIn('garde-fou', result['reason'])

    def test_probe_fails_closed_when_the_provider_errors(self):
        agent = SingleLLMAgentBaseline(
            client=_BrokenClient(), provider='groq', model='test-model'
        )
        ready, detail = agent.probe()
        self.assertFalse(ready)
        self.assertIn('Preflight LLM échoué', detail)

    def test_rejects_out_of_vocabulary_decision(self):
        with self.assertRaisesRegex(ValueError, 'Décision LLM invalide'):
            SingleLLMAgentBaseline._validate(
                {'decision': 'Approve', 'claim_risk': 0.2, 'rationale': 'x'},
                'groq',
                'test-model',
            )

    def test_geo_risk_tool_reads_the_bundled_dataset(self):
        tool = GeoRiskTool()
        self.assertTrue(tool.available)
        known = tool.risk_for('F')
        self.assertTrue(known['available'])
        self.assertFalse(tool.risk_for('Z')['available'])


class PipelineBaselineTests(unittest.TestCase):
    @unittest.skipUnless(
        RAW_FREMTPL2_FREQ.is_file(),
        'real freMTPL2 raw data not downloaded (run experiments/data/fetch_fremtpl2.py)',
    )
    def test_pipeline_returns_vocabulary_decisions_on_a_sample(self):
        result = run_b4(mode='sequential', sample_size=4, seed=1)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['sample_size'], 4)
        self.assertEqual(len(result['decisions']), 4)
        for decision in result['decisions']:
            self.assertIn(decision, PILOT_DECISIONS)


if __name__ == '__main__':
    unittest.main()
