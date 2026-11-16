"""Tests for the freMTPL2 research experiments: robustness, ablation, collaboration."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.ablation.ablation_study import (
    ABLATABLE_AGENTS,
    MODES,
    STRUCTURAL_AGENTS,
)
from experiments.ablation.ablation_study import run as run_ablation
from experiments.baselines.baseline_b3_llm_agent import (
    SingleLLMAgentBaseline,
    _repair_json,
)
from experiments.collaboration.collaboration_schemes import (
    LLM_REFERENCE,
    LLM_SCHEMES,
    _aggregate,
    _debate_prompt,
)
from experiments.collaboration.collaboration_schemes import run as run_collaboration
from experiments.robustness.robustness_tests import (
    INTEGER_COLUMNS,
    NUMERIC_COLUMNS,
    _masked_cells,
    _noisy,
)
from experiments.robustness.robustness_tests import run as run_robustness

RAW_FREMTPL2_FREQ = (
    Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'freMTPL2freq.arff'
)


def _synthetic_features(rows=40, seed=0) -> np.ndarray:
    """A small well-formed 7-column feature block (numeric + category codes)."""
    rng = np.random.RandomState(seed)
    features = np.zeros((rows, len(NUMERIC_COLUMNS) + 2), dtype=np.float32)
    features[:, 0] = rng.randint(18, 90, rows)      # driver_age (integer)
    features[:, 1] = rng.uniform(50, 200, rows)     # bonus_malus
    features[:, 2] = rng.randint(0, 30, rows)       # vehicle_age (integer)
    features[:, 3] = rng.randint(4, 20, rows)       # vehicle_power (integer)
    features[:, 4] = rng.randint(0, 2, rows)        # vehicle_gas (category)
    features[:, 5] = rng.uniform(0, 50000, rows)    # density
    features[:, 6] = rng.randint(0, 6, rows)        # area (category)
    return features


class RobustnessDegradationTests(unittest.TestCase):
    def test_noise_keeps_integer_fields_integral_and_in_range(self):
        features = _synthetic_features()
        degraded = _noisy(features, 0.20, np.random.RandomState(0))
        for column in INTEGER_COLUMNS:
            self.assertTrue(
                np.allclose(degraded[:, column], np.round(degraded[:, column]))
            )
        for column in NUMERIC_COLUMNS:
            self.assertGreaterEqual(degraded[:, column].min(), features[:, column].min())
            self.assertLessEqual(degraded[:, column].max(), features[:, column].max())

    def test_noise_leaves_categorical_columns_untouched(self):
        features = _synthetic_features()
        degraded = _noisy(features, 0.10, np.random.RandomState(1))
        self.assertTrue(np.array_equal(degraded[:, 4], features[:, 4]))
        self.assertTrue(np.array_equal(degraded[:, 6], features[:, 6]))

    def test_masked_cells_introduces_missing_values_on_numeric_fields(self):
        features = _synthetic_features(rows=200)
        degraded = _masked_cells(features, 0.30, np.random.RandomState(0))
        missing = np.isnan(degraded[:, list(NUMERIC_COLUMNS)])
        share = missing.mean()
        self.assertGreater(share, 0.15)
        self.assertLess(share, 0.45)
        # Category codes are never masked.
        self.assertFalse(np.isnan(degraded[:, 4]).any())
        self.assertFalse(np.isnan(degraded[:, 6]).any())

    @unittest.skipUnless(
        RAW_FREMTPL2_FREQ.is_file(),
        'real freMTPL2 raw data not downloaded (run experiments/data/fetch_fremtpl2.py)',
    )
    def test_run_reports_baseline_and_degraded_metrics(self):
        result = run_robustness(sample_size=500, seed=0)
        self.assertEqual(result['status'], 'ok')
        self.assertIn('roc_auc', result['baseline'])
        self.assertEqual(set(result['noise']), {'0.01', '0.05', '0.10', '0.20'})
        self.assertEqual(set(result['missing_cells']), {'0.10', '0.30'})
        for metrics in result['dropped_fields'].values():
            self.assertIn('roc_auc_delta', metrics)


class AblationStudyTests(unittest.TestCase):
    def test_structural_agents_are_not_ablatable(self):
        for agent in STRUCTURAL_AGENTS:
            self.assertNotIn(agent, ABLATABLE_AGENTS)
        self.assertNotIn('intake', ABLATABLE_AGENTS)
        self.assertNotIn('risk', ABLATABLE_AGENTS)
        self.assertNotIn('decision', ABLATABLE_AGENTS)

    def test_both_decision_modes_are_covered(self):
        self.assertEqual(set(MODES), {'automated', 'human_in_the_loop'})

    @unittest.skipUnless(
        RAW_FREMTPL2_FREQ.is_file(),
        'real freMTPL2 raw data not downloaded (run experiments/data/fetch_fremtpl2.py)',
    )
    def test_run_returns_one_block_per_mode(self):
        result = run_ablation(sample_size=5, seed=3)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(set(result['modes']), set(MODES))
        for mode, block in result['modes'].items():
            self.assertEqual(block['mode'], mode)
            self.assertIn('baseline', block)
            self.assertEqual(set(block['ablations']), set(ABLATABLE_AGENTS))
            for entry in block['ablations'].values():
                self.assertIn('decision_change_rate', entry)


class CollaborationAggregationTests(unittest.TestCase):
    def test_weighted_vote_picks_the_heaviest_ballot(self):
        ballots = [
            {'decision': 'Reject', 'claim_risk': 0.30, 'weight': 0.90},
            {'decision': 'Accept_Standard', 'claim_risk': 0.02, 'weight': 0.20},
        ]
        self.assertEqual(_aggregate(ballots), 'Reject')

    def test_unanimous_vote_is_returned(self):
        ballots = [
            {'decision': 'Accept_with_Surcharge', 'claim_risk': 0.08, 'weight': 1.0},
            {'decision': 'Accept_with_Surcharge', 'claim_risk': 0.09, 'weight': 1.0},
        ]
        self.assertEqual(_aggregate(ballots), 'Accept_with_Surcharge')

    def test_tie_falls_back_to_mean_claim_risk(self):
        ballots = [
            {'decision': 'Accept_Standard', 'claim_risk': 0.02, 'weight': 1.0},
            {'decision': 'Reject', 'claim_risk': 0.80, 'weight': 1.0},
        ]
        # Equal weight: the tie is broken by the mean claim risk (0.41 -> Reject).
        self.assertEqual(_aggregate(ballots), 'Reject')

    def test_debate_prompt_exposes_previous_proposals(self):
        previous = [
            {'agent': 'actuaire', 'decision': 'Reject', 'claim_risk': 0.7,
             'rationale': 'risque élevé'},
        ]
        prompt = _debate_prompt('DOSSIER', 2, previous)
        self.assertIn('Tour 2', prompt)
        self.assertIn('actuaire', prompt)
        self.assertIn('Reject', prompt)

    def test_fails_closed_without_llm_provider(self):
        cleared = {
            key: value
            for key, value in os.environ.items()
            if key not in {'LLM_PROVIDER', 'GEMINI_API_KEY', 'GROQ_API_KEY'}
        }
        if not RAW_FREMTPL2_FREQ.is_file():
            self.skipTest('real freMTPL2 raw data not downloaded')
        with patch.dict(os.environ, cleared, clear=True):
            result = run_collaboration(sample_size=2, seed=0)
        self.assertEqual(result['status'], 'partial')
        # The deterministic schemes always run.
        self.assertIn('pipeline', result['schemes'])
        self.assertIn('orchestrator', result['schemes'])
        for scheme in (LLM_REFERENCE, *LLM_SCHEMES):
            self.assertEqual(result['schemes'][scheme]['status'], 'unavailable')


class LlmJsonParsingTests(unittest.TestCase):
    """The LLM JSON reply is parsed locally (Groq's strict mode is not used)."""

    def test_parses_plain_object(self):
        parsed = SingleLLMAgentBaseline._parse(
            '{"decision":"Reject","claim_risk":0.8,"rationale":"risque"}'
        )
        self.assertEqual(parsed['decision'], 'Reject')

    def test_parses_object_wrapped_in_prose_or_fences(self):
        body = '{"decision":"Accept_Standard","claim_risk":0.02,"rationale":"ok"}'
        for text in (f'```json\n{body}\n```', f'Voici le résultat : {body} fin.'):
            self.assertEqual(
                SingleLLMAgentBaseline._parse(text)['decision'], 'Accept_Standard'
            )

    def test_repairs_a_dropped_closing_quote_of_the_rationale(self):
        # gpt-oss occasionally omits the closing quote of the trailing rationale.
        broken = (
            '{"decision":"Accept_with_Surcharge","claim_risk":0.12,'
            '"rationale":"Le conducteur a 57 ans et un bonus-malus élevé.}'
        )
        parsed = SingleLLMAgentBaseline._parse(broken)
        self.assertEqual(parsed['decision'], 'Accept_with_Surcharge')
        self.assertAlmostEqual(parsed['claim_risk'], 0.12)
        self.assertIn('conducteur', parsed['rationale'])

    def test_rejects_a_reply_that_is_not_an_object(self):
        with self.assertRaises(ValueError):
            SingleLLMAgentBaseline._parse('pas de json ici')

    def test_repair_returns_none_for_a_complete_object(self):
        self.assertIsNone(_repair_json('{"a": 1}'))


if __name__ == '__main__':
    unittest.main()
