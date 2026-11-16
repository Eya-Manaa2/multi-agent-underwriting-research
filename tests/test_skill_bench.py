"""Tests for the agent-skill-bench framework.

They cover the four pieces the framework is built on: the dependency graph and its
valid-combination enumeration, the shared cost counter, the bootstrap / Pareto
statistics, and the fidelity of the skill wrappers. Everything is checked against
values derived from the code under test -- no expected figure is typed by hand.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from skill_bench.config import default_config_path, load_config
from skill_bench.counter import (
    CostCounter,
    estimate_cost_usd,
    usage_from_gemini,
    usage_from_groq,
    usage_from_response,
)
from skill_bench.evaluate import (
    BASE_LIMITATIONS,
    _derive_limitations,
    _effective_objectives,
    _f1_at_05,
    _plain,
    _warm_up,
    evaluate_combination,
    save_evaluations,
    save_pareto,
)
from skill_bench.graph import SkillGraph
from skill_bench.metrics import (
    _safe_roc_auc,
    bootstrap_ci,
    bootstrap_paired_ci,
    bootstrap_roc_auc_ci,
    dominates,
    pareto_front,
)
from skill_bench.robustness import (
    NUMERIC_FIELDS,
    decision_stability,
    drop_field,
    noisy_application,
    reference_std,
)
from skill_bench.skill import FunctionSkill, SkillMetadata
from skill_bench.underwriting import (
    UNDERWRITING_CORE,
    UNDERWRITING_SKILL_SPECS,
    SkillPipeline,
    build_underwriting_skills,
    spec_skills,
    underwriting_graph,
    valid_underwriting_combinations,
)

RAW_FREMTPL2_FREQ = (
    Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'freMTPL2freq.arff'
)

# A model identifier no provider price map knows: used to check the "unpriced"
# branch without depending on whether the optional LiteLLM package is installed.
UNKNOWN_MODEL = 'agent-skill-bench-unknown-model'


# -- graph -------------------------------------------------------------------
class SkillGraphTests(unittest.TestCase):
    def test_the_search_space_is_the_core_plus_every_optional_subset(self):
        optional = len(UNDERWRITING_SKILL_SPECS) - len(UNDERWRITING_CORE)
        combinations = valid_underwriting_combinations()
        self.assertEqual(len(combinations), 2 ** optional)

    def test_every_combination_contains_the_core_and_is_closed(self):
        graph = underwriting_graph()
        for combination in valid_underwriting_combinations():
            self.assertTrue(set(UNDERWRITING_CORE).issubset(combination))
            ok, reason = graph.validate_with_core(combination, UNDERWRITING_CORE)
            self.assertTrue(ok, reason)

    def test_combinations_are_unique_and_deterministically_ordered(self):
        combinations = valid_underwriting_combinations()
        self.assertEqual(len(combinations), len(set(combinations)))
        self.assertEqual(
            list(combinations),
            sorted(combinations, key=lambda combo: (len(combo), combo)),
        )

    def test_a_missing_dependency_is_rejected(self):
        # `decision` hard-requires `risk`.
        ok, reason = underwriting_graph().validate(('intake', 'decision'))
        self.assertFalse(ok)
        self.assertIn('closure', reason)

    def test_an_unknown_skill_is_rejected(self):
        ok, reason = underwriting_graph().validate(('intake', 'ghost'))
        self.assertFalse(ok)
        self.assertIn('unknown', reason)

    def test_a_combination_without_the_core_is_rejected(self):
        ok, reason = underwriting_graph().validate_with_core(
            ('intake', 'risk'), UNDERWRITING_CORE
        )
        self.assertFalse(ok)
        self.assertIn('core', reason)

    def test_a_cycle_is_rejected(self):
        first = FunctionSkill('a', ('b',), lambda state: {})
        second = FunctionSkill('b', ('a',), lambda state: {})
        with self.assertRaises(ValueError):
            SkillGraph([first, second])

    def test_an_unknown_dependency_is_rejected(self):
        with self.assertRaises(ValueError):
            SkillGraph([FunctionSkill('a', ('ghost',), lambda state: {})])

    def test_duplicate_skill_names_are_rejected(self):
        with self.assertRaises(ValueError):
            SkillGraph(
                [
                    FunctionSkill('a', (), lambda state: {}),
                    FunctionSkill('a', (), lambda state: {}),
                ]
            )

    def test_topological_order_places_dependencies_first(self):
        graph = underwriting_graph()
        order = graph.topological_order
        position = {name: index for index, name in enumerate(order)}
        for name in order:
            for dependency in graph.dependencies_of(name):
                self.assertLess(position[dependency], position[name])


# -- counter -----------------------------------------------------------------
class CostCounterTests(unittest.TestCase):
    def test_a_call_without_model_or_tokens_is_genuinely_free(self):
        self.assertEqual(estimate_cost_usd(None, 0, 0), 0.0)

    def test_tokens_without_a_price_are_reported_as_unknown(self):
        self.assertIsNone(estimate_cost_usd(UNKNOWN_MODEL, 100, 50))

    def test_a_model_without_tokens_is_unknown_not_zero(self):
        # A model was named but nothing else is known: do not invent a figure.
        self.assertIsNone(estimate_cost_usd(UNKNOWN_MODEL, 0, 0))

    def test_the_price_table_prices_the_exact_usage(self):
        table = {'bench-model': {'input': 1.0, 'output': 2.0}}  # USD per 1M tokens
        cost = estimate_cost_usd('bench-model', 1_000_000, 500_000, table)
        self.assertAlmostEqual(cost, 1.0 + 1.0)

    def test_a_call_with_neither_model_nor_tokens_costs_zero(self):
        counter = CostCounter()
        counter.record('deterministic', 1.0)
        self.assertEqual(counter.unknown_cost_calls, 0)
        self.assertEqual(counter.total_cost_usd, 0.0)

    def test_the_counter_aggregates_latency_tokens_and_cost(self):
        table = {'bench-model': {'input': 1.0, 'output': 2.0}}  # USD per 1M tokens
        counter = CostCounter(price_table=table)
        counter.record('a', 1.5, input_tokens=3, output_tokens=4, model='bench-model')
        counter.record('b', 2.5, input_tokens=1, output_tokens=1, model='bench-model')
        self.assertEqual(counter.call_count, 2)
        self.assertAlmostEqual(counter.total_latency_ms, 4.0)
        self.assertEqual(counter.total_input_tokens, 4)
        self.assertEqual(counter.total_output_tokens, 5)
        self.assertEqual(counter.total_tokens, 9)
        self.assertEqual(counter.unknown_cost_calls, 0)
        expected = 4 / 1_000_000.0 * 1.0 + 5 / 1_000_000.0 * 2.0
        self.assertAlmostEqual(counter.total_cost_usd, expected)

    def test_one_unpriced_call_makes_the_total_unknown(self):
        counter = CostCounter()
        counter.record('free', 1.0)
        counter.record('paid', 2.0, input_tokens=10, model=UNKNOWN_MODEL)
        self.assertEqual(counter.unknown_cost_calls, 1)
        self.assertIsNone(counter.total_cost_usd)
        self.assertEqual(counter.summary()['unknown_cost_calls'], 1)

    def test_extend_merges_two_counters(self):
        left = CostCounter()
        left.record('a', 1.0)
        right = CostCounter()
        right.record('b', 2.0, input_tokens=5)
        left.extend(right)
        self.assertEqual(left.call_count, 2)
        self.assertEqual(left.total_input_tokens, 5)

    def test_per_call_yields_plain_rows(self):
        counter = CostCounter()
        counter.record('a', 1.0, ok=False, error='boom')
        rows = list(counter.per_call())
        self.assertEqual(rows[0]['skill'], 'a')
        self.assertFalse(rows[0]['ok'])
        self.assertEqual(rows[0]['error'], 'boom')

    def test_groq_usage_reader(self):
        class _Usage:
            prompt_tokens = 11
            completion_tokens = 7

        class _Response:
            usage = _Usage()

        self.assertEqual(usage_from_groq(_Response()), (11, 7))
        self.assertEqual(usage_from_response('groq', _Response()), (11, 7))
        self.assertEqual(usage_from_groq(object()), (0, 0))

    def test_gemini_usage_reader(self):
        class _Metadata:
            prompt_token_count = 5
            candidates_token_count = 2

        class _Response:
            usage_metadata = _Metadata()

        self.assertEqual(usage_from_gemini(_Response()), (5, 2))
        self.assertEqual(usage_from_response('gemini', _Response()), (5, 2))
        self.assertEqual(usage_from_gemini(object()), (0, 0))

    def test_unknown_provider_reads_as_zero_tokens(self):
        self.assertEqual(usage_from_response('anthropic', object()), (0, 0))


# -- statistics --------------------------------------------------------------
class BootstrapTests(unittest.TestCase):
    def test_the_interval_brackets_the_point_and_is_reproducible(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        first = bootstrap_ci(values, np.mean, n_boot=500, seed=7)
        second = bootstrap_ci(values, np.mean, n_boot=500, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first['point'], 3.0)
        self.assertLessEqual(first['low'], first['point'])
        self.assertLessEqual(first['point'], first['high'])

    def test_a_single_value_gives_a_degenerate_interval(self):
        result = bootstrap_ci([2.0], n_boot=100, seed=1)
        self.assertEqual(result['low'], 2.0)
        self.assertEqual(result['high'], 2.0)

    def test_no_value_gives_no_point(self):
        result = bootstrap_ci([])
        self.assertIsNone(result['point'])
        self.assertEqual(result['n'], 0)

    def test_roc_auc_interval_stays_in_the_unit_range(self):
        y_true = [0] * 20 + [1] * 20
        scores = list(np.linspace(0.1, 0.2, 20)) + list(np.linspace(0.3, 0.4, 20))
        result = bootstrap_roc_auc_ci(y_true, scores, n_boot=300, seed=3)
        self.assertAlmostEqual(result['point'], 1.0)
        self.assertLessEqual(result['low'], result['point'])
        self.assertLessEqual(result['point'], result['high'])

    def test_a_single_class_makes_roc_auc_undefined(self):
        result = bootstrap_roc_auc_ci([0] * 10, list(range(10)), n_boot=50, seed=1)
        self.assertIsNone(result['point'])

    def test_paired_interval_of_f1_brackets_the_point(self):
        y_true = np.asarray([0, 0, 1, 1, 0, 1, 0, 1])
        scores = np.asarray([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.4, 0.6])

        def f1(labels, values):
            from sklearn.metrics import f1_score

            return float(
                f1_score(labels, (values >= 0.5).astype(int), zero_division=0)
            )

        result = bootstrap_paired_ci(y_true, scores, f1, n_boot=200, seed=5)
        self.assertLessEqual(result['low'], result['point'])
        self.assertLessEqual(result['point'], result['high'])


class FastStatisticEquivalenceTests(unittest.TestCase):
    """The fast statistics must equal scikit-learn's -- including with tied scores.

    The real final risk score is an integer on a 0-100 scale, so ties are the normal
    case and the tie handling has to match.
    """

    def test_roc_auc_matches_sklearn_with_tied_scores(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(0)
        compared = 0
        for _ in range(25):
            labels = rng.randint(0, 2, 60)
            if labels.min() == labels.max():
                continue
            scores = rng.randint(0, 11, 60).astype(float)
            self.assertAlmostEqual(
                _safe_roc_auc(labels, scores),
                float(roc_auc_score(labels, scores)),
                places=12,
            )
            compared += 1
        self.assertGreater(compared, 15)

    def test_roc_auc_is_none_for_a_single_class(self):
        self.assertIsNone(_safe_roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))
        self.assertIsNone(_safe_roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))

    def test_f1_matches_sklearn(self):
        from sklearn.metrics import f1_score

        rng = np.random.RandomState(1)
        for _ in range(25):
            labels = rng.randint(0, 2, 60)
            scores = rng.rand(60)
            expected = float(
                f1_score(labels, (scores >= 0.5).astype(int), zero_division=0)
            )
            self.assertAlmostEqual(_f1_at_05(labels, scores), expected, places=12)

    def test_f1_of_an_all_negative_prediction_is_zero(self):
        labels = np.asarray([0, 1, 0, 1])
        scores = np.asarray([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(_f1_at_05(labels, scores), 0.0)


class ParetoTests(unittest.TestCase):
    def test_dominance_requires_one_strict_improvement(self):
        # The first argument is the candidate dominator.
        self.assertTrue(dominates([2.0, 2.0], [1.0, 1.0], ['max', 'max']))
        self.assertFalse(dominates([1.0, 1.0], [1.0, 1.0], ['max', 'max']))
        self.assertFalse(dominates([1.0, 3.0], [2.0, 2.0], ['max', 'max']))
        self.assertTrue(dominates([1.0, 1.0], [2.0, 2.0], ['min', 'min']))

    def test_only_the_non_dominated_records_stay_on_the_front(self):
        records = [
            {'key': 'a', 'm': {'quality': 0.9, 'latency': 10.0}},   # dominates b and c
            {'key': 'b', 'm': {'quality': 0.8, 'latency': 20.0}},
            {'key': 'c', 'm': {'quality': 0.7, 'latency': 30.0}},
        ]
        front = pareto_front(
            records, [('m.quality', 'max'), ('m.latency', 'min')], id_key='key'
        )
        self.assertEqual([row['key'] for row in front['front']], ['a'])
        self.assertEqual(sorted(front['dominated']), ['b', 'c'])
        self.assertEqual(front['front_size'], 1)

    def test_tradeoffs_are_kept_on_the_front(self):
        records = [
            {'key': 'fast', 'm': {'quality': 0.6, 'latency': 5.0}},
            {'key': 'accurate', 'm': {'quality': 0.9, 'latency': 10.0}},
        ]
        front = pareto_front(
            records, [('m.quality', 'max'), ('m.latency', 'min')], id_key='key'
        )
        self.assertEqual({row['key'] for row in front['front']}, {'fast', 'accurate'})

    def test_a_missing_objective_excludes_the_record(self):
        records = [
            {'key': 'a', 'm': {'quality': 0.9, 'latency': 10.0}},
            {'key': 'b', 'm': {'quality': None, 'latency': 5.0}},
        ]
        front = pareto_front(
            records, [('m.quality', 'max'), ('m.latency', 'min')], id_key='key'
        )
        self.assertEqual([row['key'] for row in front['front']], ['a'])
        self.assertEqual(front['excluded'], ['b'])


# -- robustness --------------------------------------------------------------
def _sample_records(size: int = 3) -> list[dict]:
    return [
        {
            'driver_age': 30 + 20 * index,
            'bonus_malus': 60.0 + 30.0 * index,
            'vehicle_age': 5 + 5 * index,
            'vehicle_power': 7 + 3 * index,
            'vehicle_gas': 'Regular',
            'density': 1000.0 * (index + 1),
            'area': 'B',
        }
        for index in range(size)
    ]


class RobustnessTests(unittest.TestCase):
    def test_noise_stays_inside_the_declared_bounds(self):
        from utils.actuarial_features import FIELD_BOUNDS

        records = _sample_records()
        std = reference_std(records)
        rng = np.random.RandomState(0)
        for record in records:
            noisy = noisy_application(record, 0.5, std, rng)
            for field in NUMERIC_FIELDS:
                low, high = FIELD_BOUNDS[field]
                self.assertGreaterEqual(noisy[field], low)
                self.assertLessEqual(noisy[field], high)

    def test_integer_fields_stay_integral(self):
        from utils.actuarial_features import INTEGER_FIELDS

        records = _sample_records()
        std = reference_std(records)
        rng = np.random.RandomState(1)
        noisy = noisy_application(records[0], 0.5, std, rng)
        for field in INTEGER_FIELDS:
            self.assertEqual(noisy[field], float(round(noisy[field])))

    def test_categorical_fields_are_never_perturbed(self):
        records = _sample_records()
        std = reference_std(records)
        rng = np.random.RandomState(2)
        noisy = noisy_application(records[0], 0.5, std, rng)
        self.assertEqual(noisy['vehicle_gas'], 'Regular')
        self.assertEqual(noisy['area'], 'B')

    def test_noise_is_reproducible_for_a_given_seed(self):
        records = _sample_records()
        std = reference_std(records)
        first = noisy_application(records[0], 0.2, std, np.random.RandomState(9))
        second = noisy_application(records[0], 0.2, std, np.random.RandomState(9))
        self.assertEqual(first, second)

    def test_dropping_a_field_never_imputes_it(self):
        degraded = drop_field(_sample_records()[0], 'driver_age')
        self.assertNotIn('driver_age', degraded)
        self.assertIn('bonus_malus', degraded)

    def test_decision_stability_counts_unchanged_decisions(self):
        self.assertEqual(decision_stability(['a', 'b'], ['a', 'b']), 1.0)
        self.assertEqual(decision_stability(['a', 'b'], ['a', 'c']), 0.5)
        self.assertEqual(decision_stability([], []), 1.0)

    def test_decision_stability_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            decision_stability(['a'], ['a', 'b'])


# -- skill interface and wrappers -------------------------------------------
class SkillInterfaceTests(unittest.TestCase):
    def test_a_function_skill_is_callable_and_self_describing(self):
        skill = FunctionSkill(
            'demo',
            ('other',),
            lambda state: {'demo_result': {'ok': True}},
            SkillMetadata(description='a demo', agent_class='DemoAgent'),
        )
        self.assertEqual(skill({}), {'demo_result': {'ok': True}})
        described = skill.describe()
        self.assertEqual(described['name'], 'demo')
        self.assertEqual(described['dependencies'], ['other'])
        self.assertEqual(described['metadata']['agent_class'], 'DemoAgent')
        self.assertEqual(skill.cost_estimate_usd(), 0.0)

    def test_the_specs_cover_the_eleven_agents(self):
        skills = spec_skills()
        self.assertEqual(len(skills), len(UNDERWRITING_SKILL_SPECS))
        self.assertEqual(
            {skill.name for skill in skills},
            {name for name, _, _, _ in UNDERWRITING_SKILL_SPECS},
        )

    def test_dependencies_are_grounded_in_the_specs(self):
        specs = {name: dependencies for name, dependencies, _, _ in UNDERWRITING_SKILL_SPECS}
        self.assertEqual(specs['risk'], ('intake',))
        self.assertEqual(specs['decision'], ('risk',))
        self.assertEqual(specs['fairness'], ('decision',))


class _RecordingOrchestrator:
    """A stand-in orchestrator whose nodes only record that they were called."""

    def __init__(self):
        self.calls: list[str] = []

    def _make(self, name: str):
        def node(state):
            self.calls.append(name)
            return {f'{name}_result': {'ran': True}, 'messages': [name]}

        return node

    def __getattr__(self, attribute: str):
        if attribute.endswith('_node'):
            return self._make(attribute[: -len('_node')])
        raise AttributeError(attribute)


class SkillPipelineTests(unittest.TestCase):
    def test_only_the_selected_skills_run(self):
        orchestrator = _RecordingOrchestrator()
        pipeline = SkillPipeline(skills=build_underwriting_skills(orchestrator))
        _, counter = pipeline.run({'driver_age': 30}, included=UNDERWRITING_CORE)
        expected = [
            name for name in pipeline.order if name in set(UNDERWRITING_CORE)
        ]
        self.assertEqual(orchestrator.calls, expected)
        self.assertEqual(counter.call_count, len(UNDERWRITING_CORE))

    def test_unknown_selected_skills_are_ignored(self):
        orchestrator = _RecordingOrchestrator()
        pipeline = SkillPipeline(skills=build_underwriting_skills(orchestrator))
        pipeline.run({'driver_age': 30}, included=list(UNDERWRITING_CORE) + ['ghost'])
        for call in orchestrator.calls:
            self.assertIn(call, UNDERWRITING_CORE)

    def test_unselected_skills_are_declared_disabled(self):
        orchestrator = _RecordingOrchestrator()
        pipeline = SkillPipeline(skills=build_underwriting_skills(orchestrator))
        state, _ = pipeline.run({'driver_age': 30}, included=UNDERWRITING_CORE)
        disabled = set(state['disabled_agents'])
        self.assertTrue(disabled)
        self.assertTrue(disabled.isdisjoint(set(orchestrator.calls)))

    def test_a_failing_skill_is_recorded_then_propagates(self):
        def boom(state):
            raise RuntimeError('expected failure')

        pipeline = SkillPipeline(
            skills=[
                FunctionSkill('intake', (), lambda state: {'intake_result': {}}),
                FunctionSkill('risk', ('intake',), boom),
            ]
        )
        counter = CostCounter()
        with self.assertRaises(RuntimeError):
            pipeline.run({}, counter=counter)
        self.assertEqual(counter.call_count, 2)
        self.assertTrue(counter.calls[0].ok)
        self.assertFalse(counter.calls[1].ok)
        self.assertIn('expected failure', counter.calls[1].error)

    def test_extracted_document_text_is_not_kept_in_the_final_state(self):
        pipeline = SkillPipeline(
            skills=[
                FunctionSkill(
                    'intake',
                    (),
                    lambda state: {'document_text': 'secret', 'document_bundle': {'x': 1}},
                )
            ]
        )
        state, _ = pipeline.run({})
        self.assertNotIn('document_text', state)
        self.assertNotIn('document_bundle', state)


# -- evaluation aggregation (on synthetic data) ------------------------------
def _decision_skill(name: str, dependencies: tuple[str, ...]):
    """A skill that emits a decision whose score varies with the application."""

    def node(state):
        density = float(state['application'].get('density', 0.0))
        score = 100.0 * (0.2 + 0.6 * ((density % 1000.0) / 1000.0))
        return {
            f'{name}_result': {'ran': True},
            'decision_result': {
                'final_decision': 'Accept_Standard',
                'final_risk_score': score,
            },
        }

    return FunctionSkill(name, dependencies, node)


def _synthetic_holdout(size: int = 40) -> dict:
    rng = np.random.RandomState(0)
    records = _sample_records(1) * size
    for index, record in enumerate(records):
        record['density'] = 100.0 * (index % 10) + index
    return {
        'records': records,
        'targets': rng.randint(0, 2, size),
        'test_indices': np.arange(size),
        'dataset_size': size,
        'test_size': size,
    }


class EvaluationAggregationTests(unittest.TestCase):
    def _pipeline(self) -> SkillPipeline:
        return SkillPipeline(
            skills=[
                FunctionSkill('intake', (), lambda state: {'intake_result': {}}),
                FunctionSkill('risk', ('intake',), lambda state: {'risk_result': {}}),
                _decision_skill('decision', ('risk',)),
            ]
        )

    def _config(self) -> dict:
        config = load_config()
        config['sample_size'] = 12
        config['seeds'] = [1, 2]
        config['robustness']['sample_size'] = 6
        config['bootstrap']['n_boot'] = 100
        return config

    def test_the_entry_is_complete_and_json_serialisable(self):
        entry = evaluate_combination(
            self._pipeline(),
            _synthetic_holdout(),
            list(UNDERWRITING_CORE),
            self._config(),
        )
        self.assertEqual(entry['key'], '+'.join(UNDERWRITING_CORE))
        self.assertEqual(entry['seed_count'], 2)
        self.assertEqual(entry['cases_evaluated'], 24)
        metrics = entry['metrics']
        for section in ('quality', 'latency', 'tokens', 'cost', 'robustness'):
            self.assertIn(section, metrics)
        # The fake skills make no provider call, so cost is a genuine zero.
        self.assertEqual(metrics['cost']['usd_per_case'], 0.0)
        self.assertFalse(metrics['cost']['unknown_calls'])
        self.assertEqual(metrics['tokens']['per_case'], 0.0)
        json.dumps(_plain(entry))

    def test_every_objective_of_the_pareto_configuration_is_present(self):
        config = self._config()
        entry = evaluate_combination(
            self._pipeline(), _synthetic_holdout(), list(UNDERWRITING_CORE), config
        )
        from skill_bench.metrics import _get_path

        for path, _direction in config['pareto']['objectives']:
            self.assertIsNotNone(
                _get_path(entry, path),
                f'objective {path} is missing from the evaluated entry',
            )

    def test_saving_writes_both_artefacts(self):
        config = self._config()
        entry = evaluate_combination(
            self._pipeline(), _synthetic_holdout(), list(UNDERWRITING_CORE), config
        )
        evaluations = {
            'meta': {
                'objectives': [list(item) for item in config['pareto']['objectives']]
            },
            'combination_count': 1,
            'combinations': [entry],
            'limitations': ['test'],
            'pareto': pareto_front(
                [entry],
                [(path, direction) for path, direction in config['pareto']['objectives']],
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            evaluations_path = save_evaluations(evaluations, directory)
            pareto_path = save_pareto(evaluations, directory)
            self.assertTrue(evaluations_path.is_file())
            self.assertTrue(pareto_path.is_file())
            reloaded = json.loads(evaluations_path.read_text(encoding='utf-8'))
            self.assertEqual(reloaded['combination_count'], 1)
            self.assertEqual(reloaded['combinations'][0]['key'], '+'.join(UNDERWRITING_CORE))


# -- derived limitations -----------------------------------------------------
class DerivedLimitationsTests(unittest.TestCase):
    """The data-dependent limits follow the measurements, not a fixed text.

    The sentences must appear and disappear with the data so the artefact can never
    claim something the JSON contradicts.
    """

    OBJECTIVES = (
        ('metrics.quality.roc_auc.point', 'max'),
        ('metrics.latency.per_case_ms.point', 'min'),
    )

    def _entry(self, roc_auc, latency, stability=None, stability_n=None) -> dict:
        entry = {
            'key': f'k{roc_auc}-{latency}',
            'metrics': {
                'quality': {'roc_auc': {'point': roc_auc}},
                'latency': {'per_case_ms': {'point': latency}},
            },
        }
        if stability is not None:
            entry['metrics']['robustness'] = {
                'noise_stability': {'point': stability, 'n': stability_n}
            }
        return entry

    def test_a_constant_axis_is_reported_as_non_discriminating(self):
        evaluated = [self._entry(0.5, 1.0), self._entry(0.5, 2.0)]
        joined = ' '.join(_derive_limitations(evaluated, self.OBJECTIVES))
        self.assertIn('metrics.quality.roc_auc.point', joined)
        self.assertNotIn('metrics.latency.per_case_ms.point', joined)
        self.assertIn('identical for every combination', joined)

    def test_a_varying_axis_drops_its_limitation(self):
        evaluated = [self._entry(0.5, 1.0), self._entry(0.6, 2.0)]
        joined = ' '.join(_derive_limitations(evaluated, self.OBJECTIVES))
        self.assertNotIn('single value', joined)
        self.assertNotIn('identical for every combination', joined)

    def test_stability_spread_is_quantised_from_the_measured_bits(self):
        evaluated = [
            self._entry(0.5, 1.0, stability=0.72, stability_n=100),
            self._entry(0.6, 2.0, stability=0.73, stability_n=100),
        ]
        joined = ' '.join(_derive_limitations(evaluated, self.OBJECTIVES))
        self.assertIn('100 measured noisy decisions', joined)
        self.assertIn('1 decision(s) out of 100', joined)

    def test_the_structural_limits_are_always_present(self):
        self.assertEqual(
            list(_derive_limitations([], self.OBJECTIVES)), list(BASE_LIMITATIONS)
        )


# -- warm-up -----------------------------------------------------------------
class WarmUpTests(unittest.TestCase):
    def test_warm_up_runs_the_expected_number_of_discarded_cases(self):
        calls: list[int] = []

        def node(state):
            calls.append(1)
            return {'intake_result': {}}

        pipeline = SkillPipeline(skills=[FunctionSkill('intake', (), node)])
        config = load_config()
        config['sample_size'] = 7
        config['seeds'] = [1]
        count = _warm_up(pipeline, _synthetic_holdout(), ['intake'], config)
        self.assertEqual(count, 7)
        self.assertEqual(len(calls), 7)

    def test_warm_up_is_capped_by_the_bootstrap_free_sample_size(self):
        calls: list[int] = []

        def node(state):
            calls.append(1)
            return {'intake_result': {}}

        pipeline = SkillPipeline(skills=[FunctionSkill('intake', (), node)])
        config = load_config()
        config['sample_size'] = 500
        config['seeds'] = [1]
        count = _warm_up(pipeline, _synthetic_holdout(120), ['intake'], config)
        self.assertEqual(count, 12)
        self.assertEqual(len(calls), 12)


# -- configuration -----------------------------------------------------------
class EffectiveObjectivesTests(unittest.TestCase):
    def test_every_configured_objective_is_kept_when_robustness_is_on(self):
        config = load_config()
        self.assertTrue(config['robustness']['enabled'])
        self.assertEqual(
            len(_effective_objectives(config)), len(config['pareto']['objectives'])
        )

    def test_the_robustness_axis_is_dropped_when_degradations_are_off(self):
        config = load_config()
        config['robustness']['enabled'] = False
        objectives = _effective_objectives(config)
        self.assertTrue(objectives)
        self.assertFalse(
            any(path.startswith('metrics.robustness') for path, _ in objectives)
        )


class ConfigTests(unittest.TestCase):
    def test_the_defaults_are_returned_as_a_fresh_copy(self):
        first = load_config()
        second = load_config()
        first['seeds'].append(99)
        self.assertNotIn(99, second['seeds'])

    def test_the_shipped_yaml_overrides_are_loaded(self):
        config = load_config(default_config_path())
        self.assertEqual(config['case'], 'underwriting')
        self.assertIn('pareto', config)
        self.assertTrue(config['pareto']['objectives'])

    def test_a_missing_file_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            load_config('this-file-does-not-exist.yaml')


# -- the real agents ---------------------------------------------------------
class RealPipelineTests(unittest.TestCase):
    @unittest.skipUnless(RAW_FREMTPL2_FREQ.is_file(), 'freMTPL2 raw data unavailable')
    def test_the_wrapped_agents_produce_a_real_recommendation(self):
        from experiments.baselines.evaluation import holdout_partition
        from orchestrator.langgraph_orchestrator import LangGraphOrchestrator

        holdout = holdout_partition()
        pipeline = SkillPipeline(LangGraphOrchestrator())
        position = int(holdout['test_indices'][0])
        application = {
            'application_id': 'SB-test',
            'product_line': 'auto',
            **holdout['records'][position],
        }
        state, counter = pipeline.run(application)

        self.assertIn('final_decision', state['decision_result'])
        self.assertIn('final_risk_score', state['decision_result'])
        self.assertTrue(0.0 <= state['decision_result']['final_risk_score'] <= 100.0)
        self.assertEqual(counter.call_count, len(pipeline.order))
        # The eleven agents are deterministic: no provider call, no token, no cost.
        self.assertEqual(counter.total_tokens, 0)
        self.assertEqual(counter.total_cost_usd, 0.0)
        self.assertNotIn('document_text', state)
        self.assertEqual(set(state['disabled_agents']), set())


if __name__ == '__main__':
    unittest.main()
