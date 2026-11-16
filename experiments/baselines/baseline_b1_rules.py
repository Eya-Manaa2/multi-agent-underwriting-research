"""B1 - underwriting rules only (transparent if-then thresholds).

A deterministic, human-readable rule set over the *real* freMTPL2 variables. It is a
research baseline, not an insurer acceptance policy: freMTPL2 ships no accept/reject
labels, so the rule bands below are a documented engineering convention. The rule
points follow the risk directions actually observed in the dataset (young drivers and
high bonus-malus file more claims; dense urban areas and powerful vehicles raise the
risk), and the point bands map onto the pilot recommendation vocabulary.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baselines.evaluation import (  # noqa: E402
    decision_summary,
    holdout_partition,
    save_result,
    score_metrics,
)
from utils.actuarial_features import CLASS_LABELS  # noqa: E402

# Each rule is (name, predicate over the real record, points). Points are deliberately
# small integers so the resulting score stays interpretable.
RULES = (
    ('young_driver', lambda r: r['driver_age'] < 25, 2),
    ('very_young_driver', lambda r: r['driver_age'] < 23, 2),
    ('high_bonus_malus', lambda r: r['bonus_malus'] > 100, 2),
    ('very_high_bonus_malus', lambda r: r['bonus_malus'] > 130, 3),
    ('powerful_vehicle', lambda r: r['vehicle_power'] >= 10, 1),
    ('dense_urban_area', lambda r: r['area'] in ('E', 'F'), 1),
    ('metropolitan_area', lambda r: r['area'] == 'F', 1),
    ('aged_vehicle', lambda r: r['vehicle_age'] >= 10, 1),
)

MAX_POINTS = sum(points for _, _, points in RULES)
SURCHARGE_POINTS = 3
REJECT_POINTS = 6


class RulesBaseline:
    """Deterministic, human-readable underwriting rules over real freMTPL2 variables."""

    def triggered(self, record) -> list[str]:
        """Return the names of the rules triggered by a record."""
        return [name for name, predicate, _ in RULES if predicate(record)]

    def points(self, record) -> int:
        """Raw rule points of a record."""
        return sum(points for name, predicate, points in RULES if predicate(record))

    def score(self, record) -> float:
        """Normalise the rule points to a 0-1 risk score."""
        return min(self.points(record) / MAX_POINTS, 1.0)

    def decide(self, record) -> str:
        """Map the rule points onto the pilot recommendation vocabulary."""
        points = self.points(record)
        if points >= REJECT_POINTS:
            return CLASS_LABELS[2]
        if points >= SURCHARGE_POINTS:
            return CLASS_LABELS[1]
        return CLASS_LABELS[0]

    def evaluate(self, records) -> dict:
        """Return per-record decisions and scores for a list of real records."""
        return {
            'decisions': [self.decide(record) for record in records],
            'scores': [self.score(record) for record in records],
            'rules_triggered': [self.triggered(record) for record in records],
        }


def run() -> dict:
    holdout = holdout_partition()
    index = holdout['test_indices']
    records = [holdout['records'][position] for position in index]
    y_true = holdout['targets'][index]

    baseline = RulesBaseline()
    evaluated = baseline.evaluate(records)
    payload = {
        'baseline': 'B1_rules_only',
        'description': 'transparent if-then underwriting rules, no learning',
        'dataset_size': holdout['dataset_size'],
        'test_size': holdout['test_size'],
        'rules': [
            {'name': name, 'points': points} for name, _, points in RULES
        ],
        'bands': {'surcharge_points': SURCHARGE_POINTS, 'reject_points': REJECT_POINTS},
        **score_metrics(y_true, evaluated['scores']),
        **decision_summary(y_true, evaluated['decisions']),
    }
    return payload


if __name__ == '__main__':
    result = run()
    print(f"Dataset: {result['dataset_size']:,} real policies")
    print(f"Hold-out partition: {result['test_size']:,} policies")
    print(f"Base claim rate: {result['base_rate']:.4f}")
    print(f"ROC-AUC: {result['roc_auc']:.4f}")
    print(f"PR-AUC : {result['pr_auc']:.4f}")
    print(f"Accuracy @0.5: {result['accuracy_at_0_5']:.4f}")
    print(f"Decision counts: {result['decision_counts']}")
    print(
        f"Adverse flag: rate {result['flagged_rate']:.4f} | "
        f"precision {result['flag_precision']:.4f} | recall {result['flag_recall']:.4f}"
    )
    print(f"Saved: {save_result('b1_rules_only', result)}")
