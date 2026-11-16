"""Batch fairness metrics for acceptance recommendations on held-out data."""

import numpy as np


class FairnessMetrics:
    """Measure selection-rate and true-positive-rate gaps across two groups."""

    @staticmethod
    def _group_values(sensitive_attribute):
        groups = np.unique(sensitive_attribute)
        if len(groups) != 2:
            return None, 'Fairness ratio requires exactly two groups'
        return groups, None

    @staticmethod
    def _ratio(rates):
        available = [rate for rate in rates.values() if rate is not None]
        if not available:
            return None
        maximum = max(available)
        return 1.0 if maximum == 0 else min(available) / maximum

    def demographic_parity(self, y_pred, sensitive_attribute):
        """Return approval-rate ratio and approval rate for each group."""
        y_pred = np.asarray(y_pred)
        sensitive_attribute = np.asarray(sensitive_attribute)
        groups, error = self._group_values(sensitive_attribute)
        if error:
            return None, error
        rates = {
            group.item() if hasattr(group, 'item') else group: float(
                np.mean(y_pred[sensitive_attribute == group])
            )
            for group in groups
        }
        return self._ratio(rates), rates

    def equal_opportunity(self, y_true, y_pred, sensitive_attribute):
        """Return the true-positive-rate ratio among truly eligible cases."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_attribute = np.asarray(sensitive_attribute)
        groups, error = self._group_values(sensitive_attribute)
        if error:
            return None, error
        rates = {}
        for group in groups:
            eligible = (sensitive_attribute == group) & (y_true == 1)
            key = group.item() if hasattr(group, 'item') else group
            rates[key] = (
                float(np.mean(y_pred[eligible]))
                if np.any(eligible)
                else None
            )
        return self._ratio(rates), rates

    def disparate_impact(self, y_pred, sensitive_attribute):
        """Return the ratio of approval rates for the two groups."""
        return self.demographic_parity(y_pred, sensitive_attribute)

    def subgroup_accuracy(self, y_true, y_pred, sensitive_attribute):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_attribute = np.asarray(sensitive_attribute)
        return {
            group.item() if hasattr(group, 'item') else group: float(
                np.mean(y_true[sensitive_attribute == group] == y_pred[sensitive_attribute == group])
            )
            for group in np.unique(sensitive_attribute)
        }

    def subgroup_error_analysis(self, y_true, y_pred, sensitive_attribute):
        return {
            group: 1.0 - accuracy
            for group, accuracy in self.subgroup_accuracy(
                y_true, y_pred, sensitive_attribute
            ).items()
        }


def evaluate_fairness(data, predictions):
    """Descriptive selection-rate parity on the model's recommendations.

    The per-row recommendations are grouped by the real freMTPL2 sensitive proxies
    (``driver_age``, ``bonus_malus``) and their approval rates compared. There is no
    external accept/reject ground truth (freMTPL2 has no underwriting decisions), so
    only selection-rate parity (demographic parity / disparate impact) is reported.
    This is a descriptive audit, not a legal finding.
    """
    if len(data) != len(predictions):
        raise ValueError('Data and predictions must have equal lengths')
    if not data:
        raise ValueError('At least one record is required for fairness evaluation')

    metrics = FairnessMetrics()
    drivers = np.asarray([record['driver_age'] for record in data])
    bonuses = np.asarray([record['bonus_malus'] for record in data])
    age_groups = (drivers >= 35).astype(int)
    bonus_groups = (bonuses <= 100).astype(int)

    y_pred = np.asarray([decision != 'Reject' for decision in predictions], dtype=int)

    age_parity, age_rates = metrics.demographic_parity(y_pred, age_groups)
    bonus_parity, bonus_rates = metrics.demographic_parity(y_pred, bonus_groups)

    return {
        'positive_outcome': 'approval (decision other than Reject)',
        'demographic_parity_age': {'ratio': age_parity, 'rates': age_rates},
        'demographic_parity_bonus_malus': {'ratio': bonus_parity, 'rates': bonus_rates},
        'disparate_impact_age': {'ratio': age_parity, 'rates': age_rates},
        'disparate_impact_bonus_malus': {'ratio': bonus_parity, 'rates': bonus_rates},
    }
