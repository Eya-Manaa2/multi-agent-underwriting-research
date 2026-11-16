"""Top-label calibration and selective-classification metrics."""

import numpy as np


class UncertaintyMetrics:
    """Calculate uncertainty metrics for classifier predictions."""

    def __init__(self, calibration_bins=10):
        if calibration_bins < 1:
            raise ValueError('calibration_bins must be positive')
        self.calibration_bins = calibration_bins

    def brier_score(self, y_true, y_prob, pos_label=None, class_labels=None):
        """Return binary or multiclass Brier score from probabilities."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob, dtype=float)
        if y_prob.ndim == 1:
            positive = 1 if pos_label is None else pos_label
            outcomes = (y_true == positive).astype(float)
            if y_prob.shape[0] != outcomes.shape[0]:
                raise ValueError('Labels and probabilities must have equal lengths')
            return float(np.mean((y_prob - outcomes) ** 2))
        if y_prob.ndim != 2 or y_prob.shape[0] != y_true.shape[0]:
            raise ValueError('Multiclass probabilities must have shape (samples, classes)')
        labels = np.asarray(class_labels) if class_labels is not None else np.unique(y_true)
        if len(labels) != y_prob.shape[1]:
            raise ValueError('Class-label order must match the probability columns')
        label_to_index = {label: index for index, label in enumerate(labels.tolist())}
        try:
            class_indices = np.asarray([label_to_index[label] for label in y_true.tolist()])
        except KeyError as error:
            raise ValueError(f'No probability column for class {error.args[0]}') from error
        one_hot = np.eye(y_prob.shape[1])[class_indices]
        return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))

    def expected_calibration_error(self, correctness, confidences, n_bins=None):
        """Calculate top-label ECE from correctness flags and confidence scores."""
        correctness = np.asarray(correctness, dtype=float)
        confidences = np.asarray(confidences, dtype=float)
        if correctness.shape != confidences.shape or correctness.size == 0:
            raise ValueError('Correctness and confidence arrays must be non-empty and equal length')
        if np.any((confidences < 0) | (confidences > 1)):
            raise ValueError('Confidence scores must be between zero and one')

        bin_count = n_bins or self.calibration_bins
        edges = np.linspace(0.0, 1.0, bin_count + 1)
        ece = 0.0
        for index in range(bin_count):
            in_bin = (confidences >= edges[index]) & (
                confidences <= edges[index + 1]
                if index == bin_count - 1
                else confidences < edges[index + 1]
            )
            if np.any(in_bin):
                ece += float(np.mean(in_bin)) * abs(
                    float(np.mean(correctness[in_bin]))
                    - float(np.mean(confidences[in_bin]))
                )
        return float(ece)

    def risk_coverage_curve(self, y_true, predictions, confidences, n_points=20):
        """Return (coverage, accuracy) points as low-confidence cases abstain."""
        y_true = np.asarray(y_true)
        predictions = np.asarray(predictions)
        confidences = np.asarray(confidences, dtype=float)
        if not (y_true.shape == predictions.shape == confidences.shape):
            raise ValueError('Labels, predictions and confidence scores must have equal lengths')
        if y_true.size == 0:
            raise ValueError('At least one prediction is required')
        order = np.argsort(-confidences)
        sorted_correct = (predictions[order] == y_true[order]).astype(float)
        points = []
        for coverage in np.linspace(1 / len(y_true), 1.0, min(n_points, len(y_true))):
            selected = max(1, int(np.ceil(coverage * len(y_true))))
            points.append((selected / len(y_true), float(np.mean(sorted_correct[:selected]))))
        return points

    def selective_accuracy(self, y_true, predictions, confidences, threshold):
        y_true = np.asarray(y_true)
        predictions = np.asarray(predictions)
        confidences = np.asarray(confidences, dtype=float)
        selected = confidences >= threshold
        if not np.any(selected):
            return 0.0, 0.0
        return float(np.mean(predictions[selected] == y_true[selected])), float(np.mean(selected))

    def escalation_rate(self, confidences, threshold):
        confidences = np.asarray(confidences, dtype=float)
        if confidences.size == 0:
            raise ValueError('At least one confidence score is required')
        return float(np.mean(confidences < threshold))

    def confidence_statistics(self, confidences):
        confidences = np.asarray(confidences, dtype=float)
        if confidences.size == 0:
            raise ValueError('At least one confidence score is required')
        return {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'median': float(np.median(confidences)),
            'percentile_25': float(np.percentile(confidences, 25)),
            'percentile_75': float(np.percentile(confidences, 75)),
        }


def evaluate_uncertainty(
    predictions,
    true_labels,
    confidences,
    probabilities=None,
    class_labels=None,
):
    """Summarize confidence calibration and abstention on a labeled test set."""
    y_pred = np.asarray(predictions)
    y_true = np.asarray(true_labels)
    y_conf = np.asarray(confidences, dtype=float)
    if not (y_pred.shape == y_true.shape == y_conf.shape) or y_true.size == 0:
        raise ValueError('Predictions, labels and confidences must be non-empty and equal length')

    metrics = UncertaintyMetrics()
    correctness = (y_pred == y_true).astype(int)
    result = {
        'accuracy': float(np.mean(correctness)),
        'confidence_statistics': metrics.confidence_statistics(y_conf),
        'top_label_ece': metrics.expected_calibration_error(correctness, y_conf),
        'top_label_brier_score': metrics.brier_score(correctness, y_conf),
        'escalation_rates': {
            f'threshold_{threshold}': metrics.escalation_rate(y_conf, threshold)
            for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]
        },
        'selective_accuracy': {
            f'threshold_{threshold}': {
                'accuracy': metrics.selective_accuracy(
                    y_true, y_pred, y_conf, threshold
                )[0],
                'coverage': metrics.selective_accuracy(
                    y_true, y_pred, y_conf, threshold
                )[1],
            }
            for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]
        },
        'risk_coverage_curve': metrics.risk_coverage_curve(y_true, y_pred, y_conf),
    }
    if probabilities is not None:
        result['multiclass_brier_score'] = metrics.brier_score(
            y_true, probabilities, class_labels=class_labels
        )
    return result
