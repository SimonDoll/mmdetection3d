import math

from .numeric_metric import NumericMetric
from .precision_per_class import PrecisionPerClass


class Precision(NumericMetric):
    """Calculates the precision (mean over classes)"""

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

        self._precision_per_class_metric = PrecisionPerClass(
            self._similarity_threshold, self._reversed_score)

    def __str__(self):
        return 'Precision'

    def evaluate(self, matching_results, data):
        self._precision_per_class_metric.similarity_threshold = self._similarity_threshold
        self._precision_per_class_metric.reversed_score = self._reversed_score

        precision_per_class = self._precision_per_class_metric.evaluate(
            matching_results, data)

        # take the mean over the values,
        # if nan skip this class (was not present)

        classes = 0
        precision_sum = 0

        for class_id, precision_res in precision_per_class().items():
            precision = precision_res()
            if math.isnan(precision):
                continue
            else:
                precision_sum += precision
                classes += 1

        if classes == 0:
            # no examples present -> value is invalid
            mean_precision = float('nan')
        else:
            mean_precision = precision_sum / classes
        return self.create_result(mean_precision)
