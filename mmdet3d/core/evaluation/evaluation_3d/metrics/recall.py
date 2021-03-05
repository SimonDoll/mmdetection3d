import math

from .numeric_metric import NumericMetric
from .recall_per_class import RecallPerClass


class Recall(NumericMetric):
    """Calculates the recall (mean over classes)"""

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

        self._recall_per_class_metric = RecallPerClass(
            self._similarity_threshold, self._reversed_score)

    def __str__(self):
        return 'Recall'

    def evaluate(self, matching_results, data):
        self._recall_per_class_metric.similarity_threshold = self._similarity_threshold
        self._recall_per_class_metric.reversed_score = self._reversed_score

        recall_per_class = self._recall_per_class_metric.evaluate(
            matching_results, data)

        # take the mean over the values,
        # if nan skip this class (was not present)

        classes = 0
        recall_sum = 0

        for class_id, recall_res in recall_per_class().items():
            recall = recall_res()
            if math.isnan(recall):
                continue
            else:
                recall_sum += recall
                classes += 1

        if classes == 0:
            # no examples present -> value is invalid
            mean_recall = float('nan')
        else:
            mean_recall = recall_sum / classes
        return self.create_result(mean_recall)
