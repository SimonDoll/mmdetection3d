from .numeric_class_metric import NumericClassMetric
from .precision_at_recall import PrecisionAtRecall


class RecallPerClass(NumericClassMetric):
    """Calculates the recall for all present classes."""

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

        self._precision_at_recall_metric = PrecisionAtRecall(
            self._similarity_threshold, self._reversed_score)

    def __str__(self):
        return 'RecallPerClass'

    def evaluate(self, matching_results, data):
        self._precision_at_recall_metric.similarity_threshold = self._similarity_threshold
        self._precision_at_recall_metric.reversed_score = self._reversed_score

        precision_at_recalls = self._precision_at_recall_metric.compute(
            matching_results, data)

        # simply extract the recall values
        # last value in the list is the recall computed for all examples
        recalls = {}
        for class_id, res in precision_at_recalls.items():
            # if the list is empty no example of this class was present -> nan
            if len(res['recalls']) == 0:
                recalls[class_id] = float('nan')
            else:
                recalls[class_id] = res['recalls'][-1].item()

        return self.create_result(recalls)
