from .numeric_class_metric import NumericClassMetric
from .precision_at_recall import PrecisionAtRecall


class PrecisionPerClass(NumericClassMetric):
    """Calculates the precision for all present classes."""

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

        self._precision_at_recall_metric = PrecisionAtRecall(
            self._similarity_threshold, self._reversed_score)

    def __str__(self):
        return 'PrecisionPerClass'

    def evaluate(self, matching_results, data):
        self._precision_at_recall_metric.similarity_threshold = self._similarity_threshold
        self._precision_at_recall_metric.reversed_score = self._reversed_score

        precision_at_recalls = self._precision_at_recall_metric.compute(
            matching_results, data)

        # simply extract the precision values
        # last value in the list is the precision computed for all examples
        precisions = {}
        for class_id, res in precision_at_recalls.items():
            # if the list is empty no example of this class was present -> nan
            if len(res['precisions']) == 0:
                precisions[class_id] = float('nan')
            else:
                precisions[class_id] = res['precisions'][-1].item()

        return self.create_result(precisions)
