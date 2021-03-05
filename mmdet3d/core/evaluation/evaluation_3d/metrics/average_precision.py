import torch

from .numeric_class_metric import NumericClassMetric
from .precision_at_recall import PrecisionAtRecall


class AveragePrecision(NumericClassMetric):
    """Average Precision Metric (area of precision x recall curve for each
    class)"""

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)
        self._precision_at_recall_metric = PrecisionAtRecall(
            similarity_threshold, reversed_score=reversed_score)

    def __str__(self):
        return 'AveragePrecision'

    def compute(self, matching_results, data=None):
        """Computes the average precision, may be used by other metrics. Should
        not be used as interface (@see BaseMetric for details).

        Args:
            matching_results (dict): Pred / Gt matches
            data (dict, optional): Model inputs (not used)

        Returns:
            dict of average precisions.
        """
        # update the p_at_r metric with the current info
        self._precision_at_recall_metric.similarity_threshold = self.similarity_threshold
        self._precision_at_recall_metric.reversed_score = self.reversed_score
        precisions_at_recalls = self._precision_at_recall_metric.compute(
            matching_results)

        average_precisions = {
            class_id: None
            for class_id in matching_results.keys()
        }

        for class_id in matching_results.keys():

            precisions = precisions_at_recalls[class_id]['precisions']
            recalls = precisions_at_recalls[class_id]['recalls']

            assert len(precisions) == len(recalls)
            if len(precisions) == 0:
                average_precisions[class_id] = None
                continue

            # add start element (needed for trapezoid rule to work)
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # get area under curve
            avg_precision = torch.trapz(precisions, recalls)
            average_precisions[class_id] = avg_precision.item()

        return average_precisions

    def evaluate(self, matching_results, data=None):
        """@see BaseMetric
        """

        average_precisions = self.compute(matching_results, data)
        return self.create_result(average_precisions)
