import torch

from .numeric_class_metric import NumericClassMetric
from .precision_at_recall import PrecisionAtRecall


class AveragePrecision(NumericClassMetric):
    def __init__(self, similarity_threshold=0.5):
        self._precision_at_recall_metric = PrecisionAtRecall(similarity_threshold)

    def __str__(self):
        return "AveragePrecision"

    @property
    def similartiy_threshold(self):
        return self._precision_at_recall_metric.similarity_threshold

    @similartiy_threshold.setter
    def similarity_threshold(self, val):
        self._precision_at_recall_metric.similarity_threshold = val

    def compute(self, matching_results, data=None):
        precisions_at_recalls = self._precision_at_recall_metric.compute(
            matching_results
        )

        average_precisions = {class_id: None for class_id in matching_results.keys()}

        for class_id in matching_results.keys():

            precisions = precisions_at_recalls[class_id]["precisions"]
            recalls = precisions_at_recalls[class_id]["recalls"]

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

        average_precisions = self.compute(matching_results, data)
        return self.create_result(average_precisions)
