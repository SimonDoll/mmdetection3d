import torch
import numpy as np

from .numeric_metric import NumericMetric
from .average_precision import AveragePrecision


class MeanAveragePrecision(NumericMetric):
    def __init__(self, similarities={"start": 0.5, "stop": 0.95, "step": 0.05}):
        """Creates the MAP metric.

        Args:
            similarities (dict, optional): Start, Stop and Steps of similarity iteration. Stop is included in calculations. Defaults to {'start': 0.5, 'stop': 0.95, 'step': 0.05}.
        """

        assert similarities["start"] >= 0
        assert similarities["start"] <= similarities["stop"]

        self._similarities = similarities

    def __str__(self):
        return "MeanAveragePrecision"

    @property
    def similarities(self):
        return self._similarities

    def evaluate(self, matching_results, data=None):
        if self._similarities["start"] == self._similarities["stop"]:
            similarity_vals = np.asarray([self._similarities["start"]])
        else:
            similarity_vals = np.arange(
                self._similarities["start"],
                self._similarities["stop"],
                self._similarities["step"],
            )
            # add stop value as well
            similarity_vals = np.concatenate(
                (similarity_vals, [self._similarities["stop"]])
            )

        average_precisions = np.zeros((len(similarity_vals)))
        ap_metric = AveragePrecision(similarity_vals[0])

        # check wether the matching_results at least one box (pred or gt)
        has_detections = False
        for i, similarity in enumerate(similarity_vals):
            ap_metric.similarity_threshold = similarity
            # compute the average precision for this similarity
            average_precision_classes = ap_metric.compute(matching_results)

            ap_sum = 0
            ap_classes = 0
            for class_id, ap in average_precision_classes.items():
                if ap is not None:
                    # the class was present
                    ap_sum += ap
                    ap_classes += 1

            if ap_classes:
                has_detections = True
                average_precisions[i] = ap_sum / ap_classes

        # if no boxes present return none
        if not has_detections:
            return self.create_result(float("-inf"))
        else:
            # get the mean over all similarities
            # and return it as python scalar
            mean_average_precision = average_precisions.mean().item()
            return self.create_result(mean_average_precision)
