from abc import ABC, abstractmethod

from .base_metric import Basemetric
from .plottable_2d_metric_result import Plottable2dMetricResult


class Plottable2dMetric(Basemetric):
    """This class serves as interfaces for metrics that return a numeric value."""

    def __init__(self, num_classes):
        self._num_classes = num_classes

    def __str__(self):
        return "NumericMetric (Abstract)"

    @abstractmethod
    def evaluate(self, matching_results, data=None):
        """
        @see BaseMetric for details.
        NumericMetric is used for all metrics that return numeric values as result
        """
        raise NotImplementedError(
            "this method needs to be implemented from child class"
        )

    @staticmethod
    def result_helper(x_name, xs, y_name, ys):
        return {"x": xs, "y": ys, "x_name": x_name, "y_name": y_name}

    @staticmethod
    def create_result(result):
        return Plottable2dMetricResult(
            result["x"], result["y"], result["x_name"], result["y_name"]
        )
