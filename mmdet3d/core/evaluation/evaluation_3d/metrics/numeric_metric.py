from abc import ABC, abstractmethod

from .base_metric import Basemetric
from .numeric_metric_result import NumericMetricResult


class NumericMetric(Basemetric):
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
    def create_result(result):
        return NumericMetricResult(float(result))