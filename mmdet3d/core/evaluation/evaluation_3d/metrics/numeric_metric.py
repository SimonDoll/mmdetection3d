from abc import ABC, abstractmethod

from .base_metric import Basemetric
from .numeric_metric_result import NumericMetricResult


class NumericMetric(Basemetric):
    """This class serves as interfaces for metrics that return a numeric
    value."""

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

    def __str__(self):
        return 'NumericMetric (Abstract)'

    @abstractmethod
    def evaluate(self, matching_results, data=None):
        """
        @see BaseMetric for details.
        NumericMetric is used for all metrics that return numeric values as result
        """
        raise NotImplementedError(
            'this method needs to be implemented from child class')

    @staticmethod
    def create_result(result):
        return NumericMetricResult(float(result))
