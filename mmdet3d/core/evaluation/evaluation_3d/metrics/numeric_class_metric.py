from abc import ABC, abstractmethod

from .class_metric import ClassMetric
from .numeric_class_metric_result import NumericClassMetricResult
from .numeric_metric_result import NumericMetricResult


class NumericClassMetric(ClassMetric):
    """This class serves as interfaces for metrics that return a numeric value
    per class."""

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

    @abstractmethod
    def evaluate(self, matching_results, data=None):
        """
        @see BaseMetric for details.
        NumericClassMetric is used for all metrics that return a dict of numeric values per class as result.
        Example: returns {class_id_1 : numeric_result , ...}

        """
        raise NotImplementedError(
            'this method needs to be implemented from child class')

    @staticmethod
    def create_result(result):
        for class_name in result.keys():
            if result[class_name] == None:
                result[class_name] = float('nan')
            result[class_name] = NumericMetricResult(float(result[class_name]))

        return NumericClassMetricResult(result)
