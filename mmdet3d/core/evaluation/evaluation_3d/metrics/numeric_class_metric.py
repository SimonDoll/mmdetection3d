from abc import ABC, abstractmethod

from .class_metric import ClassMetric
from .numeric_metric_result import NumericMetricResult
from .numeric_class_metric_result import NumericClassMetricResult


class NumericClassMetric(ClassMetric):
    """This class serves as interfaces for metrics that return a numeric value per class."""

    def __init__(self, num_classes):
        super.__init__(num_classes)

    @abstractmethod
    def evaluate(self, matching_results, data=None):
        """
        @see BaseMetric for details.
        NumericClassMetric is used for all metrics that return a dict of numeric values per class as result.
        Example: returns {class_id_1 : numeric_result , ...}

        """
        raise NotImplementedError(
            "this method needs to be implemented from child class"
        )

    @staticmethod
    def create_result(result):
        for class_name in result.keys():
            if result[class_name] == None:
                result[class_name] = float("nan")
            result[class_name] = NumericMetricResult(float(result[class_name]))

        return NumericClassMetricResult(result)