from abc import ABC, abstractmethod

from .class_metric import ClassMetric
from .plottable_2d_class_metric_result import Plottable2dClassMetricResult
from .plottable_2d_metric import Plottable2dMetric
from .plottable_2d_metric_result import Plottable2dMetricResult


class Plottable2dClassMetric(ClassMetric):

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

    @abstractmethod
    def evaluate(self, matching_results, data=None):
        """
        @see BaseMetric for details.
        Plottable2dClassMetric is used for all metrics that return a dict of plottable2d metric result per class as result.
        Example: returns {class_id_1 : result , ...}

        """
        raise NotImplementedError(
            'this method needs to be implemented from child class')

    @staticmethod
    def result_helper(x_name, y_name, x_dict, y_dict):

        result = {}
        for class_name in x_dict.keys():
            x_vals = x_dict[class_name]
            y_vals = y_dict[class_name]

            result[class_name] = Plottable2dMetricResult(
                x_name, x_vals, y_name, y_vals)
        return Plottable2dClassMetricResult.create_result(result)

    @staticmethod
    def create_result(result):
        return Plottable2dClassMetricResult(result)
