from abc import ABC, abstractmethod

from .base_metric import Basemetric


class ClassMetric(Basemetric):
    """This class serves as interfaces for metrics that return their results on a per class basis."""

    def __init__(self, num_classes):
        self._num_classes = num_classes

    def __str__(self):
        return "ClassMetric(Abstract)"

    @abstractmethod
    def evaluate(self, matching_results, data=None):
        """
        @see BaseMetric for details.
        ClassMetric is used for all metrics that return results per class (structured in a dict with class_ids as keys:
        Example: returns {class_id_1 : some_result , class_id_2 : some_result, ...}

        """
        raise NotImplementedError(
            "this method needs to be implemented from child class"
        )