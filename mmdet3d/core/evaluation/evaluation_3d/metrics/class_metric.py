from abc import ABC, abstractmethod

from .base_metric import Basemetric


class ClassMetric(Basemetric):
    """This class serves as interfaces for metrics that return their results on
    a per class basis."""

    # TODO critical:
    # pass the amount of classes to all class metrics in case no example of this class is present?
    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

    def __str__(self):
        return 'ClassMetric(Abstract)'

    @abstractmethod
    def evaluate(self, matching_results, data=None):
        """
        @see BaseMetric for details.
        ClassMetric is used for all metrics that return results per class (structured in a dict with class_ids as keys:
        Example: returns {class_id_1 : some_result , class_id_2 : some_result, ...}

        """
        raise NotImplementedError(
            'this method needs to be implemented from child class')
