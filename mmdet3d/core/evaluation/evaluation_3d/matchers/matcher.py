from abc import ABC, abstractmethod

import torch


class Matcher:
    """This class serves as interfaces for evaluation.
    A matcher is responsible to match gt to pred boxes using scores from a @see SimilarityMeasure
    """

    def __init__(self, classes):
        self._classes = classes

    @property
    def classes(self):
        return self._classes

    @abstractmethod
    def match(
        self,
        similarity_scores,
        gt_boxes,
        pred_boxes,
        gt_labels,
        pred_labels,
        pred_scores,
        data_id,
    ):
        """
        Matches each gt box to a prediction using the similarity scores.
        A match is only done if the labels match.
        Should return a list: classes x gt x pred_idx.
        Each pred box is matched only once.

        Args:
            similarity_scores ([type]): [description]
            gt_labels ([type]): [description]
            pred_labels ([type]): [description]
        """
        raise NotImplementedError(
            "this method needs to be implemented from child class"
        )
