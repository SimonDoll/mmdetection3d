from abc import ABC, abstractmethod

import torch


class SimilarityMeasure:
    """This class serves as interfaces for evaluation matching classes.
    The matcher is responsible for matching gt to pred boxes and therefore create true positives, false positives etc.
    """

    def __init__(self):
        pass

    @abstractmethod
    def calc_scores(self, gt_boxes, pred_boxes, gt_labels, pred_labels):
        """
        Calc any kind of similarity here. Result should be a tensor of shape preds x gt.
        Scores are defined > 0
        Args:
            gt_boxes ([type]): [description]
            pred_boxes ([type]): [description]
            gt_labels ([type]): [description]
            pred_labels ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError(
            "this method needs to be implemented from child class"
        )
