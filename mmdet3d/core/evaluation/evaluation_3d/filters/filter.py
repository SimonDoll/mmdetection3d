import torch
from abc import ABC, abstractmethod


class Filter:
    """This class serves as interfaces for evaluation filters.

    The filter selects a subset of gt_boxes, pred_boxes and corresponding
    labels based on some conditions e.g. min number of points per box
    """

    def __init__(self):
        pass

    @abstractmethod
    def apply(self, gt_boxes, pred_boxes, gt_labels, pred_labels, pred_scores,
              input_data):
        """
        Filters the boxes and returns only the ones matching the filter condition
        Args:
            gt_boxes (BoxType): Ground truth bounding boxes
            pred_boxes (BoxType): Predicted bounding boxes
            gt_labels (tensor): Class labels for ground truth data
            pred_labels (tensor): Class labels for predictions
            pred_scores (tensor): Confidence values of predictions
            input_data (dict): Input to model
        """
        raise NotImplementedError(
            'this method needs to be implemented from child class')

    def _apply_box_mask(
        self,
        gt_mask,
        pred_mask,
        gt_boxes,
        pred_boxes,
        gt_labels,
        pred_labels,
        pred_scores,
    ):
        """Returns only the boxes that are specified in mask.

        Args:
            gt_condition ([type]): [description]
            pred_condition ([type]): [description]
            gt_boxes ([type]): [description]
            pred_boxes ([type]): [description]
            gt_labels ([type]): [description]
            pred_labels ([type]): [description]
            pred_scores ([type]): [description]
        """
        gt_boxes_filtered = gt_boxes[gt_mask]
        gt_labels_filtered = gt_labels[gt_mask]

        pred_boxes_filtered = pred_boxes[pred_mask]
        pred_labels_filtered = pred_labels[pred_mask]
        pred_scores_filtered = pred_scores[pred_mask]

        return (
            gt_boxes_filtered,
            pred_boxes_filtered,
            gt_labels_filtered,
            pred_labels_filtered,
            pred_scores_filtered,
        )
