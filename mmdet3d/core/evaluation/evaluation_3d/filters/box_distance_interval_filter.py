import torch

from mmdet3d.core import LiDARInstance3DBoxes
from .filter import Filter


class BoxDistanceIntervalFilter(Filter):

    def __init__(self, min_radius, max_radius):
        """Filter that collects boxes in a radius around origin. Max radius is
        excluded (interval: [min_radius, max_radius))

        Args:
            box_range (int): radius to use
        """

        assert min_radius >= 0
        self._min_radius = min_radius

        assert max_radius >= min_radius
        self._max_radius = max_radius

    @property
    def max_radius(self):
        return self._max_radius

    @max_radius.setter
    def max_radius(self, max_radius):
        assert max_radius >= self._min_radius
        self._max_radius = max_radius

    @property
    def min_radius(self):
        return self._min_radius

    @min_radius.setter
    def min_radius(self, min_radius):
        assert min_radius >= 0
        self._min_radius = min_radius

    def apply(self, gt_boxes, pred_boxes, gt_labels, pred_labels, pred_scores,
              input_data):

        gt_centers = gt_boxes.gravity_center
        gt_ranges = torch.norm(gt_centers, dim=1)

        gt_boxes_mask = torch.logical_and(gt_ranges >= self._min_radius,
                                          gt_ranges <= self._max_radius)

        pred_centers = pred_boxes.gravity_center
        pred_ranges = torch.norm(pred_centers, dim=1)

        pred_boxes_mask = torch.logical_and(pred_ranges >= self._min_radius,
                                            pred_ranges < self._max_radius)

        (
            gt_boxes,
            pred_boxes,
            gt_labels,
            pred_labels,
            pred_scores,
        ) = self._apply_box_mask(
            gt_boxes_mask,
            pred_boxes_mask,
            gt_boxes,
            pred_boxes,
            gt_labels,
            pred_labels,
            pred_scores,
        )

        return gt_boxes, pred_boxes, gt_labels, pred_labels, pred_scores, input_data
