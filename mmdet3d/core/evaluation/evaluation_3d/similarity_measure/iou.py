import torch

from mmdet3d.core import BboxOverlapsNearest3D
from mmdet3d.core import MaxIoUAssigner

from .similarity_measure import SimilarityMeasure


class Iou(SimilarityMeasure):
    def __init__(self, coordinate="lidar"):
        self._coordinate = coordinate
        self._calc_iou = BboxOverlapsNearest3D(coordinate=self.coordinate)

    @property
    def coordinate(self):
        return self._coordinate

    def calc_scores(self, gt_boxes, pred_boxes, gt_labels, pred_labels):

        # Preds x Gt boxes (might be [0x0] if no boxes)
        # get the tensors of the boxes
        ious = self._calc_iou(pred_boxes.tensor, gt_boxes.tensor)
        return ious
