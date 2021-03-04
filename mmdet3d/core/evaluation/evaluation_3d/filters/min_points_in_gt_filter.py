import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes

from .filter import Filter


class MinPointsInGtFilter(Filter):
    def __init__(self, min_points=1, point_cloud_data_key="points"):
        self._min_points = min_points
        self._point_clout_data_key = point_cloud_data_key

    @property
    def min_points(self):
        return self._min_points

    @min_points.setter
    def min_points(self, min_points):
        self._min_points = min_points

    def apply(
        self, gt_boxes, pred_boxes, gt_labels, pred_labels, pred_scores, input_data
    ):
        # move data to gpu (needed for fast bbox operations)

        # get first 3 dimensions of cloud
        cloud = input_data[self._point_clout_data_key][:, :3].cuda()

        # limit to box to the relevant first 7 dimensions
        gt_boxes_cuda = LiDARInstance3DBoxes(gt_boxes.tensor[:, 0:7].cuda())

        gt_box_idxs = gt_boxes_cuda.points_in_boxes(cloud)

        # filter out points that do not lie in a gt box
        gt_box_idxs = gt_box_idxs[gt_box_idxs != -1]

        idxs, counts = torch.unique(gt_box_idxs, return_counts=True)

        valid_idxs_maks = counts >= self._min_points
        idxs = idxs[valid_idxs_maks].long()

        gt_boxes = gt_boxes[idxs]
        gt_labels = gt_labels[idxs]

        return gt_boxes, pred_boxes, gt_labels, pred_labels, pred_scores, input_data
