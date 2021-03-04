from .filter import Filter
from mmdet3d.core import LiDARInstance3DBoxes


class BoxDistanceIntervalFilter(Filter):
    def __init__(self, box_range):
        """Filter that collects boxes in a given rectangle (height / z is ignored)

        Args:
            box_range (list | torch.Tensor): the range of box
                (x_min, y_min, x_max, y_max)
        """

        assert len(box_range) == 4
        self._box_range = box_range

    @property
    def box_range(self):
        return self._box_range

    @box_range.setter
    def box_range(self, box_range):
        assert len(box_range) == 4

    def apply(
        self, gt_boxes, pred_boxes, gt_labels, pred_labels, pred_scores, input_data
    ):

        gt_boxes_mask = gt_boxes.in_range_bev(self._box_range)
        pred_boxes_mask = pred_boxes.in_range_bev(self._box_range)

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
