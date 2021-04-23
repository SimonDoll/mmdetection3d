import torch

from .similarity_measure import SimilarityMeasure


class CenterDistance2d(SimilarityMeasure):
    """Similarity based on 2d distance of box centers."""

    EPSILON = torch.finfo(torch.float32).eps

    def calc_scores(self, gt_boxes, pred_boxes, gt_labels, pred_labels):
        # create a Gt boxes x Preds tensor (might be [0x0] if no boxes)
        # the score is 1 / center_distance_2d
        # this way the best score is the maximum
        center_distances = torch.zeros(len(pred_boxes), len(gt_boxes))

        # center is bottom center (but doesen't matter as we only use bev (x/y) plane)
        gt_centers_2d = gt_boxes.center[:, :2]
        pred_centers_2d = pred_boxes.center[:, :2]

        # pred x gt x 2
        pred_all_gts = gt_centers_2d.expand(
            (len(pred_centers_2d), gt_centers_2d.shape[0],
             gt_centers_2d.shape[1]))

        pred_centers_2d = torch.unsqueeze(pred_centers_2d, 1)
        center_vectors = pred_centers_2d - pred_all_gts

        center_distances = torch.norm(center_vectors, dim=2)

        return center_distances
