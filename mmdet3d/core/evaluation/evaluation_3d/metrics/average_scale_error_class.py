import logging

import numpy as np
import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes

from .numeric_class_metric import NumericClassMetric


class AverageScaleErrorPerClass(NumericClassMetric):
    """Calculates the average scale error per class (1 - 3DIoU)"""

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        """AŚE

        Args:
            similarity_threshold (float, optional): [description]. Defaults to 0.5.
            reversed_score (bool, optional): [description]. Defaults to False.
        """
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

    def __str__(self):
        return 'ASE@{:.2g}'.format(self.similarity_threshold)

    def compute(self, matching_results, data=None):
        decisions_per_class = self.compute_decisions(
            matching_results,
            self._similarity_threshold,
            return_idxs=True,
            reversed_score=self._reversed_score,
        )

        ase = {class_id: float("inf")
               for class_id in matching_results.keys()}

        for class_id in matching_results.keys():

            class_matchings = matching_results[class_id]
            tp_mask = decisions_per_class[class_id]['tps'].numpy()
            # as the array is 1d -> [0] (single element tuple)
            tp_idxs = np.nonzero(tp_mask)[0]

            if len(tp_idxs) == 0:
                # no tp -> nothing to do
                continue

            summed_se = 0.0
            invalid = 0
            for idx in tp_idxs:
                match = class_matchings[idx]
                gt_vol = match['gt_box'].volume
                pred_vol = match['pred_box'].volume

                # check if a box  len is 0.0
                if torch.any(pred_vol <= 0.0):
                    logging.warning("ASE: ignoring obb with 0 size")
                    invalid += 1
                    continue

                # get 3D IoU
                # a match has only 1 box -> [0]
                assert len(match['gt_box']) == 1
                assert len(match['pred_box']) == 1

                gt_box = match['gt_box'][0]
                pred_box = match['pred_box'][0]

                gt_wlh = gt_box.tensor[0, 3:6]
                pred_wlh = pred_box.tensor[0, 3:6]

                min_wlh = torch.min(gt_wlh, pred_wlh)

                intersection = torch.prod(min_wlh)

                union = gt_vol + pred_vol - intersection

                iou = intersection / union

                assert iou <= 1.0 and iou >= 0.0

                summed_se += 1 - iou

            # tp_idxs is > 0
            if invalid == len(tp_idxs):
                # all boxes invalid
                ase[class_id] = float("inf")
                continue
            ase_class = summed_se / (len(tp_idxs) - invalid)
            print(ase_class)

            ase[class_id] = ase_class

        return ase

    def evaluate(self, matching_results, data=None):
        ate_per_class = self.compute(matching_results, data)

        return self.create_result(ate_per_class)
