import numpy as np

from mmdet3d.core.bbox import LiDARInstance3DBoxes

from .numeric_class_metric import NumericClassMetric


class AverageOrientationErrorPerClass(NumericClassMetric):
    """Calculates the average rotation error per class in degrees"""

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        """AOE

        Args:
            similarity_threshold (float, optional): [description]. Defaults to 0.5.
            reversed_score (bool, optional): [description]. Defaults to False.
        """
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

    def __str__(self):
        return 'AOE°@{:.2g}'.format(self.similarity_threshold)

    def compute(self, matching_results, data=None):
        decisions_per_class = self.compute_decisions(
            matching_results,
            self._similarity_threshold,
            return_idxs=True,
            reversed_score=self._reversed_score,
        )

        aoe = {class_id: float("inf")
               for class_id in matching_results.keys()}

        for class_id in matching_results.keys():

            class_matchings = matching_results[class_id]
            tp_mask = decisions_per_class[class_id]['tps'].numpy()
            # as the array is 1d -> [0] (single element tuple)
            tp_idxs = np.nonzero(tp_mask)[0]

            if len(tp_idxs) == 0:
                # no tp -> nothing to do
                continue

            summed_oe = 0.0
            for idx in tp_idxs:
                match = class_matchings[idx]
                gt_yaw = match['gt_box'].yaw.numpy()
                pred_yaw = match['pred_box'].yaw.numpy()

                # angle diff is pos on unit circle
                angle_diff_y = np.sin((gt_yaw - pred_yaw))
                angle_diff_x = np.cos((gt_yaw - pred_yaw))

                rotation_error = np.arctan2(angle_diff_y, angle_diff_x)

                # rotation error is signed (we want absolute)
                rotation_error = np.abs(rotation_error)
                # in degree
                rotation_error = np.rad2deg(rotation_error)
                summed_oe += rotation_error

            # tp_idxs is > 0
            aoe_class = summed_oe / len(tp_idxs)

            aoe[class_id] = aoe_class

        return aoe

    def evaluate(self, matching_results, data=None):
        ate_per_class = self.compute(matching_results, data)

        return self.create_result(ate_per_class)
