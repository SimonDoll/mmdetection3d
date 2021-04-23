import numpy as np

from mmdet3d.core.bbox import LiDARInstance3DBoxes

from .numeric_class_metric import NumericClassMetric


class AverageTranslationErrorPerClass(NumericClassMetric):
    """Calculates the average translation error per class"""

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        """ATE

        Args:
            similarity_threshold (float, optional): [description]. Defaults to 0.5.
            reversed_score (bool, optional): [description]. Defaults to False.
        """
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

    def __str__(self):
        return 'ATE@{}'.format(self.similarity_threshold)

    def compute(self, matching_results, data=None):
        decisions_per_class = self.compute_decisions(
            matching_results,
            self._similarity_threshold,
            return_idxs=True,
            reversed_score=self._reversed_score,
        )

        ate = {class_id: float("inf")
               for class_id in matching_results.keys()}

        for class_id in matching_results.keys():

            class_matchings = matching_results[class_id]
            tp_mask = decisions_per_class[class_id]['tps'].numpy()
            # as the array is 1d -> [0] (single element tuple)
            tp_idxs = np.nonzero(tp_mask)[0]

            if len(tp_idxs) == 0:
                # no tp -> nothing to do
                continue

            summed_te = 0.0
            for idx in tp_idxs:
                match = class_matchings[idx]
                gt_center = match['gt_box'].gravity_center.numpy()
                pred_center = match['pred_box'].gravity_center.numpy()

                translation_error = np.linalg.norm(gt_center - pred_center)
                summed_te += translation_error

            # tp_idxs is > 0
            ate_class = summed_te / len(tp_idxs)

            ate[class_id] = ate_class

        return ate

    def evaluate(self, matching_results, data=None):
        ate_per_class = self.compute(matching_results, data)

        return self.create_result(ate_per_class)
