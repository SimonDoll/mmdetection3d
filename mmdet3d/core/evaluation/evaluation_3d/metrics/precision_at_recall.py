import numpy as np
import torch

from .plottable_2d_class_metric import Plottable2dClassMetric


class PrecisionAtRecall(Plottable2dClassMetric):

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        """Precision at Recall.

        Args:
            similarity_threshold (float, optional): [description]. Defaults to 0.5.
            reversed_score (bool, optional): [description]. Defaults to False.
        """
        super().__init__(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

    def __str__(self):
        return 'PrecisionAtRecall'

    def compute(self, matching_results, data=None):

        # sort matching_results by detection confidence
        for class_id in matching_results:
            matching_results[class_id].sort(
                key=lambda match: match['pred_score'], reverse=True)

        decisions_per_class = self.compute_decisions(
            matching_results,
            self._similarity_threshold,
            return_idxs=True,
            reversed_score=self._reversed_score,
        )

        precisions_at_recalls = {
            class_id: None
            for class_id in matching_results.keys()
        }
        # calc metric values per class
        for class_id in matching_results.keys():
            tps = decisions_per_class[class_id]['tps'].float()
            fps = decisions_per_class[class_id]['fps'].float()
            gts = decisions_per_class[class_id]['gts'].sum()

            tp_cumsum = torch.cumsum(tps, dim=0)
            fp_cumsum = torch.cumsum(fps, dim=0)

            recalls = torch.div(tp_cumsum, gts + self.EPSILON)
            precisions = torch.div(tp_cumsum,
                                   tp_cumsum + fp_cumsum + self.EPSILON)

            precisions_at_recalls[class_id] = {
                'precisions': precisions,
                'recalls': recalls,
            }
        return precisions_at_recalls

    def evaluate(self, matching_results, data=None):
        precisions_at_recalls = self.compute(matching_results, data)

        precisions = {}
        recalls = {}
        for class_name, res_dict in precisions_at_recalls.items():
            precisions[class_name] = res_dict['precisions']
            recalls[class_name] = res_dict['recalls']

        return self.result_helper('recall', 'precision', recalls, precisions)
