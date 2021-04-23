import numpy as np
import torch

from .average_precision import AveragePrecision
from .numeric_metric import NumericMetric


class MeanAveragePrecision(NumericMetric):

    def __init__(self, similarities=[0.5, 0.6, 0.7], reversed_score=False):
        """Creates the MAP metric.

        Args:
            similarities (list, optional) list of similarity thresholds to use in calculations. Defaults to [0.5, 0.6, 0.7].
        """
        assert isinstance(similarities, list) or isinstance(
            similarities, np.ndarray), similarities

        self._similarities = similarities

        # TODO we indicate the absence of a single similarity threshold  by setting it to None
        super().__init__(
            similarity_threshold=None, reversed_score=reversed_score)

    def __str__(self):

        # for equally spaced intervalls we can use a short string representation:
        # [start:stop:step]
        short_repr = True
        diff = None
        if len(self._similarities) > 2:

            for i in range(len(self._similarities)-1):
                curr_t = self._similarities[i]
                next_t = self._similarities[i+1]

                if diff is None:
                    diff = next_t - curr_t
                else:
                    if not np.isclose(diff, next_t-curr_t):
                        # not equally spaced diffs
                        short_repr = False
                        break

        else:
            short_repr = False

        if short_repr:
            name = 'mAP@[{:.2g},{:.2g}:{:.2g}]'.format(self._similarities[0],
                                                       self._similarities[-1], diff)
        else:
            name = 'mAP@{}'.format(self.similarities)

        return name

    @property
    def similarities(self):
        return self._similarities

    def evaluate(self, matching_results, data=None):

        average_precisions = np.zeros((len(self._similarities)))
        ap_metric = AveragePrecision(
            self._similarities[0], reversed_score=self.reversed_score)

        # check wether the matching_results at least one box (pred or gt)
        has_detections = False
        for i, similarity in enumerate(self._similarities):
            ap_metric.similarity_threshold = similarity
            # compute the average precision for this similarity
            average_precision_classes = ap_metric.compute(matching_results)

            ap_sum = 0
            ap_classes = 0
            for _, ap in average_precision_classes.items():
                if ap is not None:
                    # the class was present
                    ap_sum += ap
                    ap_classes += 1

            if ap_classes:
                has_detections = True
                average_precisions[i] = ap_sum / ap_classes

        # if no boxes present return none
        if not has_detections:
            return self.create_result(float('nan'))
        else:
            # get the mean over all similarities
            # and return it as python scalar
            mean_average_precision = average_precisions.mean().item()
            return self.create_result(mean_average_precision)
