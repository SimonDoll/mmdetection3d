import torch
from abc import ABC, abstractmethod


class Basemetric(ABC):
    """This class serves as interfaces for evaluation.

    A metric is responsible for producing a metric result of any kind,
    subclasses refine this. Beside the evaluate interface metrics may implement
    a compute function that should only be used by other metrics. This allows
    for reusing computation results without the need to use the MetricResult
    interface TODO implement pipeline order
    """

    EPSILON = torch.finfo(torch.float32).eps

    def __init__(self, similarity_threshold=0.5, reversed_score=False):
        """Base Class init with some values that all metrics might need.

        Args:
            similarity_threshold (float, optional): Threshold for decision making. Defaults to 0.5.
            reversed_score (bool, optional): Wether similarity scores are reversed (lower = better). Defaults to False.
        """
        self._similarity_threshold = similarity_threshold
        self._reversed_score = reversed_score

    @property
    def similartiy_threshold(self):
        return self._similarity_threshold

    @similartiy_threshold.setter
    def similarity_threshold(self, val):
        self._similarity_threshold = val

    @property
    def reversed_score(self):
        return self._reversed_score

    @reversed_score.setter
    def reversed_score(self, reversed):
        assert isinstance(reversed, bool)
        self._reversed_score = reversed

    def __str__(self):
        return 'Basemetric(Abstract)'

    @abstractmethod
    def evaluate(self, matching_results, data=None):
        """Evaluates the metric. Metrics are not allowed to change the matching
        results (as they are shared by all metrics) At the moment there is no
        check for this as it would be runtime inefficient Metrics are allowed
        to permute the matching result list per frame e.g.:

        {class_id : [{match 1}, {match 2}]} == {class_id : [{match 2}, {match 1}]} is considered to be the same. This is due to the fact that the order of matches (order of preds and gts does not have any meaning. Sorting per frame is possible with the data_id flag in each matching result.

        Args:
            matching_results (dict): dictionary of classes x list x dicts containing the matched boxes and scores, see @matcher for details.
            data (model_input_data, optional): Data coming from input pipeline. Defaults to None if not used by metric
        """
        raise NotImplementedError(
            'this method needs to be implemented from child class')

    @staticmethod
    @abstractmethod
    def create_result(self, result):
        """creates the correct result type.

        Args:
            result (any): result value of some metric

        Returns:
            [MetricResult]: converted result
        """
        raise NotImplementedError(
            'this method needs to be implemented from child class')

    @staticmethod
    def compute_decisions(matching_results,
                          similarity_treshold,
                          return_idxs=False,
                          reversed_score=False):
        """Computes true positives false positives and gt_amount.

        Args:
            matching_results (dict): matching results for each class
            similarity_treshold (float): similarity treshold for decision
            return_idxs (bool, optional): whether to return indices of decisions instead of counts. Defaults to False.
            reversed_score(bool, optional): whether the similarity score is inverted (lower means better if true)

        Returns:
            dict: per class true positives, false  positives and false negatives. Values are float or list if return_idxs == True
        """
        res = {class_id: None for class_id in matching_results.keys()}

        def extract_scores(match):
            score = match['similarity_score']
            gt = match['gt_box'] != None
            return [score, gt]

        for class_id, matching_results_class in matching_results.items():

            # check if there aren't preds / gt boxes for this class
            if len(matching_results_class) == 0:
                tps = torch.tensor([], dtype=torch.bool)
                fps = torch.tensor([], dtype=torch.bool)
                gts = torch.tensor([], dtype=torch.bool)

                if not return_idxs:
                    tps = tps.sum().item()
                    fps = fps.sum().item()
                    gts = gts.sum().item()

                res[class_id] = {'tps': tps, 'fps': fps, 'gts': gts}
                continue

            # predx x [score, with_gt]
            scores_combined = torch.tensor(
                list(map(extract_scores, matching_results_class)))

            scores = scores_combined.T[0]
            gts = scores_combined.T[1].bool()

            tps = torch.zeros((len(scores), ), dtype=torch.bool)
            fps = torch.zeros((len(scores), ), dtype=torch.bool)

            # no gt box -> false positive
            fps[~gts] = True

            # if gt is true and pred is true
            # non reversed score
            # score >= thresh -> true positive
            # score < thresh -> false positive

            if not reversed_score:
                score_valid = scores >= similarity_treshold
                score_valid[torch.isinf(scores)] = False
            else:
                score_valid = scores <= similarity_treshold
                score_valid[torch.isinf(scores)] = False

            tp_mask = torch.logical_and(gts, score_valid)

            fp_mask = torch.zeros((len(scores), ), dtype=torch.bool)
            fp_mask[torch.logical_and(gts, ~score_valid)] = True

            # TODO not needed any more?
            # remove entries that are ground truth only (no pred for this gt box)
            gt_only_mask = scores == float('-inf')
            fp_mask[gt_only_mask] = False

            tps[tp_mask] = True
            fps[fp_mask] = True

            if not return_idxs:
                tps = tps.sum().item()
                fps = fps.sum().item()
                gts = gts.sum().item()

            res[class_id] = {'tps': tps, 'fps': fps, 'gts': gts}

        return res
