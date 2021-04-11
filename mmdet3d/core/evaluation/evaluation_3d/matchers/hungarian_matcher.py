import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


from .matcher import Matcher


class HungarianMatcher(Matcher):
    """Hungarian matching computes the optimal assignment to minimize the total assignment costs"""

    def __init__(self, classes):
        super().__init__(classes)

    def match(
        self,
        similarity_scores,
        gt_boxes,
        pred_boxes,
        gt_labels,
        pred_labels,
        pred_scores,
        data_id,
        reversed_score=False,
    ):
        """Matches the gt boxes and the predictions.

        Args:
            similarity_scores (torch.tensor): pred x gt similarities between all boxes
            gt_labels (torch.tensor): gt box labels
            pred_labels (toch.tensor): pred box labels
        """

        result = {c: [] for c in self.classes}

        for c in self.classes:

            pred_class_mask = pred_labels == c

            # check if predictions for this class
            preds_available = torch.any(pred_class_mask)

            # get pred_idxs of class
            pred_idxs_class = torch.nonzero(pred_class_mask, as_tuple=True)[0]

            # no need to filter the pred_labels / gt_labels as we know the class in the loop already
            pred_boxes_filtered = pred_boxes[pred_class_mask]
            pred_scores_filtered = pred_scores[pred_idxs_class]

            # the same for ground truth
            gt_class_mask = gt_labels == c
            gt_available = torch.any(gt_class_mask)

            # get gt_idxs of class
            gt_idxs_class = torch.nonzero(gt_class_mask, as_tuple=True)[0]

            gt_boxes_filtered = gt_boxes[gt_class_mask]

            # make a copy of the scores as we modify them
            # remove preds of wrong classes
            # shape preds of class x gt_boxes
            similarity_filtered = similarity_scores[pred_idxs_class].detach(
            ).clone()

            # remove gts of wrong classes
            # shape: gt_boxes class x pred boxes class
            similarity_filtered = similarity_filtered.T[gt_idxs_class]

            # find the matches
            cost_matrix = similarity_filtered.detach().numpy()
            maximize_cost = False if reversed_score else True

            gt_match_idxs, pred_match_idxs = linear_sum_assignment(
                cost_matrix, maximize=maximize_cost)

            # get the indices of the non matched predictions
            non_matched_pred_idxs = np.ones(
                similarity_filtered.size(1), dtype=bool)
            non_matched_pred_idxs[pred_match_idxs] = False

            # idx array is 1dmin -> [0]
            non_matched_pred_idxs = np.where(non_matched_pred_idxs)[0]

            non_matched_pred_idxs = non_matched_pred_idxs.astype(int)

            # get the indices of the non matched gt boxes
            non_matched_gt_idxs = np.ones(
                similarity_filtered.size(0), dtype=bool)
            non_matched_gt_idxs[gt_match_idxs] = False

            # idx array is 1dmin -> [0]
            non_matched_gt_idxs = np.where(non_matched_gt_idxs)[0]
            non_matched_gt_idxs = non_matched_gt_idxs.astype(int)

            # add the matches to the results
            # scipty returns np.int64 arrays but mmdet boxes only check for int idxs -> convert to python int lists
            for gt_idx, pred_idx in zip(gt_match_idxs.tolist(), pred_match_idxs.tolist()):
                # build the match
                similarity_score = similarity_filtered[gt_idx][
                    pred_idx].item()

                match = {
                    'pred_box': pred_boxes_filtered[pred_idx],
                    'gt_box': gt_boxes_filtered[gt_idx],
                    'label': c,
                    'pred_score':
                    pred_scores_filtered[pred_idx].item(),
                    'similarity_score': similarity_score,
                    'data_id': data_id,
                }
                result[c].append(match)

            # add the non matched gt boxes
            for unmatched_gt_idx in non_matched_gt_idxs.tolist():
                match = {
                    'pred_box': None,
                    'gt_box': gt_boxes_filtered[unmatched_gt_idx],
                    'label': c,
                    'pred_score': float('-inf'),
                    'similarity_score': float('-inf'),
                    'data_id': data_id,
                }
                result[c].append(match)

            # add the non matched pred boxes
            print("non matched preds =", non_matched_pred_idxs)
            for unmatched_pred_idx in non_matched_pred_idxs.tolist():
                print("pred idx =", unmatched_pred_idx,
                      "type =", type(unmatched_pred_idx))
                # unmatched box, create empty match
                match = {
                    'pred_box': pred_boxes_filtered[unmatched_pred_idx],
                    'gt_box': None,
                    'label': c,
                    'pred_score': pred_scores_filtered[unmatched_pred_idx].item(),
                    'similarity_score': float('-inf'),
                    'data_id': data_id,
                }
                result[c].append(match)

        # # sanity check if all gts and all preds are used in results
        # gts = 0
        # preds = 0
        # for c in result.keys():
        #     for m in result[c]:
        #         pred = m['pred_box']
        #         gt = m['gt_box']
        #         if pred:
        #             preds += 1
        #         if gt:
        #             gts += 1

        # assert len(gt_boxes) == gts
        # assert len(pred_boxes) == preds

        return result
