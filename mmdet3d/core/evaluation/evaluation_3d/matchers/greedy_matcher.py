import torch

from .matcher import Matcher


class GreedyMatcher(Matcher):
    """Greedy matching is based on using the best pred box for the current gt
    box."""

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

        # this matcher is greedy, sort preds in descending confidence and then use local max as match
        # this will result in taking the most confident box in case of same similarity
        pred_idxs = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[pred_idxs]
        pred_scores = pred_scores[pred_idxs]
        pred_labels = pred_labels[pred_idxs]

        similarity_scores = similarity_scores[pred_idxs]

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
            # get prediction idxs that are not matched yet
            non_matched_pred_idxs = set(range(similarity_filtered.shape[1]))
            for gt_idx in range(len(similarity_filtered)):

                # check if there are preds left to check
                preds_available = len(non_matched_pred_idxs) != 0

                if preds_available:
                    # find best match -> convert to python scalar
                    if not reversed_score:
                        best_match_idx = torch.argmax(
                            similarity_filtered[gt_idx]).item()
                    else:
                        best_match_idx = torch.argmin(
                            similarity_filtered[gt_idx]).item()

                    similarity_score = similarity_filtered[gt_idx][
                        best_match_idx].item()

                if preds_available and best_match_idx in non_matched_pred_idxs:
                    # this pred was not matched before -> match
                    non_matched_pred_idxs.remove(best_match_idx)
                    # set score for this pred to a minimum (or maximum if reversed) for all following gts to prevent argmax / argmin from choosing it again
                    if not reversed_score:
                        similarity_filtered[:, best_match_idx] = float('-inf')
                    else:
                        similarity_filtered[:, best_match_idx] = float('inf')

                    # build the match
                    match = {
                        'pred_box': pred_boxes_filtered[best_match_idx],
                        'gt_box': gt_boxes_filtered[gt_idx],
                        'label': c,
                        'pred_score':
                        pred_scores_filtered[best_match_idx].item(),
                        'similarity_score': similarity_score,
                        'data_id': data_id,
                    }
                    result[c].append(match)
                else:
                    # no pred box was available e.g. all taken already or no preds -> unmatched gt
                    match = {
                        'pred_box': None,
                        'gt_box': gt_boxes_filtered[gt_idx],
                        'label': c,
                        'pred_score': float('-inf'),
                        'similarity_score': float('-inf'),
                        'data_id': data_id,
                    }
                    result[c].append(match)

            # take care of all preds that have not been matched so far
            if non_matched_pred_idxs:
                # some pred boxes do not have a matched gt
                for pred_idx in non_matched_pred_idxs:
                    # unmatched box, create empty match
                    match = {
                        'pred_box': pred_boxes_filtered[pred_idx],
                        'gt_box': None,
                        'label': c,
                        'pred_score': pred_scores_filtered[pred_idx].item(),
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
