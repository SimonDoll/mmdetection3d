import numpy as np
import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.evaluation.evaluation_3d.matchers import *
from mmdet3d.core.evaluation.evaluation_3d.similarity_measure import *


class TestGreedyMatcher:
    CLASSES = [1, 2]

    def create_gt_boxes(self):

        gt_box_matched_label = self.CLASSES[0]
        gt_box_unmatched_label = self.CLASSES[1]

        # box dims are x y z w l h yaw
        gt_box_matched = torch.tensor([0, 0, 0, 1, 1, 1, 0.0])
        gt_box_unmatched = torch.tensor([5, 5, 5, 1, 1, 1, 0.0])

        gt_boxes = torch.stack((gt_box_matched, gt_box_unmatched))

        gt_boxes = LiDARInstance3DBoxes(gt_boxes)
        return gt_boxes, torch.tensor(
            [gt_box_matched_label, gt_box_unmatched_label])

    def create_pred_boxes(self, ):
        pred_a_label = self.CLASSES[0]
        pred_b_label = self.CLASSES[1]
        pred_c_label = self.CLASSES[0]

        pred_labels = torch.tensor([pred_a_label, pred_b_label, pred_c_label])

        pred_a_score = 1.0
        pred_b_score = 0.5
        pred_c_score = 0.1

        pred_scores = torch.tensor([pred_a_score, pred_b_score, pred_c_score])

        # box dims are x y z w l h yaw
        pred_box_a = torch.tensor([0.1, 0.0, 0, 1, 1, 1, 0.0])
        pred_box_b = torch.tensor([42, 42, 42, 1, 1, 1, 0.0])
        pred_box_c = torch.tensor([100, 100, 100, 1, 1, 1, 0.0])

        pred_boxes = torch.stack((pred_box_a, pred_box_b, pred_box_c))

        pred_boxes = LiDARInstance3DBoxes(pred_boxes)
        return pred_boxes, pred_labels, pred_scores

    def test_matcher_with_boxes(self, ):

        gt_boxes, gt_labels = self.create_gt_boxes()
        pred_boxes, pred_labels, pred_scores = self.create_pred_boxes()

        similarity_measure = Iou()
        matcher = GreedyMatcher(self.CLASSES)

        similarity_scores = similarity_measure.calc_scores(
            gt_boxes, pred_boxes, gt_labels, pred_labels)
        matching_results = matcher.match(
            similarity_scores,
            gt_boxes,
            pred_boxes,
            gt_labels,
            pred_labels,
            pred_scores,
            data_id=0,
        )

        # desired matching:
        # pred_a -> gt_a (high similarity) class 1
        # pred_b -> gt_b (low similarity) class 2
        # pred_c -> no gt

        # check if all boxes are present
        assert len(matching_results[self.CLASSES[0]]) == 2
        assert len(matching_results[self.CLASSES[1]]) == 1

        # check if the matches are as expeced:
        res_class_0 = matching_results[self.CLASSES[0]]
        res_class_1 = matching_results[self.CLASSES[1]]

        for match in res_class_0:
            # case for true positive
            if torch.all(match['pred_box'].tensor == pred_boxes[0].tensor):
                assert torch.all(match['gt_box'].tensor == gt_boxes[0].tensor)
                assert match['similarity_score'] >= 0.8

            # case of false positive
            elif torch.all(match['pred_box'].tensor == pred_boxes[2].tensor):
                assert match['gt_box'] == None
                assert match['similarity_score'] == float('-inf')
            else:

                assert False, 'wrong box in pred matches'

        match = res_class_1[0]

        # case of match but without overlap
        assert torch.allclose(match['pred_box'].tensor, pred_boxes[1].tensor)
        assert torch.allclose(match['gt_box'].tensor, gt_boxes[1].tensor)
        assert np.isclose(match['similarity_score'], 0.0)

    def test_matcher_without_boxes(self):
        similarity_measure = Iou()
        matcher = GreedyMatcher(self.CLASSES)

        pred_boxes = torch.tensor([], dtype=torch.float)
        gt_boxes = torch.tensor([], dtype=torch.float)

        pred_boxes = LiDARInstance3DBoxes(pred_boxes)
        gt_boxes = LiDARInstance3DBoxes(gt_boxes)

        pred_labels = torch.tensor([], dtype=torch.float)
        gt_labels = torch.tensor([], dtype=torch.float)

        pred_scores = torch.tensor([], dtype=torch.float)

        similarity_scores = similarity_measure.calc_scores(
            gt_boxes, pred_boxes, gt_labels, pred_labels)

        matching_results = matcher.match(
            similarity_scores,
            gt_boxes,
            pred_boxes,
            gt_labels,
            pred_labels,
            pred_scores,
            data_id=0,
        )

        # in the empty case a dict of classes -> empty list should be returned
        assert matching_results == {self.CLASSES[0]: [], self.CLASSES[1]: []}

    def test_matcher_with_pred_boxes_only(self):
        similarity_measure = Iou()
        matcher = GreedyMatcher(self.CLASSES)

        gt_boxes = torch.tensor([], dtype=torch.float)
        gt_boxes = LiDARInstance3DBoxes(gt_boxes)
        gt_labels = torch.tensor([], dtype=torch.float)

        pred_boxes, pred_labels, pred_scores = self.create_pred_boxes()

        similarity_scores = similarity_measure.calc_scores(
            gt_boxes, pred_boxes, gt_labels, pred_labels)

        data_id = 0
        matching_results = matcher.match(
            similarity_scores,
            gt_boxes,
            pred_boxes,
            gt_labels,
            pred_labels,
            pred_scores,
            data_id=data_id,
        )

        expected_results = {}
        for i in range(len(pred_boxes)):
            pred_box = pred_boxes[i]
            label = pred_labels[i].item()
            res = {'pred_box': pred_box,
                   'gt_box': None,
                   'label': label,
                   'pred_score': pred_scores[i].item(),
                   'similarity_score': float("-inf"),
                   'data_id': data_id
                   }
            if not label in expected_results:
                expected_results[label] = []
            expected_results[label].append(res)
        assert expected_results.keys() == matching_results.keys()

        for k in expected_results.keys():
            expected = expected_results[k]
            matching_res = matching_results[k]

            for exp, m in zip(expected,  matching_res):
                e_box = exp.pop('pred_box')
                m_box = m.pop('pred_box')
                assert(torch.allclose(e_box.tensor, m_box.tensor))

                assert(exp == m)
