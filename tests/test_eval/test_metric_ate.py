import copy

import pytest
import numpy as np
import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.evaluation.evaluation_3d.matchers import *
from mmdet3d.core.evaluation.evaluation_3d.similarity_measure import *


from mmdet3d.core.evaluation.evaluation_3d.metrics import AverageTranslationErrorPerClass


class TestMetricATEPerClass:
    CLASSES = [1, 2]

    # def test_metric_keeps_results(self):
    #     # this test should be implemented by all metrics
    #     # assures that the original matching results are not changed (permutation of results is fine)
    #     class_id = '0'
    #     matching_results = {
    #         class_id: [
    #             {
    #                 'similarity_score': 0.9,
    #                 'gt_box': True,
    #                 'pred_score': 0.5,
    #                 'data_id': 0
    #             },
    #             {
    #                 'similarity_score': 1,
    #                 'gt_box': True,
    #                 'pred_score': 0.7,
    #                 'data_id': 0
    #             },
    #             {
    #                 'similarity_score': 0.7,
    #                 'gt_box': True,
    #                 'pred_score': 0.1,
    #                 'data_id': 0
    #             },
    #         ]
    #     }

    #     match_copy = copy.deepcopy(matching_results)

    #     metric = AverageTranslationErrorPerClass()
    #     metric.evaluate(matching_results)

    #     for class_id in matching_results.keys():
    #         assert class_id in match_copy.keys()

    #         real_res = matching_results[class_id]
    #         copy_res = matching_results[class_id]

    #         assert len(real_res) == len(copy_res)

    #         for i in range(len(real_res)):
    #             real_res_dict = real_res[i]
    #             copy_res_dict = copy_res[i]
    #             assert real_res_dict == copy_res_dict

    # def test_all_frames_fps(self):
    #     class_id = '0'
    #     matching_results = {
    #         class_id: [
    #             {
    #                 'similarity_score': 0.9,
    #                 'gt_box': None,
    #                 'pred_score': 0.5,
    #                 'data_id': 0
    #             },
    #             {
    #                 'similarity_score': 1,
    #                 'gt_box': None,
    #                 'pred_score': 0.7,
    #                 'data_id': 1
    #             },
    #             {
    #                 'similarity_score': 0.7,
    #                 'gt_box': None,
    #                 'pred_score': 0.1,
    #                 'data_id': 2
    #             },
    #         ]
    #     }
    #     metric = AverageTranslationErrorPerClass()
    #     ate = metric.evaluate(matching_results)()

    #     # all boxes are fps -> should be inf
    #     assert ate[class_id]() == float("inf")

    def create_gt_boxes(self):

        gt_box1_label = self.CLASSES[0]
        gt_box2_label = self.CLASSES[1]
        gt_box_unmatched_label = self.CLASSES[0]

        # box dims are x y z w l h yaw
        gt_box_1 = torch.tensor([2, 2, 2, 1, 1, 1, 0.0])
        gt_box_2 = torch.tensor([25, 25, 25, 1, 1, 1, 0.0])
        gt_box_unmatched = torch.tensor([50, 50, 50, 1, 1, 1, 0.0])

        gt_boxes = torch.stack((gt_box_1, gt_box_2, gt_box_unmatched))

        gt_boxes = LiDARInstance3DBoxes(gt_boxes)
        return gt_boxes, torch.tensor(
            [gt_box1_label, gt_box2_label, gt_box_unmatched_label])

    def create_pred_boxes(self, ):
        pred_a_label = self.CLASSES[0]
        pred_b_label = self.CLASSES[1]
        pred_c_label = self.CLASSES[1]

        pred_labels = torch.tensor([pred_a_label, pred_b_label, pred_c_label])

        pred_a_score = 1.0
        pred_b_score = 0.5
        pred_c_score = 0.1

        pred_scores = torch.tensor([pred_a_score, pred_b_score, pred_c_score])

        # box dims are x y z w l h yaw
        pred_box_a = torch.tensor([2.0, 2.0, 2.1, 1, 1, 1, 0.0])
        pred_box_b = torch.tensor([24, 24, 24, 1, 1, 1, 0.0])
        pred_box_c = torch.tensor([100, 100, 100, 1, 1, 1, 0.0])

        pred_boxes = torch.stack((pred_box_a, pred_box_b, pred_box_c))

        pred_boxes = LiDARInstance3DBoxes(pred_boxes)
        return pred_boxes, pred_labels, pred_scores

    def test_ate_per_class(self):
        gt_boxes, gt_labels = self.create_gt_boxes()
        pred_boxes, pred_labels, pred_scores = self.create_pred_boxes()

        similarity_measure = CenterDistance2d()
        reversed_score = True
        matcher = HungarianMatcher(self.CLASSES)

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
            reversed_score=reversed_score
        )

        metric = AverageTranslationErrorPerClass(
            similarity_threshold=5.0, reversed_score=reversed_score)
        ate = metric.evaluate(matching_results)()

        ate_class_0 = ate[self.CLASSES[0]]()
        assert pytest.approx(ate_class_0, 0.1)
        ate_class_1 = ate[self.CLASSES[1]]()
        assert pytest.approx(ate_class_1, 1.732051)
