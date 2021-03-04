import torch
import numpy as np
import copy
from mmdet3d.core.evaluation.evaluation_3d.metrics import AveragePrecision


class TestMetricAveragePrecision:
    def test_metric_keeps_results(self):
        # this test should be implemented by all metrics
        # assures that the original matching results are not changed (permutation of results is fine)
        class_id = "0"
        matching_results = {
            class_id: [
                {"similarity_score": 0.9, "gt_box": True, "pred_score": 0.5},
                {"similarity_score": 1, "gt_box": True, "pred_score": 0.7},
                {"similarity_score": 0.7, "gt_box": True, "pred_score": 0.1},
            ]
        }

        match_copy = copy.deepcopy(matching_results)

        metric = AveragePrecision()
        metric.evaluate(matching_results)

        for class_id in matching_results.keys():
            assert class_id in match_copy.keys()

            real_res = matching_results[class_id]
            copy_res = matching_results[class_id]

            assert len(real_res) == len(copy_res)

            for i in range(len(real_res)):
                real_res_dict = real_res[i]
                copy_res_dict = copy_res[i]
                assert real_res_dict == copy_res_dict

    def test_tps(self):
        class_id = "0"
        matching_results = {
            class_id: [
                {"similarity_score": 0.9, "gt_box": True, "pred_score": 0.5},
                {"similarity_score": 1, "gt_box": True, "pred_score": 0.7},
                {"similarity_score": 0.7, "gt_box": True, "pred_score": 0.1},
            ]
        }
        metric = AveragePrecision(similarity_threshold=0.5)
        aps = metric.evaluate(matching_results)
        ap = aps()[class_id]()
        assert np.isclose(ap, torch.tensor(1.0))

    def test_fps(self):
        class_id = "0"
        matching_results = {
            class_id: [
                {"similarity_score": 0.9, "gt_box": None, "pred_score": 0.5},
                {"similarity_score": 1, "gt_box": None, "pred_score": 0.7},
                {"similarity_score": 0.7, "gt_box": None, "pred_score": 0.1},
            ]
        }
        metric = AveragePrecision(similarity_threshold=0.5)
        aps = metric.evaluate(matching_results)
        ap = aps()[class_id]()
        assert np.isclose(ap, torch.tensor(0.0))

    def test_fns(self):
        class_id = "0"
        matching_results = {
            class_id: [
                {"similarity_score": float(
                    "-inf"), "gt_box": True, "pred_score": 0.5},
                {"similarity_score": float(
                    "-inf"), "gt_box": True, "pred_score": 0.7},
                {"similarity_score": float(
                    "-inf"), "gt_box": True, "pred_score": 0.1},
            ]
        }
        metric = AveragePrecision(similarity_threshold=0.5)
        aps = metric.evaluate(matching_results)
        ap = aps()[class_id]()
        assert np.isclose(ap, torch.tensor(0.0))

    def gen_pseudo_matching_results(self):

        # use only one class as results are computed on a per class base
        class_id = 1

        # matching results only need to have similarity_scores for similarity, gt_boxes (none for not set) and a confidence score (make all boxes highly confident)
        matching_results = {
            class_id: [  # tp, fp, fn
                {"similarity_score": 1, "gt_box": True, "pred_score": 1.0},
                {"similarity_score": 0.1, "gt_box": None, "pred_score": 1.0},
                {"similarity_score": float(
                    "-inf"), "gt_box": True, "pred_score": 1.0},
            ]
        }
        return matching_results, class_id

    def test_combined(self):
        matchings, class_id = self.gen_pseudo_matching_results()
        metric = AveragePrecision(similarity_threshold=0.5)
        res = metric.evaluate(matchings, data=None)

        # the given boxes are tp, fp, fn (all confidence = 1)
        # conf decision prec recall
        # 1.0  TP       1/1  1/2
        # 1.0  FP       1/2  1/2
        # 1.0  FN       1/2  1/2

        precisions = torch.tensor([1, 1 / 2, 1 / 2])
        recalls = torch.tensor([1 / 2, 1 / 2, 1 / 2])

        # for trapz rule add start point at (x=0,y=1)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # we have only one class
        avg_precision = torch.trapz(precisions, recalls)

        assert np.isclose(res()[class_id](), avg_precision)
