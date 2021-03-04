import torch

from mmdet3d.core.evaluation.evaluation_3d import *
from mmdet3d.core.evaluation.evaluation_3d.metrics.base_metric import Basemetric


class TestMetrics:
    def gen_pseudo_matching_results(self):

        # use only one class as results are computed on a per class base
        class_id = 1

        # fp, tp, fn
        # matching results only need to have similarity_scores for similarity, gt_boxes (none for not set) and a confidence score (make all boxes highly confident)
        matching_results = {
            class_id: [  # fp, tp, fn
                {"similarity_score": 0.1, "gt_box": None, "pred_score": 1.0},
                {"similarity_score": 1, "gt_box": True, "pred_score": 1.0},
                {"similarity_score": float(
                    "-inf"), "gt_box": True, "pred_score": 1.0},
            ]
        }
        return matching_results, class_id

    def test_base_metric_decisions_no_boxes(self):
        class_id = "0"
        matching_results = {class_id: []}

        decisions = Basemetric.compute_decisions(
            matching_results, similarity_treshold=0.5, return_idxs=False
        )

        tps = decisions[class_id]["tps"]
        fps = decisions[class_id]["fps"]
        gts = decisions[class_id]["gts"]

        assert tps == 0
        assert fps == 0
        assert gts == 0

    def test_base_metric_tps(self):
        class_id = "0"
        matching_results = {
            class_id: [
                {"similarity_score": 0.9, "gt_box": True},
                {"similarity_score": 1, "gt_box": True},
                {"similarity_score": 0.7, "gt_box": True},
            ]
        }

        decisions = Basemetric.compute_decisions(
            matching_results, similarity_treshold=0.5, return_idxs=True
        )

        tps = decisions[class_id]["tps"]
        fps = decisions[class_id]["fps"]
        gts = decisions[class_id]["gts"]

        assert torch.all(tps)
        assert torch.all(~fps)
        assert torch.all(gts)

    def test_base_metric_fps(self):
        class_id = "0"
        matching_results = {
            class_id: [
                {"similarity_score": 0.0, "gt_box": True},
                {"similarity_score": 0.1, "gt_box": True},
                {"similarity_score": 0.2, "gt_box": None},
            ]
        }

        decisions = Basemetric.compute_decisions(
            matching_results, similarity_treshold=0.5, return_idxs=True
        )

        tps = decisions[class_id]["tps"]
        fps = decisions[class_id]["fps"]
        gts = decisions[class_id]["gts"]

        assert torch.all(~tps)
        assert torch.all(fps)
        assert torch.all(gts == torch.tensor([True, True, False]))

    def test_base_metric_fns(self):
        class_id = "0"
        matching_results = {
            class_id: [
                {"similarity_score": float("-inf"), "gt_box": True},
                {"similarity_score": float("-inf"), "gt_box": True},
                {"similarity_score": float("-inf"), "gt_box": True},
            ]
        }

        decisions = Basemetric.compute_decisions(
            matching_results, similarity_treshold=0.5, return_idxs=True
        )

        tps = decisions[class_id]["tps"]
        fps = decisions[class_id]["fps"]
        gts = decisions[class_id]["gts"]

        assert torch.all(~tps)
        assert torch.all(~fps)
        assert torch.all(gts)

    def test_base_metric_decisions(self):
        matchings, class_id = self.gen_pseudo_matching_results()
        decisions = AveragePrecision.compute_decisions(
            matchings, similarity_treshold=0.5, return_idxs=True
        )

        tps = decisions[class_id]["tps"]
        fps = decisions[class_id]["fps"]
        gts = decisions[class_id]["gts"]

        assert torch.all(tps == torch.tensor([False, True, False]))
        assert torch.all(fps == torch.tensor([True, False, False]))
        assert torch.all(gts == torch.tensor([False, True, True]))
