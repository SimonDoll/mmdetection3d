import copy
import pytest
from mmdet3d.core.evaluation.evaluation_3d.metrics import FalsePositivesPerFrameClassMetric, FalsePositivesPerFrame


class TestMetricFalsePositivesPerFrame:
    def test_metric_keeps_results(self):
        # this test should be implemented by all metrics
        # assures that the original matching results are not changed (permutation of results is fine)
        class_id = '0'
        matching_results = {
            class_id: [
                {
                    'similarity_score': 0.9,
                    'gt_box': True,
                    'pred_score': 0.5,
                    'data_id': 0
                },
                {
                    'similarity_score': 1,
                    'gt_box': True,
                    'pred_score': 0.7,
                    'data_id': 0
                },
                {
                    'similarity_score': 0.7,
                    'gt_box': True,
                    'pred_score': 0.1,
                    'data_id': 0
                },
            ]
        }

        match_copy = copy.deepcopy(matching_results)

        metric = FalsePositivesPerFrameClassMetric()
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

    def test_all_frames_fps(self):
        class_id = '0'
        matching_results = {
            class_id: [
                {
                    'similarity_score': 0.9,
                    'gt_box': None,
                    'pred_score': 0.5,
                    'data_id': 0
                },
                {
                    'similarity_score': 1,
                    'gt_box': None,
                    'pred_score': 0.7,
                    'data_id': 1
                },
                {
                    'similarity_score': 0.7,
                    'gt_box': None,
                    'pred_score': 0.1,
                    'data_id': 2
                },
            ]
        }
        metric = FalsePositivesPerFrameClassMetric()
        fps_per_frame = metric.evaluate(matching_results)()

        # all boxes are fps, 3 frames, 3 boxes -> 1.0
        assert fps_per_frame[class_id]() == 1.0

    def test_not_all_frames_fps(self):
        class_id = '0'
        matching_results = {
            class_id: [
                {
                    'similarity_score': 0.9,
                    'gt_box': True,
                    'pred_score': 0.5,
                    'data_id': 1
                },
                {
                    'similarity_score': 1,
                    'gt_box': None,
                    'pred_score': 0.7,
                    'data_id': 2
                },
                {
                    'similarity_score': 0.7,
                    'gt_box': None,
                    'pred_score': 0.1,
                    'data_id': 3
                },
            ]
        }
        metric = FalsePositivesPerFrameClassMetric()
        fps_per_frame = metric.evaluate(matching_results)()

        # 3 frames, 2 fps
        assert fps_per_frame[class_id]() == 2/3

    def test_only_tps(self):
        class_id = '0'
        matching_results = {
            class_id: [
                {
                    'similarity_score': 0.9,
                    'gt_box': True,
                    'pred_score': 0.5,
                    'data_id': 0
                },
                {
                    'similarity_score': 1,
                    'gt_box': True,
                    'pred_score': 0.7,
                    'data_id': 0
                },
                {
                    'similarity_score': 0.7,
                    'gt_box': True,
                    'pred_score': 0.1,
                    'data_id': 0
                },
            ]
        }
        metric = FalsePositivesPerFrameClassMetric()
        fps_per_frame = metric.evaluate(matching_results)()

        # 3 frames, 2 fps
        assert fps_per_frame[class_id]() == 0.0

    def test_fps_per_frame(self):
        class_id = '0'
        matching_results = {
            class_id: [
                {
                    'similarity_score': 0.9,
                    'gt_box': True,
                    'pred_score': 0.5,
                    'data_id': 0
                },
                {
                    'similarity_score': 1,
                    'gt_box': None,
                    'pred_score': 0.7,
                    'data_id': 0
                },
                {
                    'similarity_score': 0.7,
                    'gt_box': None,
                    'pred_score': 0.1,
                    'data_id': 0
                },
            ]
        }
        metric_per_class = FalsePositivesPerFrameClassMetric()
        fps_per_frame_per_class = metric_per_class.evaluate(matching_results)()

        metric_mean_classes = FalsePositivesPerFrame()
        fps_per_frame = metric_mean_classes.evaluate(matching_results)()

        assert pytest.approx(fps_per_frame, fps_per_frame_per_class)
