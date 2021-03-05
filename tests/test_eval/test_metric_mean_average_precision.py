import copy
import numpy as np
import torch

from mmdet3d.core.evaluation.evaluation_3d.metrics import (AveragePrecision,
                                                           MeanAveragePrecision
                                                           )


class TestMetricAveragePrecision:

    def test_metric_keeps_results(self):
        # this test should be implemented by all metrics
        # assures that the original matching results are not changed (permutation of results is fine)
        class_id = '0'
        matching_results = {
            class_id: [
                {
                    'similarity_score': 0.9,
                    'gt_box': True,
                    'pred_score': 0.5
                },
                {
                    'similarity_score': 1,
                    'gt_box': True,
                    'pred_score': 0.7
                },
                {
                    'similarity_score': 0.7,
                    'gt_box': True,
                    'pred_score': 0.1
                },
            ]
        }

        match_copy = copy.deepcopy(matching_results)

        metric = MeanAveragePrecision(similarities=[0.5])
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

    def test_single_class_high_confidence(self):
        class_id = '0'
        matching_results = {
            class_id: [
                {
                    'similarity_score': 0.9,
                    'gt_box': True,
                    'pred_score': 0.5
                },
                {
                    'similarity_score': 1,
                    'gt_box': True,
                    'pred_score': 0.7
                },
                {
                    'similarity_score': 0.7,
                    'gt_box': True,
                    'pred_score': 0.1
                },
            ]
        }
        # all boxes will be tps at each iteration -> res should be same as ap (as we have only a single class)
        similarities = np.arange(
            0.5,
            0.7,
            0.01,
        )
        m_ap_metric = MeanAveragePrecision(similarities)
        m_ap = m_ap_metric.evaluate(matching_results)()

        ap_metric = AveragePrecision()

        ap = ap_metric.evaluate(matching_results)()[class_id]()

        assert np.isclose(m_ap, ap)

    def gen_pseudo_matching_results(self):

        # use only one class as results are computed on a per class base
        class_id = 1

        # matching results only need to have similarity_scores for similarity, gt_boxes (none for not set) and a confidence score (make all boxes highly confident)
        matching_results = {
            class_id: [  # tp, fp, fn
                {
                    'similarity_score': 0.7,
                    'gt_box': True,
                    'pred_score': 1.0
                },
                {
                    'similarity_score': 0.5,
                    'gt_box': None,
                    'pred_score': 1.0
                },
                {
                    'similarity_score': float('-inf'),
                    'gt_box': True,
                    'pred_score': 1.0
                },
            ]
        }
        return matching_results, class_id

    def test_combined_single_class(self):
        matchings, class_id = self.gen_pseudo_matching_results()

        start = 0.1
        stop = 0.9
        step = 0.1

        similarities = np.arange(
            0.1,
            0.9,
            0.1,
        )
        m_ap_metric = MeanAveragePrecision(similarities=similarities)

        ap_metric = AveragePrecision(similarity_threshold=[0])
        m_ap = m_ap_metric.evaluate(matchings, data=None)()

        ap_list = []

        # do the mean by hand
        for i, similarity in enumerate(similarities):
            ap_metric.similarity_threshold = similarity
            ap = ap_metric.evaluate(matchings)()
            ap = ap[class_id]()

            ap_sum = 0
            classes = 0
            if ap is not None:
                ap_sum += ap
                classes += 1
            ap_list.append(ap_sum / classes)

        m_ap_own = np.asarray(ap_list).mean()

        assert np.isclose(m_ap, m_ap_own)
