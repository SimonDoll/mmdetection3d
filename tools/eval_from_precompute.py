import mmcv
import os

import argparse
import tempfile
import time
import os.path as osp
import shutil
import tempfile
import datetime
import pathlib
import json
from mmdet3d.core.evaluation.evaluation_3d import similarity_measure

import torch
import pickle
import tqdm

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.core import wrap_fp16_model
from mmdet.core import encode_mask_results
from mmdet3d.core import BboxOverlapsNearest3D
from mmdet3d.core import MaxIoUAssigner

from tools.fuse_conv_bn import fuse_module


from mmdet3d.core.evaluation.evaluation_3d.similarity_measure import *
from mmdet3d.core.evaluation.evaluation_3d.filters import *
from mmdet3d.core.evaluation.evaluation_3d.metrics import *
from mmdet3d.core.evaluation.evaluation_3d.matchers import GreedyMatcher, HungarianMatcher


class EvalPipeline:

    _result_paths_file_name = "result_paths.json"
    _cat_to_id_file_name = "cat2id.json"

    _dist_eval_intervals = [0, 20, 40, 60, 80, 100, 120]
    _m_ap_steps_iou = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    _m_ap_steps_cd = [0.1, 0.5, 1.0, 2.0, 5.0]

    _ap_steps_iou = [0.5, 0.75]
    _ap_steps_cd = [0.1, 0.5, 1.0, 2.0, 5.0]

    _tp_metrics_iou = 0.5
    _tp_metrics_cd = 2.0

    def __init__(self, args):
        self._precompute_path = pathlib.Path(args.precompute_path)

        print("Results are loaded from: {}".format(self._precompute_path))
        assert self._precompute_path.is_dir(), "Precomputes do not exist"

        self._result_paths_file = self._precompute_path.joinpath(
            self._result_paths_file_name)
        assert self._result_paths_file.is_file(), "result paths list not found"

        self._cat_to_id_file = self._precompute_path.joinpath(
            self._cat_to_id_file_name)
        assert self._cat_to_id_file.is_file(), "cat2id file not found"

        self._cat2id = None
        with open(self._cat_to_id_file) as fp:
            self._cat2id = json.load(fp)
        self._init_metrics()

    def _init_metrics(self, sim_measure):

        if sim_measure == "iou":
            self._similarity_measure = Iou()

        self._similarity_measure = Iou()
        # similarity_measure = CenterDistance2d()

        # if centerpoint dist reverse matching order (lower is better)
        self._reversed_score = False

        # we use class ids for matching, cat2id can be used to assign a category name later on
        self._matcher = HungarianMatcher(self._cat2id.values())

        self._similarity_threshold = similarity_threshold

        # metrics
        avg_precision_metric = AveragePrecision(
            similarity_threshold=self._similarity_threshold, reversed_score=self._reversed_score)

        mean_avg_precision_metric = MeanAveragePrecision(
            self._m_ap_steps, reversed_score=self._reversed_score)

        recall_metric = Recall(self._similarity_threshold)
        precision_metric = Precision(self._similarity_threshold)

        fp_per_frame_metric = FalsePositivesPerFrame(
            self._similarity_threshold)

        self._metric_pipeline_annotated = MetricPipeline(
            [avg_precision_metric, mean_avg_precision_metric, precision_metric, recall_metric, fp_per_frame_metric])

        self._metric_pipeline_non_annoated = MetricPipeline(
            [fp_per_frame_metric])

    def _eval_preprocess(self, result_paths, similarity_measure, reversed_score):
        annotation_count = 0
        annotated_frames_count = 0
        non_annotated_frames_count = 0

        matching_results = {c: [] for c in self._cat2id.values()}

        for data_id, result_path in tqdm.tqdm(result_paths.items()):

            # TODO check if this result format holds for all models
            result = None
            with open(result_path, "rb") as fp:
                result = pickle.load(fp)

            assert result is not None, "pickeled result not found {}".format(
                result_path)

            pred_boxes = result['pred_boxes']
            pred_labels = result['pred_labels']
            pred_scores = result['pred_scores']

            gt_boxes = result['gt_boxes']
            gt_labels = result['gt_labels']

            # calculate the similarity for the boxes
            similarity_scores = similarity_measure.calc_scores(
                gt_boxes, pred_boxes, gt_labels, pred_labels)

            # match gt and predictions
            single_matching_result = self._matcher.match(
                similarity_scores,
                gt_boxes,
                pred_boxes,
                gt_labels,
                pred_labels,
                pred_scores,
                data_id,
                reversed_score=reversed_score,
            )

            for c in single_matching_result.keys():
                matching_results[c].extend(
                    single_matching_result[c])
            if len(gt_labels) == 0:
                # no annotations for this frame
                non_annotated_frames_count += 1

            else:
                annotated_frames_count += 1
                annotation_count += len(gt_labels)

        return matching_results, annotated_frames_count, non_annotated_frames_count, annotation_count

    def _eval_single(self, similarity_measure, reversed_score, pipeline, result_paths):
        full_range_results = self._eval_single_full_range(
            similarity_measure, reversed_score, pipeline, result_paths)

        multi_range_results = self._eval_single_multi_range(
            similarity_measure,  reversed_score, pipeline, result_paths)

        # TODO critical combine results here

    def _eval_single_full_range(self, similarity_measure, reversed_score, pipeline, result_paths):
        matchings, annotated_frames_count, non_annotated_frames_count, annotation_count = self._eval_preprocess(
            result_paths, similarity_measure, reversed_score)

        eval_results = pipeline.evaluate(matchings)
        return eval_results

    def _eval_single_multi_range(self, similarity_measure, reversed_score, pipeline, result_paths):

        metrics = MultiDistanceMetric(
            self._cat2id.values(), pipeline, self._dist_eval_intervals,
            matcher=self._matcher, similarity_measure=similarity_measure, reversed_score=reversed_score)

        eval_interval_results = metrics.evaluate(
            result_paths)
        return eval_interval_results

    def _eval_iterate(self, sim_measure_config, result_paths):
        """iterates all thresholds and similarity measures and evals each combination

        sim_measure_config (dict): dict has as keys the similarity measure name, the values are dicts {aps, map, tp : } that contain lists / a value for tp metrics thresh of the thresholds to use for the corresponding metrics

        """

        for sim_measure_name, thresholds in sim_measure_config.items():
            if sim_measure_name == "iou":
                sim_measure = Iou()
                reversed_score = False
            elif sim_measure_name == "centerdist":
                sim_measure = CenterDistance2d()
                reversed_score = True
            else:
                raise ValueError(
                    "unknown sim measure: {}".format(sim_measure_name))

            ap_threshs = thresholds['aps']
            map_thresh_list = thresholds['map']

            tp_metric_thresh = thresholds['tp']

            # build metrics
            metrics = []
            for ap_thresh in ap_threshs:
                ap = AveragePrecision(ap_thresh, reversed_score)
                metrics.append(ap)

            map = MeanAveragePrecision(map_thresh_list, reversed_score)
            metrics.append(map)

            # TODO critical tp metrics

            # build up a pipeline
            pipeline = MetricPipeline(metrics)
            results = self._eval_single(sim_measure, reversed_score, pipeline)
            # TODO do something with the results

    def _split_non_annoatated(self, result_paths):
        annotated = {}
        non_annoated = {}
        for data_id, result_path in enumerate(tqdm.tqdm(result_paths)):

            # TODO check if this result format holds for all models
            result = None
            with open(result_path, "rb") as fp:
                result = pickle.load(fp)

            assert result is not None, "pickeled result not found {}".format(
                result_path)

            gt_labels = result['gt_labels']
            if len(gt_labels) == 0:
                non_annoated[data_id] = result_path
            else:
                annotated[data_id] = result_path
        return annotated, non_annoated

    def compute_metrics(self, result_paths):
        start = datetime.datetime.now()

        annotated_paths, non_annoated_paths = self._split_non_annoatated(
            result_paths)

        self._eval_full_range(annotated_paths, non_annoated_paths)
        self._eval_distance_intervals(annotated_paths, non_annoated_paths)

        end = datetime.datetime.now()
        print('runtime eval millis =', (end - start).total_seconds() * 1000)

    def run_eval(self):
        """Runs inference on the validation dataset specified by the model and
        computes metrics.

        """
        # get the precomputed result paths
        result_paths = None
        with open(self._result_paths_file) as fp:
            result_paths = json.load(fp)

        self.compute_metrics(result_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Custom non dataset dependent eval pipeline')

    parser.add_argument(
        'precompute_path', help='Folder containing eval precomutes')

    parser.add_argument(
        '--similarity', choices=['iou', 'centerdist'], help="Used similarity measure")

    args = parser.parse_args()

    eval_pipeline = EvalPipeline(args)
    eval_pipeline.run_eval()
