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

import logging

import pickle
import tqdm


from mmdet3d.core.evaluation.evaluation_3d.similarity_measure import *
from mmdet3d.core.evaluation.evaluation_3d.filters import *
from mmdet3d.core.evaluation.evaluation_3d.metrics import *
from mmdet3d.core.evaluation.evaluation_3d import metrics as metrics
from mmdet3d.core.evaluation.evaluation_3d.matchers import HungarianMatcher


class EvalPipeline:

    _result_paths_file_name = "result_paths.json"
    _cat_to_id_file_name = "cat2id.json"

    _dist_eval_intervals = [0, 20, 40, 60, 80, 100, 120]

    _gt_filter_bounds = [0, -40, 120, 40]

    _m_ap_steps_iou = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    _m_ap_steps_cd = [0.1, 0.5, 1.0, 2.0, 5.0]

    _ap_steps_iou = [0.5, 0.75]
    _ap_steps_cd = [0.1, 0.5, 1.0, 2.0, 5.0]

    _tp_metrics_iou = 0.5
    _tp_metrics_cd = 2.0

    def __init__(self, precompute_path, out_file=None, verbose=True):
        self._precompute_path = pathlib.Path(precompute_path)

        logging.info("Results are loaded from: {}".format(
            self._precompute_path))
        assert self._precompute_path.is_dir(), "Precomputes do not exist"

        self._out_file = None
        if out_file is not None:
            self._out_file = pathlib.Path(out_file)
            assert self._out_file.parent.is_dir(), "out file parent does not exist"
            assert self._out_file.suffix == ".json", "wrong file extension for output"

        self._result_paths_file = self._precompute_path.joinpath(
            self._result_paths_file_name)
        assert self._result_paths_file.is_file(), "result paths list not found"

        self._cat_to_id_file = self._precompute_path.joinpath(
            self._cat_to_id_file_name)
        assert self._cat_to_id_file.is_file(), "cat2id file not found"

        self._cat2id = None
        with open(self._cat_to_id_file) as fp:
            self._cat2id = json.load(fp)

        # we use class ids for matching, cat2id can be used to assign a category name later on
        self._matcher = HungarianMatcher(self._cat2id.values())

        self._verbose = verbose

    def _eval_preprocess(self, result_paths, similarity_measure, reversed_score):
        annotation_count = 0
        annotated_frames_count = 0
        non_annotated_frames_count = 0

        matching_results = {c: [] for c in self._cat2id.values()}

        for data_id, result_path in enumerate(result_paths):

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

            # check if the gt boxes are inside the required range
            boxes_mask = gt_boxes.in_range_bev(self._gt_filter_bounds)
            gt_boxes = gt_boxes[boxes_mask]
            gt_labels = gt_labels[boxes_mask]

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

        if self._verbose:
            MetricPipeline.print_results(full_range_results)

        multi_range_results = self._eval_single_multi_range(
            similarity_measure,  reversed_score, pipeline, result_paths)

        # MultiDistanceMetric.print_results(multi_range_results)

        # TODO critical combine results here
        return full_range_results, multi_range_results

    def _eval_single_full_range(self, similarity_measure, reversed_score, pipeline, result_paths):
        matchings, annotated_frames_count, non_annotated_frames_count, annotation_count = self._eval_preprocess(
            result_paths, similarity_measure, reversed_score)

        if self._verbose:
            print("annotated =", annotated_frames_count, "non annotated =",
                  non_annotated_frames_count, "gt count =", annotation_count)

        eval_results = pipeline.evaluate(matchings)
        return eval_results

    def _eval_single_multi_range(self, similarity_measure, reversed_score, pipeline, result_paths):

        metrics = MultiDistanceMetric(
            self._cat2id.values(), pipeline, self._dist_eval_intervals,
            matcher=self._matcher, similarity_measure=similarity_measure, reversed_score=reversed_score, verbose=self._verbose)

        eval_interval_results = metrics.evaluate(
            result_paths)
        return eval_interval_results

    def _extract_single_class_results(self, results_single_range, class_id=0):
        metric_results = {}
        for metric_name, metric_return in results_single_range.items():
            metric_val = None
            if isinstance(metric_return, metrics.numeric_metric_result.NumericMetricResult):
                # this value is a number
                metric_val = float(metric_return())

            elif isinstance(metric_return, metrics.numeric_class_metric_result.NumericClassMetricResult):
                # pick the right class
                assert class_id in metric_return(), "class {} not found".format(class_id)

                metric_val = metric_return()[class_id]()

            assert metric_name not in metric_results, "metric {} did exist already"            .format(
                metric_name)

            metric_results[metric_name] = metric_val
        return metric_results

    def _eval_iterate(self, sim_measure_config, result_paths):
        """iterates all thresholds and similarity measures and evals each combination

        sim_measure_config (dict): dict has as keys the similarity measure name, the values are dicts {aps, map, tp : } that contain lists / a value for tp metrics thresh of the thresholds to use for the corresponding metrics

        """
        sim_measure_results = {}
        for sim_measure_name, thresholds in sim_measure_config.items():
            if self._verbose:
                print("\n\n" + "=" * 40)
                print("Similarity measure: {}".format(sim_measure_name))
                print("=" * 40)
            if sim_measure_name == "IoU":
                sim_measure = Iou()
                reversed_score = False
            elif sim_measure_name == "CenterDistance":
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
            results_full_range, results_multi_range = self._eval_single(
                sim_measure, reversed_score, pipeline, result_paths)

            # results full range is dict 'metric' : result (class or numeric)

            # results dist interval is list
            # {min_dist, max_dist, pred_count, results{same as full range}}

            results_full_range = self._extract_single_class_results
            (results_full_range)

            results_interval_full_names = {}
            for results_interval in results_multi_range:
                # get the interval information
                start_range = results_interval['min_dist']
                end_range = results_interval['max_dist']

                metric_vals = results_interval['results']
                metric_results = self._extract_single_class_results(
                    metric_vals)

                # now append the interval to the names
                interval_suffix = " r: [{},{})".format(start_range, end_range)
                for key in metric_results:
                    full_key = key + interval_suffix
                    assert full_key not in results_interval_full_names, "key {} did exist already".format(
                        full_key)
                    results_interval_full_names[full_key] = metric_results[key]

            sim_measure_results[sim_measure_name] = {
                'full': results_full_range, 'interval': results_interval_full_names}

        return sim_measure_results

    def _split_non_annoatated(self, result_paths):
        annotated = {}
        non_annoated = {}
        for data_id, result_path in enumerate(result_paths):

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

    def run_eval(self):
        """Runs inference on the validation dataset specified by the model and
        computes metrics.

        """
        # get the precomputed result paths
        result_paths = None
        with open(self._result_paths_file) as fp:
            result_paths = json.load(fp)

        # TODO only use annotated?
        # annotated, non_annotated = self._split_non_annoatated(result_paths)
        # result_paths = list(annotated.values())

        start = datetime.datetime.now()

        # build up the threshold config (TODO with cmd line args)
        iou = {'aps': self._ap_steps_iou,
               'map': self._m_ap_steps_iou,
               'tp': self._tp_metrics_iou}

        cd = {'aps': self._ap_steps_cd,
              'map': self._m_ap_steps_cd,
              'tp': self._tp_metrics_cd}

        thresh_config = {'IoU': iou,
                         'CenterDistance': cd
                         }

        sim_measure_results = self._eval_iterate(
            thresh_config, result_paths)

        if self._out_file is not None:
            logging.info("storing json results to {}".format(self._out_file))

            with open(self._out_file, 'w') as fp:
                json.dump(sim_measure_results, fp, indent=4)

        end = datetime.datetime.now()
        logging.info('runtime eval: {} seconds'.format(
            (end - start).total_seconds()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Custom non dataset dependent eval pipeline')

    parser.add_argument(
        'precompute_path', help='Folder containing eval precomutes')

    parser.add_argument('--out', type=str, help="file to store .json results")

    parser.add_argument('--verbose', action='store_true',
                        help="Wether output should be printed or not")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    eval_pipeline = EvalPipeline(
        args.precompute_path, args.out, args.verbose)
    eval_pipeline.run_eval()
