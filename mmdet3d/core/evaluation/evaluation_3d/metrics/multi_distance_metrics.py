import math
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import pickle

from tqdm import tqdm
from matplotlib import pyplot as plt

from mmdet3d.core.evaluation.evaluation_3d import CenterDistance2d, GreedyMatcher
from mmdet3d.core.evaluation.evaluation_3d.filters import BoxDistanceIntervalFilter

from .metric_pipeline import MetricPipeline
from .numeric_class_metric_result import NumericClassMetricResult
from .numeric_metric_result import NumericMetricResult


class MultiDistanceMetric:
    """Can be used to evaluate a pipeline of metrics at different distance
    intervals."""

    SUBPLOTS_PER_ROW = 2
    NUMERIC_PLOT_NAME = 'numeric_metrics.png'
    CLASS_NUMERIC_PLOT_NAME = 'class_numeric_metrics.png'
    GT_OBJECT_COUNT_COLUMN = '#Ground truth boxes'
    PRED_OBJECT_COUNT_COLUMN = '#Prediction boxes'

    def __init__(
        self,
        classes,
        metric_pipeline,
        distance_intervals=[0, 20, 40, 60, 80, 100],
        matcher=None,
        similarity_measure=None,
        reversed_score=False,
        additional_filter_pipeline=None,
    ):
        """Creates the MultiDistanceMetric.

        Args:
            classes (list): List of class ids that are present in the dataset
            metric_pipeline (MetricPipeline): Metrics to evaluate
            distance_intervals (list, optional): List of interval borders, each number is start and end of an interval. Defaults to [0, 20, 40, 60, 80, 100].
            matcher (Matcher, optional): Matcher to use, if None Greedymatcher is used. Defaults to None.
            similarity_measure (SimilarityMeasure, optional): OBB similarity measure, if None CenterDistance2d is used. Defaults to None.
            reversed_score (bool, optional): Wether similarity scores are reversed. Defaults to False.
            additional_filter_pipeline (FilterPipeline, optional): Filters to apply at each interval e.g. min points filter. Defaults to None.
        """
        self._matcher = matcher
        if not self._matcher:
            matcher = GreedyMatcher(classes)

        self._classes = classes

        self._similarity_measure = similarity_measure
        if not self._similarity_measure:
            self._similarity_measure = CenterDistance2d()

        assert isinstance(metric_pipeline, MetricPipeline)

        self._metric_pipeline = metric_pipeline
        self._additional_filter_pipeline = additional_filter_pipeline

        assert len(
            distance_intervals) > 1, 'intervals needs 2 borders at least'
        self._distance_intervals = distance_intervals

        self._distance_filter = None

        self._reversed_scores = reversed_score

    def _preprocess_single(self, result_path, data_id):

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
        input_data = result['data']

        # only consider boxes in the current distance interval
        (
            gt_boxes,
            pred_boxes,
            gt_labels,
            pred_labels,
            pred_scores,
            input_data,
        ) = self._distance_filter.apply(
            gt_boxes,
            pred_boxes,
            gt_labels,
            pred_labels,
            pred_scores,
            input_data,
        )

        # apply additional filters if any
        if self._additional_filter_pipeline:
            (
                gt_boxes,
                pred_boxes,
                gt_labels,
                pred_labels,
                pred_scores,
                input_data,
            ) = self._additional_filter_pipeline.apply(
                gt_boxes,
                pred_boxes,
                gt_labels,
                pred_labels,
                pred_scores,
                input_data,
            )

        # perform matching
        # calculate the similarity for the boxes
        similarity_scores = self._similarity_measure.calc_scores(
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
            reversed_score=self._reversed_scores,
        )

        pred_count = len(pred_boxes)
        pred_count_per_class = {}
        gt_count = len(gt_boxes)
        gt_count_per_class = {}

        for class_id in self._classes:
            if not class_id in gt_count_per_class:
                # init all classes with empty gt counts
                gt_count_per_class[class_id] = 0

            if not class_id in pred_count_per_class:
                # init all classes with empty pred counts
                pred_count_per_class[class_id] = 0

        # get the count for this interval
        c = 0
        for gt_box_label in gt_labels:
            gt_count_per_class[gt_box_label.item()] += 1
            c += 1
        assert c == gt_count

        # get the pred count for this interval
        c = 0
        for pred_box_label in pred_labels:
            pred_count_per_class[pred_box_label.item()] += 1
            c += 1
        assert c == pred_count

        return single_matching_result, pred_count_per_class, gt_count_per_class

    def _evaluate_interval(self, result_paths, min_distance, max_distance):

        # TODO critical change the filter to allow for 360° filtering currently front only
        if not self._distance_filter:
            self._distance_filter = BoxDistanceIntervalFilter(
                min_radius=min_distance, max_radius=max_distance)
        else:
            self._distance_filter.max_radius = max_distance
            self._distance_filter.min_radius = min_distance

        pred_count = 0
        gt_count = 0
        gt_count_per_class = {}
        pred_count_per_class = {}

        matchings_for_interval = {c: [] for c in self._classes}
        for class_id in self._classes:
            gt_count_per_class[class_id] = 0
            pred_count_per_class[class_id] = 0

        for data_id, result_path in result_paths.items():
            # perform matching
            m_result, preds_per_class, gts_per_class = self._preprocess_single(
                result_path, data_id)

            # accumulate matchings for all frames
            for c in m_result.keys():
                matchings_for_interval[c].extend(m_result[c])

            # count boxes
            for c in self._classes:
                gt_count_per_class[c] += gts_per_class[c]
                gt_count += gts_per_class[c]
                pred_count_per_class[c] += preds_per_class[c]
                pred_count += preds_per_class[c]

        # evaluate the metrics for this interval
        metric_results = self._metric_pipeline.evaluate(matchings_for_interval)

        interval_result = {
            'min_dist': min_distance,
            'max_dist': max_distance,
            'pred_count': pred_count,
            'pred_count_per_class': pred_count_per_class,
            'results': metric_results,
            'gt_count': gt_count,
            'gt_count_per_class': gt_count_per_class
        }
        return interval_result

    def evaluate(self, result_paths):
        interval_results = []
        # -1 because interval is defined from dist[i], dist[i+1]
        for interval_idx in tqdm(range(len(self._distance_intervals) - 1)):
            min_dist = self._distance_intervals[interval_idx]
            max_dist = self._distance_intervals[interval_idx + 1]

            interval_result = self._evaluate_interval(
                result_paths, min_dist, max_dist)
            interval_results.append(interval_result)

        return interval_results

    def evaluate_OLD(self, inference_results):
        """Runs the module. For each interval performs filtering, matching and
        then runs the eval pipeline.

        Args:
            inference_results (list): Model outputs
            data (list, optional): Model inputs, if needed by the metrics / filters. Defaults to None.
        """

        # inference results is list of dicts with
        # gt_boxes, gt_labels, pred_labels, pred_boxes, pred_scores, input_data

        interval_results = []
        # -1 because interval is defined from dist[i], dist[i+1]
        for interval_idx in tqdm(range(len(self._distance_intervals) - 1)):
            min_dist = self._distance_intervals[interval_idx]
            max_dist = self._distance_intervals[interval_idx + 1]

            # TODO critical change the filter to allow for 360° filtering currently front only
            if not self._distance_filter:
                self._distance_filter = BoxDistanceIntervalFilter(
                    min_radius=min_dist, max_radius=max_dist)
            else:
                self._distance_filter.max_radius = max_dist
                self._distance_filter.min_radius = min_dist

            matching_results = {c: [] for c in self._classes}

            pred_count = 0
            gt_count = 0
            gt_count_per_class = {}
            pred_count_per_class = {}

            for data_id, frame in enumerate(inference_results):
                pred_boxes = frame['pred_boxes']
                pred_labels = frame['pred_labels']
                pred_scores = frame['pred_scores']
                gt_boxes = frame['gt_boxes']
                gt_labels = frame['gt_labels']
                input_data = frame['input_data']

                # only consider boxes in the current distance interval
                (
                    gt_boxes,
                    pred_boxes,
                    gt_labels,
                    pred_labels,
                    pred_scores,
                    input_data,
                ) = self._distance_filter.apply(
                    gt_boxes,
                    pred_boxes,
                    gt_labels,
                    pred_labels,
                    pred_scores,
                    input_data,
                )

                # apply additional filters if any
                if self._additional_filter_pipeline:
                    (
                        gt_boxes,
                        pred_boxes,
                        gt_labels,
                        pred_labels,
                        pred_scores,
                        input_data,
                    ) = self._additional_filter_pipeline.apply(
                        gt_boxes,
                        pred_boxes,
                        gt_labels,
                        pred_labels,
                        pred_scores,
                        input_data,
                    )

                # perform matching
                # calculate the similarity for the boxes
                similarity_scores = self._similarity_measure.calc_scores(
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
                    reversed_score=self._reversed_scores,
                )

                # accumulate matching results of multiple frames/instances
                for c in single_matching_result.keys():
                    matching_results[c].extend(single_matching_result[c])

                pred_count += len(pred_boxes)

                gt_count += len(gt_boxes)

                for class_id in self._classes:
                    if not class_id in gt_count_per_class:
                        # init all classes with empty gt counts
                        gt_count_per_class[class_id] = 0

                    if not class_id in pred_count_per_class:
                        # init all classes with empty pred counts
                        pred_count_per_class[class_id] = 0

                # get the count for this interval
                for gt_box_label in gt_labels:
                    gt_count_per_class[gt_box_label.item()] += 1

                # get the pred count for this interval
                for pred_box_label in pred_labels:
                    pred_count_per_class[pred_box_label.item()] += 1

            # evaluate the metrics for this interval
            metric_results = self._metric_pipeline.evaluate(matching_results)

            # short sanity check wether gt_count and gt_count per class are equal
            gt_sum = 0
            for class_id, gt_count_class in gt_count_per_class.items():
                gt_sum += gt_count_class

            assert gt_sum == gt_count

            # same for preds
            pred_sum = 0
            for class_id, pred_count_class in pred_count_per_class.items():
                pred_sum += pred_count_class

            assert pred_sum == pred_count

            interval_results.append({
                'min_dist': min_dist,
                'max_dist': max_dist,
                'results': metric_results,
                'pred_count': pred_count,
                'pred_count_per_class': pred_count_per_class,
                'gt_count': gt_count,
                'gt_count_per_class': gt_count_per_class
            })

        return interval_results

    @staticmethod
    def _extract_numeric_class_metric_results(distance_metric_results,
                                              exclude_if_no_gt=False):

        def bin_to_string(bin):
            return '[' + str(bin['min_dist']) + ' , ' + str(
                bin['max_dist']) + ')'

        index = []
        # metric name x classes x results for each bin
        metric_results = {}
        # class name x intervall x gt_count
        gt_counts = {}
        # class name x intervall x pred_count
        pred_counts = {}
        for bin in distance_metric_results:
            if exclude_if_no_gt and bin['gt_count'] == 0:
                continue
            bin_results = bin['results']

            # add the gt counts
            gt_count_per_class = bin['gt_count_per_class']

            for class_id, count in gt_count_per_class.items():
                if not class_id in gt_counts:
                    gt_counts[class_id] = []

                gt_counts[class_id].append(count)

            # add the pred counts
            pred_count_per_class = bin['pred_count_per_class']
            for class_id, count in pred_count_per_class.items():
                if not class_id in pred_counts:
                    pred_counts[class_id] = []

                pred_counts[class_id].append(count)

            # create a name for this bin
            index.append(bin_to_string(bin))

            for metric_name, metric_return in bin_results.items():
                if isinstance(metric_return, NumericClassMetricResult):

                    if not metric_name in metric_results:
                        # first iteration, create the dict
                        metric_results[metric_name] = {}

                    # this return val is a dict  classes x numeric value
                    for class_name, value in metric_return().items():
                        if not class_name in metric_results[metric_name]:
                            # first iteration, add the list for the interval values
                            metric_results[metric_name][class_name] = []

                        # get the numeric vale of the NumericMetricResult
                        value = value()
                        if math.isnan(value):
                            value = 0.0
                            warnings.warn('setting nan to 0.0 for plotting')

                        metric_results[metric_name][class_name].append(value)

        # create a df for the object counts per class
        gt_df = pd.DataFrame(
            {
                class_name: gt_count
                for class_name, gt_count in gt_counts.items()
            },
            index=index)

        pred_df = pd.DataFrame(
            {
                class_name: pred_count
                for class_name, pred_count in pred_counts.items()
            },
            index=index)

        metric_dfs = {}
        # now create the dataframes:
        for metric_name, data in metric_results.items():
            df = pd.DataFrame(
                {class_name: vals
                 for class_name, vals in data.items()},
                index=index)
            metric_dfs[metric_name] = df

        return metric_dfs, gt_df, pred_df

    def _extract_numeric_metric_results(distance_metric_results,
                                        exclude_if_no_gt=False):

        def bin_to_string(bin):
            return '[' + str(bin['min_dist']) + ' , ' + str(
                bin['max_dist']) + ')'

        index = []
        metric_count = None
        # metric name x results for each bin
        metric_results = {}
        gt_counts = []
        pred_counts = []
        for bin in distance_metric_results:
            if exclude_if_no_gt and bin['gt_count'] == 0:
                continue

            gt_counts.append(bin['gt_count'])
            pred_counts.append(bin['pred_count'])
            bin_results = bin['results']

            # create a name for this bin
            index.append(bin_to_string(bin))

            # just as sanity check assert that all bins contain the same metrics
            current_metric_count = 0

            for metric_name, metric_return in bin_results.items():

                if isinstance(metric_return, NumericMetricResult):

                    if not metric_name in metric_results:
                        # first iteration, add the list
                        metric_results[metric_name] = []

                    # this value is a number
                    result_value = float(metric_return())

                    if math.isnan(result_value):
                        warnings.warn('plotting nan result as 0')
                        result_value = 0.0

                    metric_results[metric_name].append(result_value)

                    current_metric_count += 1

            if metric_count is None:
                metric_count = current_metric_count
            else:
                assert metric_count == current_metric_count

        # now fill the dataframe
        df = pd.DataFrame(
            {name: vals
             for name, vals in metric_results.items()}, index=index)

        object_df = pd.DataFrame(
            {
                MultiDistanceMetric.GT_OBJECT_COUNT_COLUMN: gt_counts,
                MultiDistanceMetric.PRED_OBJECT_COUNT_COLUMN: pred_counts
            },
            index=index)

        return df, object_df

    @staticmethod
    def _plot_numeric_metric_results(numeric_df, object_df, out_file):

        full_df = pd.concat([numeric_df, object_df], axis=1)

        ax = full_df.plot(
            kind='bar',
            secondary_y=[
                MultiDistanceMetric.GT_OBJECT_COUNT_COLUMN,
                MultiDistanceMetric.PRED_OBJECT_COUNT_COLUMN
            ],
            rot=0)
        ax.set_xlabel('Distance interval in [m]')
        ax.set_ylabel('Metric value')
        fig = ax.get_figure()
        fig.suptitle('Numeric metrics')
        fig.savefig(out_file)

    @staticmethod
    def _plot_numeric_class_metric_results(metric_dfs,
                                           gt_df,
                                           pred_df,
                                           out_file,
                                           figsize=(10, 10)):
        # subplot for each metric, all combined in a single figure (2 subplots per row)
        cols = MultiDistanceMetric.SUBPLOTS_PER_ROW

        # +1 for the gt object / pred objects plot, if  division is uneven this causes no extra row
        rows = len(metric_dfs) / cols + 2

        last_empty = not rows.is_integer()
        rows = int(rows)

        fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=figsize)

        count = 0
        for metric_name, df in metric_dfs.items():
            # get current row and col
            row = math.floor(count / cols)
            col = count % cols

            df.plot.bar(rot=0, ax=axes[row, col])
            axes[row, col].set_xlabel('Distance interval in [m]')
            axes[row, col].set_ylabel(metric_name)

            count += 1
        # add the gt objects plot
        row = math.floor(count / cols)
        col = count % cols

        # gt objects
        gt_df.plot.bar(rot=0, ax=axes[row, col])

        axes[row, col].set_xlabel('Distance interval in [m]')
        axes[row, col].set_ylabel('#Ground truth')

        # jump to next plot
        count += 1
        row = math.floor(count / cols)
        col = count % cols

        # pred objects
        pred_df.plot.bar(rot=0, ax=axes[row, col])
        axes[row, col].set_xlabel('Distance interval in [m]')
        axes[row, col].set_ylabel('#Predictions')

        if last_empty:
            # remove empty plot (if present)
            axes.flat[-1].set_visible(False)

        fig.suptitle('Numeric class metrics')
        fig.savefig(out_file)

    @staticmethod
    def print_results(distance_metric_results,
                      plot_folder,
                      exclude_if_no_gt=False):
        # process al numeric metrics
        numeric_df, numeric_gt_df = MultiDistanceMetric._extract_numeric_metric_results(
            distance_metric_results, exclude_if_no_gt)

        # process all class numeric metrics
        numeric_class_dfs, class_gt_df, class_pred_df = MultiDistanceMetric._extract_numeric_class_metric_results(
            distance_metric_results, exclude_if_no_gt)

        # check the output folder for the plots
        plot_folder = Path(plot_folder)
        assert plot_folder.is_dir()

        numeric_plot_file = plot_folder.joinpath(
            MultiDistanceMetric.NUMERIC_PLOT_NAME)
        class_numeric_plot_file = plot_folder.joinpath(
            MultiDistanceMetric.CLASS_NUMERIC_PLOT_NAME)

        # plot the numeric results ( all in one plot )
        MultiDistanceMetric._plot_numeric_metric_results(
            numeric_df, numeric_gt_df, numeric_plot_file)

        # plot the numeric class results (each metric in one plot)
        MultiDistanceMetric._plot_numeric_class_metric_results(
            numeric_class_dfs, class_gt_df, class_pred_df,
            class_numeric_plot_file)
