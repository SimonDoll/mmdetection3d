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
from mmdet3d.core.evaluation.evaluation_3d.matchers import GreedyMatcher


class EvalPipeline:

    def __init__(self, args):
        self._init_cfg(args.config_file)
        # set random seeds
        if args.seed is not None:
            set_random_seed(args.seed, deterministic=args.deterministic)

        self._init_data()
        self._init_model(args.checkpoint_file)

        self._temp_dir = tempfile.TemporaryDirectory()

    def __del__(self):
        # clean up the temp directory
        self._temp_dir.cleanup()

    def _init_cfg(self, config_file):
        self.cfg = Config.fromfile(config_file)

        # set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        self.cfg.model.pretrained = None
        self.cfg.data.val.test_mode = False  # Assure to get ground truth

    def _init_data(self):
        # build the dataloader
        # TODO right config (train / val / test?)
        samples_per_gpu = 1
        dataset = build_dataset(self.cfg.data.val)
        self.data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=1,
            dist=False,
            shuffle=False,
        )

        # get class information
        self.cat2id = dataset.cat2id

    def _init_model(self, checkpoint_file):
        """Initializes the detector from the config and given checkpoint.

        Args:
            checkpoint_file (str): Checkpoint file of trained model
        """
        # TODO which configs?
        self.model = build_detector(
            self.cfg.model, None, test_cfg=self.cfg.test_cfg)
        fp16_cfg = self.cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)
        checkpoint = load_checkpoint(
            self.model, checkpoint_file, map_location='cpu')

        # if problems with backward compatibility (see test.py of mmdetection3d for a fix)
        self.model.CLASSES = checkpoint['meta']['CLASSES']

    def _single_gpu_eval(self):
        self.model.eval()
        results = []
        dataset = self.data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))

        result_paths = []
        for i, data in enumerate(self.data_loader):

            with torch.no_grad():
                annos = dataset.get_ann_info(i)

                gt_boxes = annos['gt_bboxes_3d']
                gt_labels = annos['gt_labels_3d']
                # gt labels are a numpy array -> bring to torch
                gt_labels = torch.from_numpy(gt_labels)

                points = data['points'][0].data[0][0]

                result = self.model(return_loss=False, rescale=True, **data)

                pred_boxes = result[0]['pts_bbox']['boxes_3d']
                pred_labels = result[0]['pts_bbox']['labels_3d']
                pred_scores = result[0]['pts_bbox']['scores_3d']

                result_with_gt_and_data = {
                    'gt_boxes': gt_boxes,
                    'gt_labels': gt_labels,
                    'pred_boxes': pred_boxes,
                    'pred_scores': pred_scores,
                    'pred_labels': pred_labels,
                    'points': points
                }

                result_path = pathlib.Path(
                    str(self._temp_dir.name)).joinpath(str(i) + ".pkl")
                # pickle the data to avoid shared memory overflow
                with open(result_path, "wb") as fp:
                    pickle.dump(result_with_gt_and_data, fp)

                result_paths.append(result_path)

            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
        return result_paths

    def compute_metrics(self, result_paths):
        start = datetime.datetime.now()

        # similarity_meassure = Iou()
        similarity_meassure = CenterDistance2d()

        # if centerpoint dist reverse matching order (lower is better)
        reversed_score = True

        # we use class ids for matching, cat2id can be used to assign a category name later on
        matcher = GreedyMatcher(self.cat2id.values())

        # metrics
        avg_precision_metric = AveragePrecision(
            similarity_threshold=0.5, reversed_score=reversed_score)
        mean_avg_precision_metric = MeanAveragePrecision(
            [0.5, 1, 3], reversed_score=reversed_score)

        metric_pipeline = MetricPipeline(
            [avg_precision_metric, mean_avg_precision_metric])

        matching_results = {c: [] for c in self.cat2id.values()}

        for data_id, result_path in enumerate(tqdm.tqdm(result_paths)):
            # TODO check if this result format holds for all models

            result = None
            with open(result_path, "rb") as fp:
                result = pickle.load(fp)

            assert result is not None, "pickeled result not found"

            pred_boxes = result['pred_boxes']
            pred_labels = result['pred_labels']
            pred_scores = result['pred_scores']

            gt_boxes = result['gt_boxes']
            gt_labels = result['gt_labels']

            if len(gt_labels) == 0:
                continue

            if len(pred_labels) == 0:
                continue

            # print("pred =", pred_boxes)
            # print("gt =", gt_boxes)

            # calculate the similarity for the boxes
            similarity_scores = similarity_meassure.calc_scores(
                gt_boxes, pred_boxes, gt_labels, pred_labels)

            # match gt and predictions
            single_matching_result = matcher.match(
                similarity_scores,
                gt_boxes,
                pred_boxes,
                gt_labels,
                pred_labels,
                pred_scores,
                data_id,
                reversed_score=reversed_score,
            )

            # accumulate matching results of multiple frames/instances
            for c in single_matching_result.keys():
                matching_results[c].extend(single_matching_result[c])

        # evaluate metrics on the results
        metric_results = metric_pipeline.evaluate(matching_results, data=None)

        metric_pipeline.print_results(metric_results)

        end = datetime.datetime.now()
        print('runtime eval millis =', (end - start).total_seconds() * 1000)

    def compute_metrics2(self, inference_results):
        raise NotImplementedError()
        start = datetime.datetime.now()

        similarity_measure = Iou()
        # similarity_measure = CenterDistance2d()

        # if centerpoint dist reverse matching order (lower is better)
        reversed_score = False

        filter_points_in_box = MinPointsInGtFilter()

        filter_pipeline = FilterPipeline([filter_points_in_box])

        matcher = GreedyMatcher(self.cat2id.values())

        similarity_threshold = 0.5
        # metrics
        avg_precision_metric = AveragePrecision(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

        precision_per_class = PrecisionPerClass(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

        recall_per_class = RecallPerClass(
            similarity_threshold=similarity_threshold,
            reversed_score=reversed_score)

        # mean_avg_precision_metric = MeanAveragePrecision(
        #     [0.5, 1, 2, 4], reversed_score=reversed_score)

        mean_avg_precision_metric = MeanAveragePrecision(
            [0.3, 0.5, 0.7], reversed_score=reversed_score)

        precision = Precision(similarity_threshold, reversed_score)
        recall = Recall(similarity_threshold, reversed_score)

        metric_pipeline = MetricPipeline([
            avg_precision_metric, precision_per_class, recall_per_class,
            mean_avg_precision_metric, precision, recall
        ])

        inference_results_preprocessed = []

        for data_id, res in enumerate(tqdm.tqdm(inference_results)):
            # TODO check if this result format holds for all models

            # get inference resutls and input data
            input_data = res['data']
            inference_result = res['result'][0]

            inference_result = inference_result['pts_bbox']

            pred_boxes = inference_result['boxes_3d']

            pred_labels = inference_result['labels_3d']
            pred_scores = inference_result['scores_3d']
            # gt are wrapped in additional lists
            # TODO check reason
            gt_boxes = inference_result['gt_bboxes_3d'][0][0]
            gt_labels = inference_result['gt_labels_3d'][0][0]

            res_preprocessed = {
                'input_data': input_data,
                'gt_boxes': gt_boxes,
                'pred_boxes': pred_boxes,
                'gt_labels': gt_labels,
                'pred_labels': pred_labels,
                'pred_scores': pred_scores,
            }
            inference_results_preprocessed.append(res_preprocessed)

        print("frames =", len(inference_results_preprocessed))

        multi_distance_metric = MultiDistanceMetric(
            self.cat2id.values(),
            metric_pipeline,
            distance_intervals=[0, 10, 20, 30],
            similarity_measure=similarity_measure,
            reversed_score=reversed_score,
            matcher=matcher,
            additional_filter_pipeline=filter_pipeline,
        )

        distance_interval_results = multi_distance_metric.evaluate(
            inference_results_preprocessed)

        MultiDistanceMetric.print_results(distance_interval_results,
                                          '/workspace/work_dirs/plots')

        # for dist in distance_interval_results:
        #     print("interval =", dist["min_dist"], ",", dist["max_dist"])
        #     print("gt =", dist["gt_count"], ", pred =", dist["pred_count"])
        #     # print("res =", dist["results"])
        #     MetricPipeline.print_results(dist["results"])

        end = datetime.datetime.now()
        print('runtime eval millis =', (end - start).total_seconds() * 1000)

    def run_eval(self, tmpdir=None):
        """Runs inference on the validation dataset specified by the model and
        computes metrics.

        Args:
            tmpdir (str, optional): Folder to save intermediate results to (mainly for debugging). Defaults to None.
        """
        self.model = MMDataParallel(self.model, device_ids=[0])
        result_paths = self._single_gpu_eval()
        self.compute_metrics(result_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Custom non dataset dependent eval pipeline')

    parser.add_argument(
        'config_file', type=str, help='Model configuration (test?)')
    parser.add_argument(
        'checkpoint_file', type=str, help='Trained model checkpoint')

    # TODO output folder
    # parser.add_argument('--out', help='output result file in pickle format')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.',
    )

    # TODO adapt?
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    eval_pipeline = EvalPipeline(args)
    eval_pipeline.run_eval()
