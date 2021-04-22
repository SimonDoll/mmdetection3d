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
from mmdet3d.core.evaluation.evaluation_3d.matchers import GreedyMatcher, HungarianMatcher


class EvalPipeline:

    _dist_eval_intervals = [0, 20, 40, 60, 80, 100, 120]
    _m_ap_steps = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def __init__(self, args):
        self._init_cfg(args.config_file)
        # set random seeds
        if args.seed is not None:
            set_random_seed(args.seed, deterministic=args.deterministic)

        self._init_data()
        self._init_model(args.checkpoint_file)

        self._temp_dir = tempfile.TemporaryDirectory()

        if args.out:
            self._result_base_path = pathlib.Path(
                args.out).joinpath("pkl_results")
            print("Results will be saved in: {}".format(self._result_base_path))

        else:
            self._result_base_path = pathlib.Path(str(self._temp_dir.name))

        self._intermediate_res_path = self._result_base_path.joinpath(
            "pkl_results")

        self._intermediate_res_path.mkdir()

        self._init_metrics()

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
        self.cfg.data.test.test_mode = False  # Assure to get ground truth

    def _init_data(self):
        # build the dataloader
        # TODO right config (train / val / test?)
        samples_per_gpu = 1
        dataset = build_dataset(self.cfg.data.test)
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

    def _init_metrics(self, similarity_threshold=0.5):
        self._similarity_measure = Iou()
        # similarity_measure = CenterDistance2d()

        # if centerpoint dist reverse matching order (lower is better)
        self._reversed_score = False

        # we use class ids for matching, cat2id can be used to assign a category name later on
        self._matcher = HungarianMatcher(self.cat2id.values())

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

    def _single_gpu_eval(self):
        self.model.eval()
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

                # TODO add images if present
                data = {}
                data['points'] = points

                result_with_gt_and_data = {
                    'gt_boxes': gt_boxes,
                    'gt_labels': gt_labels,
                    'pred_boxes': pred_boxes,
                    'pred_scores': pred_scores,
                    'pred_labels': pred_labels,
                    'data': data
                }

                result_path = self._intermediate_res_path.joinpath(
                    str(i) + ".pkl")
                # pickle the data to avoid shared memory overflow
                with open(result_path, "wb") as fp:
                    pickle.dump(result_with_gt_and_data, fp)

                result_paths.append(result_path)

            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
        return result_paths

    def _eval_preprocess(self, result_paths):
        annotation_count = 0
        annotated_frames_count = 0
        non_annotated_frames_count = 0

        matching_results = {c: [] for c in self.cat2id.values()}

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
                reversed_score=self._reversed_score,

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

    def _eval_full_range(self, annotated_paths, non_annoated_paths):
        # annoated frames
        matchings_annotated, annotated_frames_count, non_annotated_frames_count, annotation_count = self._eval_preprocess(
            annotated_paths)

        print("=" * 40)
        print("Eval on annotated frames ({}) with {} annotations".format(
            annotated_frames_count, annotation_count))
        result_annotated = self._metric_pipeline_annotated.evaluate(
            matchings_annotated)
        self._metric_pipeline_annotated.print_results(result_annotated)

        # non annotated frames
        matchings_non_annotated, annotated_frames_count, non_annotated_frames_count, annotation_count = self._eval_preprocess(
            non_annoated_paths)
        print("=" * 40)
        print("Eval on non annotated frames ({})".format(
            non_annotated_frames_count))
        result_non_annotated = self._metric_pipeline_non_annoated.evaluate(
            matchings_non_annotated)
        self._metric_pipeline_non_annoated.print_results(result_non_annotated)

    def _eval_distance_intervals(self, annotated_paths, non_annoated_paths):
        multi_distance_metric_annoated = MultiDistanceMetric(
            self.cat2id.values(),
            self._metric_pipeline_annotated,
            distance_intervals=self._dist_eval_intervals,
            similarity_measure=self._similarity_measure,
            reversed_score=self._reversed_score,
            matcher=self._matcher,
            additional_filter_pipeline=None,
        )
        multi_distance_metric_non_annoated = MultiDistanceMetric(
            self.cat2id.values(),
            self._metric_pipeline_non_annoated,
            distance_intervals=self._dist_eval_intervals,
            similarity_measure=self._similarity_measure,
            reversed_score=self._reversed_score,
            matcher=self._matcher,
            additional_filter_pipeline=None,
        )

        interval_results_annotated = multi_distance_metric_non_annoated.evaluate(
            annotated_paths)

        MultiDistanceMetric.print_results(
            interval_results_annotated, '/workspace/work_dirs/plots')

        interval_results_non_annotated = multi_distance_metric_non_annoated.evaluate(
            non_annoated_paths)

        MultiDistanceMetric.print_results(
            interval_results_non_annotated, '/workspace/work_dirs/plots2')

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

    parser.add_argument(
        '--out', help='output result file in pickle format', default=None)

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
