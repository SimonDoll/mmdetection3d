import argparse
import datetime
import mmcv
import os
import pickle
import shutil
import tempfile
import time
import torch
import tqdm
from lost_cargo.eval import similarity_measure
from lost_cargo.eval.filters import *
from lost_cargo.eval.matchers.greedy_matcher import GreedyMatcher
from lost_cargo.eval.metrics import *
from lost_cargo.eval.similarity_measure import *
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from os import path as osp
from tools.fuse_conv_bn import fuse_module
from torch import distributed as dist

from mmdet3d.apis import single_gpu_test
from mmdet3d.core import BboxOverlapsNearest3D, MaxIoUAssigner
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.core import encode_mask_results, wrap_fp16_model


class EvalPipeline:

    def __init__(self, args):
        self._init_cfg(args.config_file)
        self._init_processing(args)
        self._init_data()
        self._init_model(args.checkpoint_file)

    def _init_cfg(self, config_file):
        self.cfg = Config.fromfile(config_file)

        # set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        self.cfg.model.pretrained = None
        self.cfg.data.custom_val.test_mode = False  # Assure to get ground truth

    def _init_data(self):
        # build the dataloader
        # TODO right config (train / val / test?)
        samples_per_gpu = self.cfg.data.custom_val.pop('samples_per_gpu', 1)
        dataset = build_dataset(self.cfg.data.custom_val)
        self.data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=self.distributed,
            shuffle=False,
        )

        # get class information
        self.cat2id = dataset.cat2id

    def _init_model(self, checkpoint_file):
        """Initializes the detector from the config and given checkpoint.

        Args:
            checkpoint_file (str): Checkpoint file of trained model
        """
        # TODO configs?
        self.model = build_detector(
            self.cfg.model, None, test_cfg=self.cfg.test_cfg)
        fp16_cfg = self.cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)
        checkpoint = load_checkpoint(
            self.model, checkpoint_file, map_location='cpu')
        if args.fuse_conv_bn:
            self.model = fuse_module(self.model)

        # if problems with backward compatibility (see test.py of mmdetection3d for a fix)
        self.model.CLASSES = checkpoint['meta']['CLASSES']

    def _init_processing(self, args):
        """Sets up distributed processing
        Args:
            args (Namespace): program options
        """
        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            self.distributed = False
        else:
            self.distributed = True
            init_dist(args.launcher, **self.cfg.dist_params)

        # set random seeds
        if args.seed is not None:
            set_random_seed(args.seed, deterministic=args.deterministic)

    def _single_gpu_eval(self):
        # TODO implement for non distributed models
        raise NotImplementedError(
            'Run distributed with single gpu as argument')

    def _multi_gpu_eval(self, tmpdir=None):
        """Test model with multiple gpus.

        @see mmdet apis/test.py: multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False)
        Saves the results on different gpus to 'tmpdir'
        and collects them by the rank 0 worker.

        Args:
            tmpdir (str): Path of directory to save the temporary results from
                different gpus.
        Returns:
            list: The prediction results.
        """
        self.model.eval()
        results = []

        dataset = self.data_loader.dataset
        # check which process
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))

        # MMDET?
        time.sleep(2)  # This line can prevent deadlock problem in some cases.

        for i, data in enumerate(self.data_loader):

            # TODO remove
            # if i > 3:
            #     break

            with torch.no_grad():
                # TODO use data to get points in boxes etc.
                # remove gt to use the regular test mode
                gt_bboxes_3d = data.pop('gt_bboxes_3d').data
                gt_labels_3d = data.pop('gt_labels_3d').data

                gt_result = {
                    'gt_bboxes_3d': gt_bboxes_3d,
                    'gt_labels_3d': gt_labels_3d
                }

                # test format needs an additional dimension
                data['img_metas'] = [data['img_metas']]
                data['points'] = [data['points']]

                # predict
                result = self.model(return_loss=False, rescale=True, **data)

                # print("gt bboxes =", gt_bboxes_3d)
                # print("gt labels =", gt_labels_3d)
                # print("res =", result)

                # TODO check if this interface is the same for other models / datasets
                # add gt information (pred is [{'pts_bbox : pts infos}])
                result[0]['pts_bbox'].update(gt_result)

                # encode mask results
                if isinstance(result[0], tuple):
                    # TODO is this needed for our models?
                    result = [(bbox_results, encode_mask_results(mask_results))
                              for bbox_results, mask_results in result]

            # clean the data a bit to make it easier to use in eval
            # TODO check for different models
            data['points'] = data['points'][0].data[0][0]

            # store data and inference result
            combined = {'data': data, 'result': result}
            results.append(combined)

            if rank == 0:
                batch_size = len(result)
                for _ in range(batch_size * world_size):
                    prog_bar.update()

        # collect results from all ranks
        results = self.collect_results_cpu(results, len(dataset), tmpdir)
        return results

    @staticmethod
    def collect_results_cpu(result_part, size, tmpdir=None):
        """
        @taken from mmdet/apis/test.py
        copied because it is not in the official apis
        """
        rank, world_size = get_dist_info()
        # create a tmp dir if it is not specified
        if tmpdir is None:
            MAX_LEN = 512
            # 32 is whitespace
            dir_tensor = torch.full((MAX_LEN, ),
                                    32,
                                    dtype=torch.uint8,
                                    device='cuda')
            if rank == 0:
                mmcv.mkdir_or_exist('.dist_test')
                tmpdir = tempfile.mkdtemp(dir='.dist_test')
                tmpdir = torch.tensor(
                    bytearray(tmpdir.encode()),
                    dtype=torch.uint8,
                    device='cuda')
                dir_tensor[:len(tmpdir)] = tmpdir
            dist.broadcast(dir_tensor, 0)
            tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
        else:
            mmcv.mkdir_or_exist(tmpdir)
        # dump the part result to the dir
        mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
        dist.barrier()
        # collect all parts
        if rank != 0:
            return None
        else:
            # load results of all parts from tmp dir
            part_list = []
            for i in range(world_size):
                part_file = osp.join(tmpdir, f'part_{i}.pkl')
                part_list.append(mmcv.load(part_file))
            # sort the results
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
            # remove tmp dir
            shutil.rmtree(tmpdir)
            return ordered_results

    def compute_metrics(self, inference_results):
        start = datetime.datetime.now()

        # similarity_meassure = Iou()
        similarity_meassure = CenterDistance2d()

        # if centerpoint dist reverse matching order (lower is better)
        reversed_score = True

        filter_range_interval = BoxDistanceIntervalFilter(
            box_range=[0, 0, 30, 30])

        filter_points_in_box = MinPointsInGtFilter()

        filter_pipeline = FilterPipeline(
            [filter_range_interval, filter_points_in_box])

        # we use class ids for matching, cat2id can be used to assign a category name later on
        matcher = GreedyMatcher(self.cat2id.values())

        # metrics
        avg_precision_metric = AveragePrecision(
            similarity_threshold=4, reversed_score=reversed_score)
        mean_avg_precision_metric = MeanAveragePrecision(
            [0.5, 1, 2, 4], reversed_score=reversed_score)

        metric_pipeline = MetricPipeline(
            [avg_precision_metric, mean_avg_precision_metric])

        matching_results = {c: [] for c in self.cat2id.values()}

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

            # TODO remove critical debug only
            # pred_boxes = gt_boxes
            # pred_labels = gt_labels
            # pred_scores = pred_scores[0 : len(pred_labels)]

            # print("pred labels =", pred_labels)
            # print("gt labels =", gt_labels)

            # filter the boxes
            (
                gt_boxes,
                pred_boxes,
                gt_labels,
                pred_labels,
                pred_scores,
                input_data,
            ) = filter_pipeline.apply(gt_boxes, pred_boxes, gt_labels,
                                      pred_labels, pred_scores, input_data)

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
        start = datetime.datetime.now()

        # similarity_meassure = Iou()
        similarity_measure = CenterDistance2d()

        # if centerpoint dist reverse matching order (lower is better)
        reversed_score = True

        filter_points_in_box = MinPointsInGtFilter()

        filter_pipeline = FilterPipeline([filter_points_in_box])

        matcher = GreedyMatcher(self.cat2id.values())

        similarity_threshold = 4
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

        mean_avg_precision_metric = MeanAveragePrecision(
            [0.5, 1, 2, 4], reversed_score=reversed_score)

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
        if not self.distributed:
            self.model = MMDataParallel(self.model, device_ids=[0])
            results = self._single_gpu_eval(self.model, self.data_loader,
                                            args.show, args.show_dir)
        else:
            print('distributed eval')
            self.model = MMDistributedDataParallel(
                self.model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
            )
            results = self._multi_gpu_eval(tmpdir)

            self.compute_metrics2(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Custom non dataset dependent eval pipeline')

    parser.add_argument(
        'config_file', type=str, help='Model configuration (test?)')
    parser.add_argument(
        'checkpoint_file', type=str, help='Trained model checkpoint')

    # TODO output folder
    # parser.add_argument('--out', help='output result file in pickle format')

    # --------------------------------
    # MMDETECTION3D test arguments
    # --------------------------------
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed',
    )

    # TODO visualization
    # parser.add_argument('--show', action='store_true', help='show results')
    # parser.add_argument(
    #     '--show-dir', help='directory where results will be saved')

    # TODO gpu collect needed?
    # parser.add_argument(
    #     '--gpu-collect',
    #     action='store_true',
    #     help='whether to use gpu to collect results.')
    # parser.add_argument(
    #     '--tmpdir',
    #     help='tmp directory used for collecting results from multiple '
    #     'workers, available when gpu_collect is not specified')
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
