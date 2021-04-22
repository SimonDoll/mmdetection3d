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


class EvalPrecompute:
    _result_paths_file = "result_paths.json"
    _cat_to_id = "cat2id.json"

    def __init__(self, config_file, checkpoint, out, mode, seed=42, deterministic=True):
        self._init_cfg(config_file)
        # set random seeds
        if seed is not None:
            set_random_seed(seed, deterministic=deterministic)

        self._result_base_path = pathlib.Path(out)
        logging.info("Results will be saved in: {}".format(
            self._result_base_path))

        self._intermediate_res_path = self._result_base_path.joinpath(
            "pkl_results")

        self._intermediate_res_path.mkdir()

        self._init_data(mode)
        self._init_model(checkpoint)

    def _init_cfg(self, config_file):
        self.cfg = Config.fromfile(config_file)

        # set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        self.cfg.model.pretrained = None
        self.cfg.data.val.test_mode = False  # Assure to get ground truth
        self.cfg.data.test.test_mode = False  # Assure to get ground truth

    def _init_data(self, mode):
        # build the dataloader
        samples_per_gpu = 1

        dataset = None
        if mode == "train":
            dataset = build_dataset(self.cfg.data.train)
        elif mode == "test":
            dataset = build_dataset(self.cfg.data.test)
        elif mode == "val":
            dataset = build_dataset(self.cfg.data.val)
        else:
            raise ValueError("unsupportet dataset type {}".format(mode))

        self.data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=1,
            dist=False,
            shuffle=False,
        )

        # get class information
        self.cat2id = dataset.cat2id

        cat2id_file = self._result_base_path.joinpath(self._cat_to_id)

        with open(cat2id_file, 'w') as fp:
            json.dump(self.cat2id, fp, indent=4)

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

    def _single_gpu_eval(self,):
        self.model.eval()
        dataset = self.data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))

        result_paths = []
        for i, data in enumerate(self.data_loader):

            # # TODO remove
            # if i > 10:
            #     break

            with torch.no_grad():
                annos = dataset.get_ann_info(i)

                gt_boxes = annos['gt_bboxes_3d']

                gt_labels = annos['gt_labels_3d']
                # gt labels are a numpy array -> bring to torch
                gt_labels = torch.from_numpy(gt_labels)

                result = self.model(return_loss=False, rescale=True, **data)

                # TODO critical do clean checking for nested heads and allow multi head
                is_nested = "pts_bbox" in result[0]

                if is_nested:
                    pred_boxes = result[0]['pts_bbox']['boxes_3d']
                    pred_labels = result[0]['pts_bbox']['labels_3d']
                    pred_scores = result[0]['pts_bbox']['scores_3d']

                else:
                    pred_boxes = result[0]['boxes_3d']
                    pred_labels = result[0]['labels_3d']
                    pred_scores = result[0]['scores_3d']

                result_with_gt_and_data = {
                    'gt_boxes': gt_boxes,
                    'gt_labels': gt_labels,
                    'pred_boxes': pred_boxes,
                    'pred_scores': pred_scores,
                    'pred_labels': pred_labels,
                    'data': {}
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

    def run(self):
        """Runs inference on the validation dataset specified by the model stores intermediate pkl result files

        """
        self.model = MMDataParallel(self.model, device_ids=[0])
        result_paths = self._single_gpu_eval()

        # convert pathlib paths to str
        result_paths = list(map(lambda p: str(p), result_paths))

        result_paths_file = self._result_base_path.joinpath(
            self._result_paths_file)

        with open(result_paths_file, 'w') as fp:
            json.dump(result_paths, fp, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Preprocomputes matching results for evaluation')

    parser.add_argument(
        'config_file', type=str, help='Model configuration (test?)')
    parser.add_argument(
        'checkpoint_file', type=str, help='Trained model checkpoint')

    parser.add_argument(
        'out', type=str, help="Base path for outputs")

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.',
    )

    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "val"], help="Dataset to use", default="test")

    args = parser.parse_args()

    config_file = args.config_file
    assert pathlib.Path(config_file).is_file(), "config file does not exist"

    checkpoint = args.checkpoint_file
    assert pathlib.Path(checkpoint).is_file(), "checkpoint file does not exist"

    out = args.out
    assert pathlib.Path(out).is_dir(), "output folder does not exist"

    seed = args.seed
    deterministic = args.deterministic
    mode = args.mode

    runner = EvalPrecompute(config_file, checkpoint, out,
                            mode, seed, deterministic)
    runner.run()
