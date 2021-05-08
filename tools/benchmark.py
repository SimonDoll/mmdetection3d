import argparse
from random import sample
import time
import json
import pathlib

import torch
import mmcv
import tqdm
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmdet.core import wrap_fp16_model
from tools.fuse_conv_bn import fuse_conv_bn, fuse_module


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=2000, help='samples to benchmark')
    parser.add_argument(
        '--log_interval', default=50, type=int, help='interval of logging')
    parser.add_argument(
        '--fuse_conv_bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')

    parser.add_argument('--initialize', action='store_true',
                        help='wether to initialize the default process group')

    parser.add_argument('--out', type=str,
                        help="json file to export the fps to", default=None)

    args = parser.parse_args()
    return args


class RuntimeCalculator:
    def __init__(self, config, checkpoint, initialize=False, fuse_conv=False) -> None:
        if initialize:
            # TODO this allows us to use models with sync bn on non distributed envs
            import torch.distributed as dist
            dist.init_process_group(
                'gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

        self._model, self._data_loader, self._dataset = self.setup(
            config, checkpoint, fuse_conv)

    def setup(self, config_file, checkpoint_file, fuse_conv):
        cfg = Config.fromfile(config_file)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None,
                               test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, checkpoint_file, map_location='cpu')
        if fuse_conv:
            model = fuse_module(model)

        self._fuse_conv = fuse_conv

        model = MMDataParallel(model, device_ids=[0])

        model.eval()

        return model, data_loader, dataset

    def run(self, samples=2000, log_interval=0, out_file=None):

        logging = log_interval > 0

        # the first several iterations may be very slow so skip them
        num_warmup = 5
        pure_inf_time = 0

        fps = -1

        with tqdm.tqdm(total=samples) as pbar:

            # benchmark with several samples and take the average
            for i, data in enumerate(self._data_loader):
                pbar.update()
                torch.cuda.synchronize()
                start_time = time.perf_counter()

                with torch.no_grad():
                    self._model(return_loss=False, rescale=True, **data)

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time

                if i >= num_warmup:
                    pure_inf_time += elapsed
                    if logging and (i + 1) % log_interval == 0:
                        fps = (i + 1 - num_warmup) / pure_inf_time
                        print(f'Done sample [{i + 1:<3}/ {samples}], '
                              f'fps: {fps:.1f} samples / s')

                if (i + 1) == samples:
                    pure_inf_time += elapsed
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    if logging:
                        print(f'Overall fps: {fps:.1f} samples / s')
                    break

            if out_file is not None:
                out_file = pathlib.Path(out_file)
                assert out_file.parent.is_dir(), "out root does not exist"
                with open(out_file, "w") as fp:

                    fps_result = {"fps": fps,
                                  "samples": samples,
                                  "fuse_conv": self._fuse_conv}
                    json.dump(fps_result, fp)


def main():
    args = parse_args()

    calc = RuntimeCalculator(args.config, args.checkpoint,
                             args.initialize, args.fuse_conv_bn)

    calc.run(samples=args.samples,
             log_interval=args.log_interval, out_file=args.out)


if __name__ == '__main__':
    main()
