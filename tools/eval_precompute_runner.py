import argparse
import pathlib
import logging
import json

from tools.eval_precompute import EvalPrecompute


class EvalPrecomputeRunner:

    def __init__(self, eval_base_dir) -> None:
        self._eval_base_dir = pathlib.Path(eval_base_dir)

    def _run_single(self, config_file, checkpoint, out_dir_base):

        out_dir_test = out_dir_base.joinpath("test")
        if out_dir_test.is_dir():
            logging.warning(
                "Precompute {} exists already, skipping".format(out_dir_test))
        else:
            out_dir_test.mkdir()
            logging.info("Running precompute for {}, mode={}, out={}".format(
                config_file.name, "test", out_dir_test))

            # dont initalize each time (will raise error)
            eval_precompute_test = EvalPrecompute(
                str(config_file), str(checkpoint), out_dir_test, "test", seed=42, deterministic=True, initialize=False)

            eval_precompute_test.run()

        out_dir_val = out_dir_base.joinpath("val")

        if out_dir_val.is_dir():
            logging.warning(
                "Precompute {} exists already, skipping".format(out_dir_val))
        else:
            out_dir_val.mkdir()

            logging.info("Running precompute for {}, mode={}, out={}".format(
                config_file.name, "val", out_dir_val))
            # dont initalize each time (will raise error)
            eval_precompute_val = EvalPrecompute(
                str(config_file), str(checkpoint), out_dir_val, "val", seed=42, deterministic=True, initialize=False)

            eval_precompute_val.run()

    def run(self,):
        # find the config .json files assumes ony .json for configs
        precompute_config_files = list(
            self._eval_base_dir.rglob('*.eval_config'))
        logging.info("Found {} precompute config files".format(
            len(precompute_config_files)))

        # TODO critical: this is a workaround for distributed models
        import torch.distributed as dist
        dist.init_process_group(
            'gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

        for precompute_cfg_path in precompute_config_files:
            precompute_config = None
            with open(precompute_cfg_path) as fp:
                precompute_config = json.load(fp)

            base_path = precompute_cfg_path.parent.absolute().resolve()

            config_file = pathlib.Path(precompute_config['config'])
            assert config_file.is_file(), "config {}  not found".format(config_file)

            checkpoint_file = base_path.joinpath(
                precompute_config['checkpoint'])

            assert checkpoint_file.is_file(), "checkpoint {} not found".format(checkpoint_file)

            self._run_single(config_file, checkpoint_file, base_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Utility to run a series of eval precomputes")

    parser.add_argument("eval_checkpoints", type=str,
                        help="Folder containing all eval checkpoints")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    runner = EvalPrecomputeRunner(args.eval_checkpoints)
    runner.run()
