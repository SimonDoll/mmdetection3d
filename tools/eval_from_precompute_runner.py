import argparse
import pathlib
import logging
import json
import re
from functools import cmp_to_key

import numpy as np
import pandas as pd
from tabulate import tabulate
import tqdm

from tools.eval_from_precompute import EvalPipeline

# use seaborn style plots
import seaborn as sns
sns.set()


class EvalFromPrecomputeRunner:
    _precompute_file = "result_paths.json"

    _datasets = ["val", "test", "easy_test"]

    _plot_metric_iou = "AP@0.5"
    _plot_metric_cd = "AP@0.1"

    def __init__(self, eval_base_dir, out_folder) -> None:
        self._eval_base_dir = pathlib.Path(eval_base_dir)

        assert self._eval_base_dir.is_dir(), "eval base dir not found"

        self._out_base = pathlib.Path(out_folder)

        assert self._out_base.is_dir(), "output folder does not exist"

    def _run_single(self, precompute_folder, out_file):
        eval_pipeline = EvalPipeline(
            precompute_folder, out_file, verbose=False)

        result = eval_pipeline.run_eval()
        return result

    def _eval_results_to_df(self, eval_results_single, method_name):

        full_range = eval_results_single['full']
        interval = eval_results_single['interval']

        full_range_df = pd.DataFrame(full_range, index=[0])
        interval_df = pd.DataFrame(interval, index=[0])

        combined_df = full_range_df.join(interval_df)

        # TODO refactor in metric pipeline
        # insert linebreaks to long header names
        def two_line_name(name, break_str=" r: ["):
            if break_str in name:
                loc = name.find(break_str)
                name_two_line = name[0:loc] + "\n" + name[loc:]
                return name_two_line
            else:
                return name

        def remove_map_interval(name, map_str="mAP@[", end_str="]"):
            if map_str in name:
                start_loc = name.find(map_str)
                end_loc = name.find(end_str, start_loc) + 1

                if end_loc <= start_loc:
                    raise ValueError("malformed name {}".format(name))

                short_map = name[:start_loc] + "mAP" + name[end_loc:]
                return short_map
            else:
                return name

        # combined_df.rename(columns=two_line_name, inplace=True)
        combined_df.rename(columns=remove_map_interval, inplace=True)

        combined_df.insert(0, "method", [method_name], allow_duplicates=False)

        return combined_df

    def _collect_precompute_cfg_paths(self):

        config_paths = list(self._eval_base_dir.rglob('*.eval_config'))

        return config_paths

    def run(self,):
        # # find the config .json files assumes ony .json for configs
        precompute_config_files = self._collect_precompute_cfg_paths()
        logging.info("Found {} precompute config files".format(
            len(precompute_config_files)))

        logging.info("precompute configs: {}".format(list(
            map(lambda x: str(x.stem), precompute_config_files))))

        # init result folders
        dataset_out_folders = {}
        for dataset_name in self._datasets:
            dataset_out_folder = self._out_base.joinpath(dataset_name)
            dataset_out_folder.mkdir(exist_ok=True)
            dataset_out_folders[dataset_name] = dataset_out_folder

        eval_results = {dataset_name: {} for dataset_name in self._datasets}
        eval_results_dicts = {dataset_name: {}
                              for dataset_name in self._datasets}

        for precompute_cfg_path in precompute_config_files:
            # we use the config name as result name
            method_name = precompute_cfg_path.stem

            # check if precomputes are available
            base_path = precompute_cfg_path.parent.absolute().resolve()

            precompute_dirs = [x for x in base_path.glob('./*') if x.is_dir()]

            i = 0
            for dir in tqdm.tqdm(precompute_dirs):

                dataset_name = dir.name

                if dataset_name not in self._datasets:
                    logging.warning(
                        "unknown dataset folder {}, skipping".format(dataset_name))
                    continue

                # check if precompute is available
                precompute_file = dir.joinpath(self._precompute_file)

                if not precompute_file.is_file():
                    logging.warning("no precomputes for {} in {}, skipping".format(
                        method_name, base_path))
                    continue

                dataset_out_folder = dataset_out_folders[dataset_name]
                out_file_name = "{}_{}.json".format(dataset_name, method_name)

                out_file = dataset_out_folders[dataset_name].joinpath(
                    out_file_name)

                results_single = None
                if out_file.is_file():
                    # result exist already,  load
                    with open(out_file) as fp:
                        results_single = json.load(fp)
                    logging.info(
                        "result exists, loading from {}".format(out_file))
                else:
                    results_single = self._run_single(dir, out_file)

                for similarity_measure, eval_res_single in tqdm.tqdm(results_single.items()):
                    # create dict for this eval type
                    if not similarity_measure in eval_results[dataset_name]:
                        eval_results[dataset_name][similarity_measure] = {}
                        eval_results_dicts[dataset_name][similarity_measure] = {
                        }
                    # convert results to dataframe
                    eval_res_single_df = self._eval_results_to_df(
                        eval_res_single, method_name)

                    # store results
                    eval_results_dicts[dataset_name][similarity_measure][method_name] = eval_res_single
                    eval_results[dataset_name][similarity_measure][method_name] = eval_res_single_df

        # now store all results
        for dataset, dataset_res in eval_results.items():
            for sim_measure, methods in dataset_res.items():
                # combine methods
                full_df = None
                for method_name, df in methods.items():
                    if full_df is None:
                        full_df = df
                    else:
                        full_df = full_df.append(df)

                plot_metric = None
                if sim_measure == "IoU":
                    plot_metric = self._plot_metric_iou
                elif sim_measure == "CenterDistance":
                    plot_metric = self._plot_metric_cd
                else:
                    logging.warning(
                        "ignoring unknown sim measure {} for plotting".format(sim_measure))

                if plot_metric:
                    self.plot_interval(
                        full_df, dataset, sim_measure, self._out_base, plot_metric)

                # round to 4 decimals
                full_df = full_df.round(decimals=4)
                print("{} {} \n".format(dataset, sim_measure), full_df)

                # save df
                df_file_name_csv = "{}_{}.csv".format(
                    dataset, sim_measure)
                df_file_csv = self._out_base.joinpath(df_file_name_csv)
                full_df.to_csv(df_file_csv, encoding='utf-8', index=False)

                df_file_name_md = "{}_{}.md".format(
                    dataset, sim_measure)

                df_file_md = self._out_base.joinpath(df_file_name_md)

                with open(df_file_md, 'w') as outputfile:
                    table = tabulate(full_df.values,
                                     full_df.columns, tablefmt="pipe")
                    outputfile.write(table)

    # def _plot_single(self, dataset, sim_measure, method_results, metric_identifier="AP@0.5"):
    #     # for each method extract the interval values
    #     interval_results = []
    #     interval_starts = []
    #     interval_ends = []

    #     filtered_method_results = []

    #     for method, results in method_results.items():
    #         interval_res = results['interval']
    #         intervals = []
    #         for full_metric_name, metric_result in interval_res.items():
    #             match_name = re.search("([^\s]+)", full_metric_name)
    #             metric_name = match_name.group()

    #             if metric_name == metric_identifier:

    #             filtered_method_results.append({
    #                 'method':  method,
    #                 'interval': full_metric_name,
    #                 'value': metric_result
    #             })

    #     res_df = pd.DataFrame(filtered_method_results)
    #     print(res_df)

    # def _plot_single(self, dataset, sim_measure, method_results, metric_identifier="AP@0.5"):
    #     # for each method extract the interval values
    #     interval_results = []
    #     interval_starts = []
    #     interval_ends = []

    #     filtered_method_results = {}

    #     for method, results in method_results.items():
    #         interval_res = results['interval']
    #         for full_metric_name, metric_result in interval_res.items():
    #             # match the interval start in each metric name
    #             match_interval_start = re.search(
    #                 'r: \[(\d+)', full_metric_name)
    #             match_interval_end = re.search(
    #                 "r: \[\d+,(\d+)", full_metric_name)
    #             match_name = re.search("([^\s]+)", full_metric_name)

    #             interval_end = match_interval_end.group(1)
    #             interval_start = match_interval_start.group(1)

    #             metric_name = match_name.group()
    #             if metric_name == metric_identifier:
    #                 if not method in filtered_method_results:
    #                     filtered_method_results[method] = []
    #                 filtered_method_results[method].append({'start': float(interval_start),
    #                                                         'end': float(interval_end),
    #                                                         'val': float(metric_result)})

    #     def sort_by_interval(e1, e2):
    #         if e1['start'] < e2['start']:
    #             return -1
    #         elif e1['start'] > e2['start']:
    #             return 1
    #         else:
    #             return 0

    #     # sort by interval start border and check that all intervals are the same

    #     method_names = []
    #     interval_borders = None
    #     amount_intervals = None
    #     method_plot_vals = []
    #     for method in filtered_method_results:
    #         filtered_method_results[method].sort(
    #             key=cmp_to_key(sort_by_interval))

    #         if amount_intervals is None:
    #             amount_intervals = len(filtered_method_results[method])

    #         assert amount_intervals == len(filtered_method_results[method])

    #         method_vals = list(
    #             map(lambda e: e['val'], filtered_method_results[method]))

    #         interval_borders = list(
    #             map(lambda e: e['start'], filtered_method_results[method]))

    #         method_plot_vals.append(method_vals)
    #         method_names.append(method)

    #     plt.p

    def plot_interval(self, results_df, dataset, sim_measure, out_root, metric_name="AP@0.5"):
        full_metric_name = metric_name + " r:"
        column_name_regex = full_metric_name + "|method"
        print(results_df)
        # only take desired metric for all methods
        # methods x intervals
        results_df = results_df.filter(regex=column_name_regex)

        # beatuify the names by removing the method name
        def remove_metric_name(name):
            return name.replace(full_metric_name, "")
        results_df = results_df.rename(columns=remove_metric_name)

        results_df = results_df.set_index('method')
        # prevent index to be shown in legend
        results_df.index.name = None

        results_df = results_df.T

        ax = results_df.plot()

        ax.set_xlabel("Range interval")
        ax.set_ylabel(metric_name)

        ax.set_title(dataset + " " + sim_measure, fontsize=20)

        fig = ax.get_figure()

        out_root = pathlib.Path(out_root)
        out_path = out_root.joinpath(
            "interval_plot_{}_{}.pdf".format(dataset, sim_measure))

        fig.savefig(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Utility to run a series of eval precomputes")

    parser.add_argument("eval_checkpoints", type=str,
                        help="Folder containing all eval checkpoints")

    parser.add_argument("out", type=str, help="folder to store outputs in")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    runner = EvalFromPrecomputeRunner(args.eval_checkpoints, args.out)
    runner.run()
