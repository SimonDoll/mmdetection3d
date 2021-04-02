from os import path as osp
from mmdet3d.core.evaluation.evaluation_3d.metrics.recall import Recall

import torch
import numpy as np
import pyquaternion
import tempfile

import mmcv
from mmdet.datasets import DATASETS
from ..core import show_result
from ..core.bbox import Box3DMode, LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset

from mmdet3d.core.evaluation.evaluation_3d.matchers import GreedyMatcher
from mmdet3d.core.evaluation.evaluation_3d.similarity_measure import CenterDistance2d, Iou
from mmdet3d.core.evaluation.evaluation_3d.metrics import AveragePrecision, MeanAveragePrecision, Recall, Precision, MetricPipeline, FalsePositivesPerFrame

from mmdet3d.core.evaluation.evaluation_3d.metrics.numeric_metric_result import NumericMetricResult


@DATASETS.register_module()
class CarlaDataset(Custom3DDataset):
    r"""Carla Dataset.

    This class serves as the API for experiments on Datasets created with Carla following the Dataset3d standard.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to False.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to False.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    CLASSES = (
        "lost_cargo"
    )

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=False,
        test_mode=False,
        with_velocity=True,
        eval_point_cloud_range=None
    ):
        self.load_interval = load_interval
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
            )

        self._with_velocity = with_velocity

        if eval_point_cloud_range is not None:
            assert isinstance(eval_point_cloud_range, list)
            assert len(eval_point_cloud_range) == 6

        self._eval_point_cloud_range = eval_point_cloud_range

        # setup the metric pipeline for evaluation
        # we use class ids for matching, cat2id can be used to assign a category name later on
        self.matcher = GreedyMatcher(self.cat2id.values())
        # similarity_meassure = Iou()
        self.similarity_meassure = Iou()
        # if centerpoint dist reverse matching order (lower is better)
        self.similarity_reversed_score = False

        similarity_threshold = 0.5
        # metrics
        avg_precision_metric = AveragePrecision(
            similarity_threshold=similarity_threshold, reversed_score=self.similarity_reversed_score)
        mean_avg_precision_metric = MeanAveragePrecision(
            [0.3, 0.5, 0.7, 0.9], reversed_score=self.similarity_reversed_score)
        precision_metric = Precision(similarity_threshold)
        recall_metric = Recall(similarity_threshold)

        self.metric_pipeline_annotated = MetricPipeline(
            [avg_precision_metric, mean_avg_precision_metric, recall_metric, precision_metric])

        fppf_metric = FalsePositivesPerFrame(
            similarity_threshold, reversed_score=self.similarity_reversed_score)
        self.metric_pipeline_non_annoated = MetricPipeline([fppf_metric])

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]

        gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)

        # data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

        data_infos = list(data["infos"])

        data_infos = data_infos[:: self.load_interval]

        return data_infos

    def info_to_input_dict(self, info):
        # get all info from an info dict and convert its relevant keys to the input dict
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            timestamp=info["timestamp"] / 1e6,
        )

        if self.modality["use_camera"]:
            image_paths = []
            cam_names = []
            img_T_lidar_list = []
            for cam_name, cam_info in info["cams"].items():

                image_paths.append(cam_info["data_path"])
                cam_names.append(cam_name)
                # obtain lidar to image transformation matrix
                lidar_T_cam = np.eye(4)
                lidar_T_cam[:3, :3] = cam_info["lidar_R_sensor"]
                lidar_T_cam[:3, 3] = cam_info["lidar_t_sensor"]

                # invert to gain the lidar to cam transformation
                cam_T_lidar = np.linalg.inv(lidar_T_cam)

                # camera intrinsics and projection matrix
                intrinsic = cam_info["cam_intrinsic"]
                K = np.eye(4)
                K[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

                img_T_lidar = K @ cam_T_lidar
                # imgs are undistorted normally you would use the pin hole camera model here ...

                img_T_lidar_list.append(img_T_lidar)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    img_T_lidar=img_T_lidar_list,
                    camera_names=cam_names
                )
            )
        return input_dict

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # process the prev frames first
        prev_input_dicts = []
        for i in range(len(info["prev"])):
            prev = info["prev"][i]
            prev_input_dict = self.info_to_input_dict(prev)
            # add the tfs current_T_prev
            prev_input_dict["lidar_current_t_lidar_prev"] = prev[
                "lidar_current_t_lidar_prev"
            ]
            prev_input_dict["lidar_current_R_lidar_prev"] = prev[
                "lidar_current_R_lidar_prev"
            ]

            prev_input_dicts.append(prev_input_dict)

        # now the current frame
        input_dict = self.info_to_input_dict(info)

        # add the prev infos
        input_dict["prev"] = prev_input_dicts

        # add the annotations
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]

        gt_bboxes_3d = info["gt_boxes"]
        if self._with_velocity:
            gt_velocity = info["gt_velocity"]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_names_3d = info["gt_names"]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(
                0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, gt_names=gt_names_3d
        )
        return anns_results

    def _gt_range_filter(self, gt_annos):
        filtered_gt_annos = []
        for frame_annos in gt_annos:
            boxes = frame_annos['gt_bboxes_3d']

            # bev range is x_min, x_max, y_min, y_max
            bev_range = [self._eval_point_cloud_range[0], self._eval_point_cloud_range[1],
                         self._eval_point_cloud_range[3], self._eval_point_cloud_range[4]]

            boxes_mask = boxes.in_range_bev(bev_range)

            boxes = boxes[boxes_mask]

            boxes_mask = boxes_mask.numpy()

            labels = frame_annos['gt_labels_3d'][boxes_mask]

            names = frame_annos['gt_names'][boxes_mask]

            filtered_frame_annos = dict(
                gt_bboxes_3d=boxes, gt_labels_3d=labels, gt_names=names)

            filtered_gt_annos.append(filtered_frame_annos)
        return filtered_gt_annos

    def eval_preprocess(self, results):
        # the results are in the order of the data_infos, but do not contain annos as the eval hook
        # for the carla datasets no "fixed test set exists".
        # Therefore the boxes are available (different to nuscenes etc)
        # ->  we load the annos from file (in the order of the predictions)

        # collect the gt data
        gt_annos = [self.get_ann_info(i) for i in range(len(results))]

        if self._eval_point_cloud_range is not None:
            # filter out of range bboxes
            gt_annos = self._gt_range_filter(gt_annos)

        matching_results_with_annotations = {
            c: [] for c in self.cat2id.values()}
        matching_results_no_annotations = {c: [] for c in self.cat2id.values()}

        annotation_count = 0
        annotated_frames_count = 0
        non_annotated_frames_count = 0
        for i, res in enumerate(results):

            pred_boxes = res['boxes_3d']
            pred_scores = res['scores_3d']
            pred_labels = res['labels_3d']

            gt_boxes = gt_annos[i]['gt_bboxes_3d']
            gt_labels = gt_annos[i]['gt_labels_3d']

            # gt labels are a numpy array -> bring to torch
            gt_labels = torch.from_numpy(gt_labels)

            if len(gt_labels) == 0:
                non_annotated_frames_count += 1
            else:
                annotated_frames_count += 1
            annotation_count += len(gt_labels)

            # TODO load input data aswell
            # calculate the similarity for the boxes
            similarity_scores = self.similarity_meassure.calc_scores(
                gt_boxes, pred_boxes, gt_labels, pred_labels
            )

            # match gt and predictions
            single_matching_result = self.matcher.match(
                similarity_scores,
                gt_boxes,
                pred_boxes,
                gt_labels,
                pred_labels,
                pred_scores,
                i,
                reversed_score=self.similarity_reversed_score,
            )

            for c in single_matching_result.keys():
                if len(gt_labels) == 0:
                    # no annotations for this frame
                    matching_results_no_annotations[c].extend(
                        single_matching_result[c])
                else:
                    matching_results_with_annotations[c].extend(
                        single_matching_result[c])

        return matching_results_with_annotations, annotated_frames_count, annotation_count, matching_results_no_annotations, non_annotated_frames_count

    def evaluate_sinlge(self, results):

        assert isinstance(results, list)
        assert len(results) == len(
            self), "Result length does not fit dataset length!"
        matching_results_with_annotations, annotated_frames_count, annotation_count, matching_results_no_annotations, non_annotated_frames_count = self.eval_preprocess(
            results)

        # evaluate metrics on the annotated frames
        metric_results_annotated = self.metric_pipeline_annotated.evaluate(
            matching_results_with_annotations, data=None)

        metric_results_non_annoated = self.metric_pipeline_non_annoated.evaluate(
            matching_results_no_annotations, data=None)

        print("Results on all {} annotated frames, annotations: {} ".format(
            annotated_frames_count, annotation_count))
        self.metric_pipeline_annotated.print_results(metric_results_annotated)

        # evaluate metrics on the non annotated frames
        print("Results on non annotated frames {}".format(
            non_annotated_frames_count))
        self.metric_pipeline_non_annoated.print_results(
            metric_results_non_annoated)

        if self._eval_point_cloud_range:
            print("Used eval range (bev, centers): {}".format(
                self._eval_point_cloud_range))

        results_numeric_logging = {}
        # for now only collect single numeric results of annotated frames
        for metric_name, metric_return in metric_results_annotated.items():
            if isinstance(metric_return, NumericMetricResult):
                results_numeric_logging[metric_name] = float(metric_return())
        return results_numeric_logging

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
    ):

        print("result[0] =", results[0])

        # the results can either be [{bboxes, labels, scores},..]
        # or nested: [{pts_bbox: {bboxes,..}}]

        is_nested = all(name in results[0] for name in result_names)

        if is_nested:
            results_logging = {}
            for nested_key in result_names:
                print("Evaluating boxes of {}".format(nested_key))
                # collect the results for this branch e.g. pts branch
                results_nested = list(
                    map(lambda res: res[nested_key], results))
                results_current_logging = self.evaluate_sinlge(results_nested)
                # logger does not support nested dicts -> append key with its result branch
                for k, v in results_current_logging.items():
                    results_logging[nested_key + "_" + k] = v

        else:
            results_logging = self.evaluate_sinlge(results)

        return results_logging

    def show(self, results, out_dir):
        raise NotImplementedError
