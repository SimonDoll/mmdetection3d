import mmcv
import numpy as np
import pyquaternion
import tempfile
from os import path as osp

from mmdet.datasets import DATASETS
from ..core import show_result
from ..core.bbox import Box3DMode, LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset


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
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
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
            img_T_lidar_list = []
            for _, cam_info in info["cams"].items():

                image_paths.append(cam_info["data_path"])
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

        # TODO for now we always add the velocity, maybe make switchable later
        # TODO critical!!!!
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

    def _evaluate_single(
        self, result_path, logger=None, metric="bbox", result_name="pts_bbox"
    ):
        raise NotImplementedError

    def format_results(self, results, jsonfile_prefix=None):
        raise NotImplementedError

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
        raise NotImplementedError

    def show(self, results, out_dir):
        raise NotImplementedError
