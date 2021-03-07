import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
import pathlib
import json

from dataset_3d.data_loaders.dataset_loader import DatasetLoader
from dataset_3d.data_structures.dataset import Dataset
import dataset_3d.utils.loading_utils as loading_utils

carla_categories = (
    "lost_cargo"
)
train_set_folder_name = "train"
val_set_folder_name = "val"
test_set_folder_name = "test"


def create_carla_infos(
    root_path, info_prefix, max_prev_samples=10
):
    """Create info file of a carla generated dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_prev_samples (int): Max number of previous samples to include per sample.
            Default: 10
    """

    # check wether splits are available
    root_path = pathlib.Path(root_path)

    assert root_path.is_dir(), "dataset root {} does not exist.".format(root_path)

    lidar_name = "lidar_top"
    ego_pose_sensor_name = "imu_perfect"
    camera_names = ["cam_front"]
    # train set
    train_set = root_path.joinpath(train_set_folder_name)
    train_infos, train_meta = _run_for_set(
        train_set, max_prev_samples, lidar_name, ego_pose_sensor_name, camera_names)
    # val set
    val_set = root_path.joinpath(val_set_folder_name)
    val_infos, val_meta = _run_for_set(
        val_set, max_prev_samples, lidar_name, ego_pose_sensor_name, camera_names)
    # test set
    test_set = root_path.joinpath(test_set_folder_name)
    test_infos, test_meta = _run_for_set(
        test_set, max_prev_samples, lidar_name, ego_pose_sensor_name, camera_names)

    print("train samples: {}".format(len(train_infos)))
    data = dict(infos=train_infos, metadata=train_meta)
    info_path = osp.join(
        root_path, "{}_infos_train.pkl".format(info_prefix))
    mmcv.dump(data, info_path)

    print("val samples: {}".format(len(val_infos)))
    data = dict(infos=val_infos, metadata=val_meta)
    info_path = osp.join(
        root_path, "{}_infos_val.pkl".format(info_prefix))
    mmcv.dump(data, info_path)

    print("test samples: {}".format(len(test_infos)))
    data = dict(infos=test_infos, metadata=test_meta)
    info_path = osp.join(
        root_path, "{}_infos_test.pkl".format(info_prefix))
    mmcv.dump(data, info_path)


def _run_for_set(set_folder, max_prev_samples=10, lidar_name="lidar_top", ego_pose_sensor_name="imu_perfect", camera_names=["cam_front"]):

    loader = DatasetLoader(set_folder)
    loader.setup()
    sample_infos = _fill_scene_infos(
        loader, max_prev_samples, lidar_name, ego_pose_sensor_name, camera_names)

    return sample_infos, loader.dataset_meta


def _fill_scene_infos(loader, max_prev_samples=10, lidar_name="lidar_top", ego_pose_sensor_name="imu_perfect", camera_names=["cam_front"]):

    # create a mapping between category_name and category_token
    category_name_to_token = {}
    category_token_to_name = {}
    for cateogry_token in loader.category_tokens:
        category = loader.get_category(cateogry_token)
        category_name_to_token[category.name] = cateogry_token
        category_token_to_name[cateogry_token] = category.name

    full_frame_infos = []

    for sample_token in mmcv.track_iter_progress(loader.sample_tokens):
        sample = loader.get_sample(sample_token)

        sample_infos = _collect_sample_data_infos(
            sample, loader, max_prev_samples, lidar_name, ego_pose_sensor_name, camera_names)

        current_frame_info = sample_infos[0]
        prev_frames_infos = sample_infos[1:]

        # add annotations, (only needed for the current frame the previous ones are for data only)
        current_frame_info = sample_infos[0]
        # obtain annotations
        annotations = [loader.get_annotation(
            token) for token in sample.annotation_tokens]

        # extract the bounding boxes
        # TODO what about pitch and roll?
        # include check that those are 0.0
        # boxes are xyz_wlh_yaw -> 7
        gt_boxes = np.empty((len(annotations), 7))
        gt_velocity = np.empty((len(annotations), 2))
        names = []
        for i, annotation in enumerate(annotations):

            xyz_wlh_yaw = annotation_to_lidar(
                annotation, current_frame_info['ego_t_lidar'], current_frame_info['ego_R_lidar'], current_frame_info['global_t_ego'], current_frame_info['global_R_ego'])

            # only use velocity in the x-y plane
            velocity = annotation.velocity[0:2]
            gt_velocity[i] = velocity
            # TODO double check with rotated boxes!!
            # i think we dont need this anymore
            # we need to convert rot to SECOND format.
            # TODO source?
            # xyz_wlh_yaw[6] = -xyz_wlh_yaw[6] - np.pi / 2
            gt_boxes[i] = xyz_wlh_yaw
            category_name = category_token_to_name[annotation.category_token]
            names.append(category_name)

        names = np.array(names)

        assert len(gt_boxes) == len(
            annotations), f"{len(gt_boxes)}, {len(annotations)}"
        current_frame_info["gt_boxes"] = gt_boxes
        current_frame_info["gt_names"] = names
        # for now we do not add obb dynamics
        current_frame_info["gt_velocity"] = gt_velocity

        current_frame_info["prev"] = prev_frames_infos

        full_frame_infos.append(current_frame_info)
    return full_frame_infos


def _transform_to_translation_rotation(transform_matrix):
    translation = transform_matrix[0:3, 3]
    rotation = transform_matrix[0:3, 0:3]
    return translation, rotation


def _transform_from_translation_rotation(translation, rotation_matrix):
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 0:3] = rotation_matrix
    transform_matrix[0:3, 3] = translation

    return transform_matrix


def _load_ego_pose(data_loader, ego_pose_file):
    ego_pose_sensor_path = str(
        data_loader.dataset_root.joinpath(ego_pose_file))
    ego_pose_sensor_dict = None
    with open(ego_pose_sensor_path, "r") as fp:
        ego_pose_sensor_dict = json.load(fp)
    # we need only the transform and not the other dynamics
    return np.asarray(ego_pose_sensor_dict['transform'])


def _collect_sample_data_infos(sample, data_loader, max_prev_samples, lidar_name="lidar_top", ego_pose_sensor_name="imu_perfect", camera_names=["cam_front"]):
    """Collects information about the current and the last x samples
    This info contains tfs from camera to lidar, camera data tfs lidar ego global and a timestamp

    Args:
        sample ([nusc.sample]): current sample.
        data_loader (Dataset): dataset_3d dataloader to use.
        max_prev_samples (int): amount of previous samples to use.

    Returns:
        list: list of dicts with information about the samples ([0] is current,... last is oldest)
    """
    # for each sample we need the previous x samples that we want to collect data about as well
    # collect the samples (current sample is counted -> sample + max_prev_samples -1)
    sample_list = [sample]
    current_sample_token = sample.token
    for _ in range(max_prev_samples - 1):
        current_sample = data_loader.get_sample(current_sample_token)
        # check if there is a sample
        if current_sample.prev_token is not None:
            # get the prev sample
            current_sample_token = current_sample.prev_token
            sample_list.append(data_loader.get_sample(current_sample_token))
        else:
            # no further previous samples
            break

    # load the sensors
    lidar_sensor, lidar_calib = loading_utils.load_sensor_with_calib(
        data_loader, lidar_name)
    ego_pose_sensor, ego_pose_calib = loading_utils.load_sensor_with_calib(
        data_loader, ego_pose_sensor_name)

    camera_sensors, camera_calibs = loading_utils.load_sensors_with_calibs(
        data_loader, camera_names)

    # sample list is the list of the last x frames (including current)
    # sample list[0] = current ... sample_list[-1] = oldest
    # sample_infos follows the same convention
    sample_infos = []
    for sample in sample_list:
        # get the sensor_data
        lidar_sensor_data = loading_utils.load_sensor_data(
            data_loader, sample, lidar_sensor)
        ego_pose_sensor_data = loading_utils.load_sensor_data(
            data_loader, sample, ego_pose_sensor)

        cameras_sensor_data = loading_utils.load_sensors_data(
            data_loader, sample, camera_sensors, require_all=True)

        ego_t_lidar, ego_R_lidar = _transform_to_translation_rotation(
            lidar_calib.transform)

        # get the ego pose
        ego_T_ego_pose_sensor = ego_pose_calib.transform
        global_T_ego_pose_sensor = _load_ego_pose(
            data_loader, ego_pose_sensor_data.file)
        global_T_ego = np.dot(global_T_ego_pose_sensor,
                              np.linalg.inv(ego_T_ego_pose_sensor))

        global_t_ego, global_R_ego = _transform_to_translation_rotation(
            global_T_ego)
        info = {
            "lidar_path": str(data_loader.dataset_root.joinpath(lidar_sensor_data.file)),
            "token": sample.token,
            "cams": dict(),
            "ego_t_lidar": ego_t_lidar,
            "ego_R_lidar": ego_R_lidar,
            "global_t_ego": global_t_ego,
            "global_R_ego": global_R_ego,
            "timestamp": sample.timestamp
        }

        # add cameras info
        for camera_name in cameras_sensor_data:
            camera_info = {}
            camera_info['data_path'] = str(data_loader.dataset_root.joinpath(
                cameras_sensor_data[camera_name].file))
            camera_info['type'] = camera_name

            ego_t_cam, ego_R_cam = _transform_to_translation_rotation(
                camera_calibs[camera_name].transform)

            # data is from the same sample so the ego position is the same
            lidar_t_cam, lidar_R_cam = get_sensor_relative_to(
                ego_t_cam, ego_R_cam, ego_t_lidar, ego_R_lidar, global_t_ego, global_R_ego, global_t_ego, global_R_ego)

            camera_info["lidar_t_sensor"] = lidar_t_cam
            camera_info["lidar_R_sensor"] = lidar_R_cam
            camera_info["cam_intrinsic"] = camera_calibs[camera_name].camera_intrinsic

            info['cams'].update({camera_name: camera_info})

        sample_infos.append(info)

    # now we need to add tf information from the prev frames to the current one
    current_sample_infos = sample_infos[0]
    # get the transforms from lidar to global at current time stamp
    ego_t_lidar_current = current_sample_infos["ego_t_lidar"]
    ego_R_lidar_current = current_sample_infos["ego_R_lidar"]
    global_t_ego_current = current_sample_infos["global_t_ego"]
    global_R_ego_current = current_sample_infos["global_R_ego"]

    # this loop runs only of there are previous samples
    for i in range(len(sample_infos)):
        if i == 0:
            # the first sample is the current one -> skip
            continue

        prev = sample_infos[i]
        ego_t_lidar_prev = prev['ego_t_lidar']
        ego_R_lidar_prev = prev['ego_R_lidar']

        global_t_ego_prev = prev['global_t_ego']
        global_R_ego_prev = prev['global_R_ego']

        lidar_current_t_lidar_prev, lidar_current_R_lidar_prev = get_sensor_relative_to(
            ego_t_lidar_prev, ego_R_lidar_prev, ego_t_lidar_current, ego_R_lidar_current, global_t_ego_prev, global_R_ego_prev, global_t_ego_current, global_R_ego_current)

        # extract only the transform information (the rest is set already)
        # TODO maybe rework the interface to avoid this second stage here
        prev["lidar_current_t_lidar_prev"] = lidar_current_t_lidar_prev
        prev["lidar_current_R_lidar_prev"] = lidar_current_R_lidar_prev
        # update the sample infos
        sample_infos[i] = prev

    return sample_infos


def annotation_to_lidar(annotation, ego_t_lidar, ego_R_lidar, global_t_ego, global_R_ego):
    ego_T_lidar = _transform_from_translation_rotation(
        ego_t_lidar, ego_R_lidar)
    global_T_ego = _transform_from_translation_rotation(
        global_t_ego, global_R_ego)

    global_T_annotation = np.asarray(annotation.transform)

    global_T_lidar = np.dot(global_T_ego, ego_T_lidar)

    lidar_T_annotation = np.dot(np.linalg.inv(
        global_T_lidar), global_T_annotation)

    # convert annotation to the framework format
    xyz_lwh_yaw = loading_utils.obb_to_xyz_sizes_yaw_vector(
        lidar_T_annotation, annotation.size)

    return xyz_lwh_yaw


def get_sensor_relative_to(ego_t_sensor_source, ego_R_sensor_source, ego_t_sensor_target, ego_R_sensor_target, world_t_ego_source, world_R_ego_source, world_t_ego_target, world_R_ego_target):

    ego_T_sensor_source = _transform_from_translation_rotation(
        ego_t_sensor_source, ego_R_sensor_source)
    world_T_ego_source = _transform_from_translation_rotation(
        world_t_ego_source, world_R_ego_source)

    ego_T_sensor_target = _transform_from_translation_rotation(
        ego_t_sensor_target, ego_R_sensor_target)
    world_T_ego_target = _transform_from_translation_rotation(
        world_t_ego_target, world_R_ego_target)

    world_T_source = np.dot(world_T_ego_source, ego_T_sensor_source)
    world_T_target = np.dot(world_T_ego_target, ego_T_sensor_target)

    target_T_source = np.dot(np.linalg.inv(world_T_target), world_T_source)
    return _transform_to_translation_rotation(target_T_source)
