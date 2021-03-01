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
    root_path, info_prefix, version, max_prev_samples=10
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
    print(root_path)
    set_folder = root_path.joinpath(version)
    assert set_folder.is_dir(), "dataset has no set folder for {}".format(version)

    # load the trainset
    loader = DatasetLoader(set_folder)
    loader.setup()


    sample_infos = _fill_scene_infos(loader, max_prev_samples, lidar_name="lidar_top", ego_pose_sensor_name="imu_perfect", camera_names=["cam_front"])

    exit(0)

    metadata = dict(version=version)
    if test:
        print("test sample: {}".format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(
            root_path, "{}_infos_test.pkl".format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print(
            "train sample: {}, val sample: {}".format(
                len(train_nusc_infos), len(val_nusc_infos)
            )
        )
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(
            root_path, "{}_infos_train.pkl".format(info_prefix))
        mmcv.dump(data, info_path)
        data["infos"] = val_nusc_infos
        info_val_path = osp.join(
            root_path, "{}_infos_val.pkl".format(info_prefix))
        mmcv.dump(data, info_val_path)



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

        sample_infos = _collect_sample_data_infos(sample, loader, max_prev_samples, lidar_name, ego_pose_sensor_name, camera_names)

        current_frame_info = sample_infos[0]
        prev_frames_infos = sample_infos[1:]

        # add annotations, (only needed for the current frame the previous ones are for data only)
        current_frame_info = sample_infos[0]
        # obtain annotations
        annotations = [loader.get_annotation(token) for token in sample.annotation_tokens]

        # extract the bounding boxes
        # boxes are xyz_wlh_yaw -> 7
        gt_boxes = np.empty((len(annotations), 7))
        names = []
        for i, annotation in enumerate(annotations):
            xyz_wlh_yaw = loading_utils.obb_to_xyz_sizes_yaw_vector(annotation.transform, annotation.size)

            # we need to convert rot to SECOND format.
            # TODO source?
            xyz_wlh_yaw[6] = -xyz_wlh_yaw[6] - np.pi / 2

            gt_boxes[i] = xyz_wlh_yaw
            category_name = category_token_to_name[annotation.category_token]
            names.append(category_name)

        names = np.array(names)

        assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)}, {len(annotations)}"
        current_frame_info["gt_boxes"] = gt_boxes
        current_frame_info["gt_names"] = names
        # for now we do not add obb dynamics
        current_frame_info["gt_velocity"] = None

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
    lidar_sensor, lidar_calib = loading_utils.load_sensor_with_calib(data_loader, lidar_name)
    ego_pose_sensor, ego_pose_calib = loading_utils.load_sensor_with_calib(data_loader, ego_pose_sensor_name)

    camera_sensors, camera_calibs = loading_utils.load_sensors_with_calibs(data_loader, camera_names)
    

    # sample list is the list of the last x frames (including current)
    # sample list[0] = current ... sample_list[-1] = oldest
    # sample_infos follows the same convention
    sample_infos = []
    for sample in sample_list:
        # get the sensor_data
        lidar_sensor_data = loading_utils.load_sensor_data(data_loader, sample, lidar_sensor)
        ego_pose_sensor_data = loading_utils.load_sensor_data(data_loader, sample, ego_pose_sensor)

        cameras_sensor_data = loading_utils.load_sensors_data(data_loader, sample, camera_sensors, require_all=True)

        ego_t_lidar, ego_R_lidar = _transform_to_translation_rotation(lidar_calib.transform)

        # get the ego pose
        ego_T_ego_pose_sensor = ego_pose_calib.transform
        global_T_ego_pose_sensor = _load_ego_pose(data_loader, ego_pose_sensor_data.file)
        global_T_ego = np.dot(global_T_ego_pose_sensor, np.linalg.inv(ego_T_ego_pose_sensor))

        global_t_ego, global_R_ego = _transform_to_translation_rotation(global_T_ego)
        info = {
            "lidar_path": lidar_sensor_data.file,
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
            camera_info['data_path'] = data_loader.dataset_root.joinpath(cameras_sensor_data[camera_name].file)
            camera_info['type'] = camera_name

            ego_t_cam, ego_R_cam = _transform_to_translation_rotation(camera_calibs[camera_name].transform)

            # data is from the same sample so the ego position is the same
            lidar_t_cam, lidar_R_cam = get_sensor_relative_to(ego_t_cam, ego_R_cam, ego_t_lidar, ego_R_lidar, global_t_ego, global_R_ego, global_t_ego, global_R_ego)
        
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
        

        lidar_current_t_lidar_prev, lidar_current_R_lidar_prev = get_sensor_relative_to(ego_t_lidar_prev, ego_R_lidar_prev, ego_t_lidar_current, ego_R_lidar_current, global_t_ego_prev, global_R_ego_prev, global_t_ego_current, global_R_ego_current)


        # extract only the transform information (the rest is set already)
        # TODO maybe rework the interface to avoid this second stage here
        prev["lidar_current_t_lidar_prev"] = lidar_current_t_lidar_prev
        prev["lidar_current_R_lidar_prev"] = lidar_current_R_lidar_prev
        # update the sample infos
        sample_infos[i] = prev

    return sample_infos


def get_sensor_relative_to(ego_t_sensor_source, ego_R_sensor_source, ego_t_sensor_target, ego_R_sensor_target, world_t_ego_source, world_R_ego_source, world_t_ego_target, world_R_ego_target):

    ego_T_sensor_source = _transform_from_translation_rotation(ego_t_sensor_source, ego_R_sensor_source)
    world_T_ego_source = _transform_from_translation_rotation(world_t_ego_source, world_R_ego_source)

    ego_T_sensor_target = _transform_from_translation_rotation(ego_t_sensor_target, ego_R_sensor_target)
    world_T_ego_target = _transform_from_translation_rotation(world_t_ego_target, world_R_ego_target)

    world_T_source = np.dot(world_T_ego_source, ego_T_sensor_source)
    world_T_target = np.dot(world_T_ego_target, ego_T_sensor_target)

    target_T_source = np.dot(np.linalg.inv(world_T_target), world_T_source)
    return _transform_to_translation_rotation(target_T_source)

def transform_sensor_to_lidar_top(
    nusc,
    sensor_token,
    ego_t_lidar,
    ego_R_lidar,
    global_t_ego,
    global_R_ego,
    sensor_type="lidar",
):
    """Obtain the info with RT matrix from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        ego_t_lidar (np.ndarray): Translation from lidar to ego in shape (1, 3) (at time of lidar)
        ego_R_lidar (np.ndarray): Rotation matrix from lidar to ego (at time of lidar).
            in shape (3, 3).
        global_t_ego (np.ndarray): Translation from ego to global in shape (1, 3) (at time of lidar).
        global_R_ego (np.ndarray): Rotation matrix from ego to global (at time of lidar).
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sensor_data_transformed (dict): Sensor data information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor",
                         sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))

    # theese transforms are at time of sensor e.g. camera
    ego_sensor_t_sensor = cs_record["translation"]
    ego_sensor_R_sensor = Quaternion(cs_record["rotation"]).rotation_matrix

    global_sensor_t_ego = pose_record["translation"]
    global_sensor_R_ego = Quaternion(pose_record["rotation"]).rotation_matrix

    sensor_data_transformed = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "ego_t_sensor": ego_sensor_t_sensor,
        "ego_R_sensor": ego_sensor_R_sensor,
        "global_t_ego": global_sensor_t_ego,
        "global_R_ego": global_sensor_R_ego,
        "timestamp": sd_rec["timestamp"],
    }

    # timestamp of sensor_type
    # {global,t_sensor}_T_sensor
    ego_T_sensor = np.eye(4)
    ego_T_sensor[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
    ego_T_sensor[:3, 3] = cs_record["translation"]

    global_sensor_T_ego = np.eye(4)
    global_sensor_T_ego[:3, :3] = Quaternion(
        pose_record["rotation"]).rotation_matrix
    global_sensor_T_ego[:3, 3] = pose_record["translation"]

    # global_T_sensor = global_sensor_T_ego @ ego_T_sensor

    # lidar is currently reference
    # transformation at lidar time
    ego_T_lidar = np.eye(4)
    ego_T_lidar[:3, :3] = ego_R_lidar
    ego_T_lidar[:3, 3] = ego_t_lidar

    global_lidar_T_ego = np.eye(4)
    global_lidar_T_ego[:3, :3] = global_R_ego
    global_lidar_T_ego[:3, 3] = global_t_ego

    # lidar_T_camera with correction using global frame discrepancy
    # lidar_T_sensor = lidar_T_ego . ego_T_global . global_T_ego . ego_T_sensor
    lidar_T_sensor = np.linalg.inv(ego_T_lidar) @ (
        np.linalg.inv(
            global_lidar_T_ego) @ (global_sensor_T_ego @ ego_T_sensor)
    )

    sensor_data_transformed["lidar_R_sensor"] = lidar_T_sensor[:3, :3]
    sensor_data_transformed["lidar_t_sensor"] = lidar_T_sensor[:3, 3]
    return sensor_data_transformed


def export_2d_annotation(root_path, info_path, version):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
    """
    # get bbox annotations for camera
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    nusc_infos = mmcv.load(info_path)["infos"]
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=carla_categories.index(cat_name), name=cat_name)
        for cat_name in carla_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info["cams"][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info["sample_data_token"],
                visibilities=["", "1", "2", "3", "4"],
            )
            (height, width, _) = mmcv.imread(cam_info["data_path"]).shape
            coco_2d_dict["images"].append(
                dict(
                    file_name=cam_info["data_path"],
                    id=cam_info["sample_data_token"],
                    width=width,
                    height=height,
                )
            )
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info["segmentation"] = []
                coco_info["id"] = coco_ann_id
                coco_2d_dict["annotations"].append(coco_info)
                coco_ann_id += 1
    mmcv.dump(coco_2d_dict, f"{info_path[:-4]}.coco.json")


def get_2d_boxes(
    nusc, sample_data_token: str, visibilities: List[str]
) -> List[OrderedDict]:
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token: Sample data token belonging to a camera keyframe.
        visibilities: Visibility filter.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get("sample_data", sample_data_token)

    assert sd_rec["sensor_modality"] == "camera", (
        "Error: get_2d_boxes only works" " for camera sample_data!"
    )
    if not sd_rec["is_key_frame"]:
        raise ValueError(
            "The 2D re-projections are available only for keyframes.")

    s_rec = nusc.get("sample", sd_rec["sample_token"])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    camera_intrinsic = np.array(cs_rec["camera_intrinsic"])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get("sample_annotation", token)
                for token in s_rec["anns"]]
    ann_recs = [
        ann_rec for ann_rec in ann_recs if (ann_rec["visibility_token"] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec["sample_annotation_token"] = ann_rec["token"]
        ann_rec["sample_data_token"] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec["token"])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec["translation"]))
        box.rotate(Quaternion(pose_rec["rotation"]).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec["translation"]))
        box.rotate(Quaternion(cs_rec["rotation"]).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = (
            view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
        )

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(
            ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec["filename"]
        )
        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords]
        )

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(
    ann_rec: dict,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    sample_data_token: str,
    filename: str,
) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec["sample_data_token"] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        "attribute_tokens",
        "category_name",
        "instance_token",
        "next",
        "num_lidar_pts",
        "num_radar_pts",
        "prev",
        "sample_annotation_token",
        "sample_data_token",
        "visibility_token",
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec["bbox_corners"] = [x1, y1, x2, y2]
    repro_rec["filename"] = filename

    coco_rec["file_name"] = filename
    coco_rec["image_id"] = sample_data_token
    coco_rec["area"] = (y2 - y1) * (x2 - x1)

    if repro_rec["category_name"] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec["category_name"]]
    coco_rec["category_name"] = cat_name
    coco_rec["category_id"] = carla_categories.index(cat_name)
    coco_rec["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec["iscrowd"] = 0

    return coco_rec
