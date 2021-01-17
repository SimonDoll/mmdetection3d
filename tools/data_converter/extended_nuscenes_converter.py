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


from mmdet3d.datasets import NuScenesDataset

nus_categories = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)


def create_nuscenes_infos(
    root_path, info_prefix, version="v1.0-trainval", max_prev_samples=10
):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_prev_samples (int): Max number of previous samples to include per sample.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits

    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )

    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print(
            "train scene: {}, val scene: {}".format(len(train_scenes), len(val_scenes))
        )
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_prev_samples=max_prev_samples
    )

    metadata = dict(version=version)
    if test:
        print("test sample: {}".format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, "{}_infos_test.pkl".format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print(
            "train sample: {}, val sample: {}".format(
                len(train_nusc_infos), len(val_nusc_infos)
            )
        )
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, "{}_infos_train.pkl".format(info_prefix))
        mmcv.dump(data, info_path)
        data["infos"] = val_nusc_infos
        info_val_path = osp.join(root_path, "{}_infos_val.pkl".format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print("total scene num: {}".format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(
    nusc, train_scenes, val_scenes, test=False, max_prev_samples=10
):
    """Generate the train/val infos from the raw data.
    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_prev_samples (int): Max number of previous samples to use. Default: 10.
    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmcv.track_iter_progress(nusc.sample):
        sample_infos = _collect_sample_data_infos(sample, nusc, max_prev_samples)
        current_frame_info = sample_infos[0]
        prev_frames_infos = sample_infos[1:]

        # add annotations, (only needed for the current frame the previous ones are for data only)
        current_frame_info = sample_infos[0]
        # obtain annotation
        if not test:
            # get the sample infos
            lidar_token = sample["data"]["LIDAR_TOP"]
            sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
            pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
            lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

            l2e_r = current_frame_info["lidar2ego_rotation"]
            l2e_t = current_frame_info["lidar2ego_translation"]
            e2g_r = current_frame_info["ego2global_rotation"]
            e2g_t = current_frame_info["ego2global_translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
                -1, 1
            )
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample["anns"]]
            )
            valid_flag = np.array(
                [
                    (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                    for anno in annotations
                ],
                dtype=bool,
            ).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations
            ), f"{len(gt_boxes)}, {len(annotations)}"
            current_frame_info["gt_boxes"] = gt_boxes
            current_frame_info["gt_names"] = names
            current_frame_info["gt_velocity"] = velocity.reshape(-1, 2)
            current_frame_info["num_lidar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations]
            )
            current_frame_info["num_radar_pts"] = np.array(
                [a["num_radar_pts"] for a in annotations]
            )
            current_frame_info["valid_flag"] = valid_flag

        current_frame_info["prev"] = prev_frames_infos

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(current_frame_info)
        else:
            val_nusc_infos.append(current_frame_info)

    return train_nusc_infos, val_nusc_infos


def _collect_sample_data_infos(sample, nusc, max_prev_samples):
    """Collects information about the current and the last x samples

    Args:
        sample ([nusc.sample]): current sample.
        nusc (Nuscenes devkit instance): nuscenes devkit instance to use.
        max_prev_samples (int): amount of previous samples to use.

    Returns:
        list: list of dicts with information about the samples ([0] is current,... last is oldest)
    """
    # for each sample we need the previous x samples that we want to collect data about as well
    # collect the samples (current sample is counted -> sample + max_prev_samples -1)
    sample_list = [sample]
    most_recent_sample_token = sample["token"]
    current_sample_token = sample["token"]
    for _ in range(max_prev_samples - 1):
        current_sample = nusc.get("sample", current_sample_token)
        # check if there is a sample
        if current_sample["prev"]:
            # get the prev sample
            current_sample_token = current_sample["prev"]
            sample_list.append(nusc.get("sample", current_sample_token))
        else:
            # no further previous samples
            break

    # sample list is the list of the last x frames (including current)
    # sample list[0] = current ... sample_list[-1] = oldest
    # sample_infos follows the same convention
    sample_infos = []
    for sample in sample_list:
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)

        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "cams": dict(),
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        for cam in camera_types:
            cam_token = sample["data"][cam]
            _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = transform_sensor_to_lidar_top(
                nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
            )
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info["cams"].update({cam: cam_info})

        sample_infos.append(info)
    return sample_infos


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
        ego_t_lidar (np.ndarray): Translation from lidar to ego in shape (1, 3).
        ego_R_lidar (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        global_t_ego (np.ndarray): Translation from ego to global in shape (1, 3).
        global_R_ego (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sensor_data_transformed (dict): Sensor data information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sensor_data_transformed = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }

    # timestamp of sensor_type
    # {global,t_sensor}_T_sensor
    ego_T_sensor = np.eye(4)
    ego_T_sensor[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
    ego_T_sensor[:3, 3] = cs_record["translation"]

    global_sensor_T_ego = np.eye(4)
    global_sensor_T_ego[:3, :3] = Quaternion(pose_record["rotation"]).rotation_matrix
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
        np.linalg.inv(global_lidar_T_ego) @ (global_sensor_T_ego @ ego_T_sensor)
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
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
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
        raise ValueError("The 2D re-projections are available only for keyframes.")

    s_rec = nusc.get("sample", sd_rec["sample_token"])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    camera_intrinsic = np.array(cs_rec["camera_intrinsic"])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get("sample_annotation", token) for token in s_rec["anns"]]
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
    coco_rec["category_id"] = nus_categories.index(cat_name)
    coco_rec["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec["iscrowd"] = 0

    return coco_rec
