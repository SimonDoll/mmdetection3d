import numpy as np
import torch
from matplotlib import pyplot as plt


import mmcv

from mmdet.datasets import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type


@PIPELINES.register_module()
class LoadPrevPointsFromFile:
    def __init__(
        self,
        filter_non_matched=True,
        coord_type="LIDAR",
        max_sweeps=10,
        load_dim=5,
        use_dim=[0, 1, 2],
        shift_height=False,
        file_client_args=dict(backend="disk"),
    ):
        """
        This module only loads the pointclouds of previous samples from disk
        Args:
            filter_non_matched (bool) wether to remove points that do not lie in the images. Defaults to True
        """
        self._filter_non_matched = filter_non_matched
        self._coord_type = coord_type
        self._max_sweeps = max_sweeps

        self.shift_height = shift_height
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self._load_dim = load_dim
        self._use_dim = use_dim

        self._file_client_args = file_client_args.copy()
        self._file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self._file_client is None:
            self._file_client = mmcv.FileClient(**self._file_client_args)
        try:
            pts_bytes = self._file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        points = points.reshape(-1, self._load_dim)
        points = points[:, self._use_dim]
        attribute_dims = None
        # TODO attributes

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points, np.expand_dims(height, 1)], 1)
            attribute_dims = dict(height=3)

        points_class = get_points_type(self._coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )

        return points

    def __call__(self, results):

        prev_list = results["prev"]
        for i in range(min(len(prev_list), self._max_sweeps)):
            pointcloud = self._load_points(prev_list[i]["pts_filename"])
            prev_list[i]["points"] = pointcloud

        results["prev"] = prev_list

        return results


@PIPELINES.register_module()
class LoadPrevMultiViewImagesFromFile(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged", max_elems=10):
        self._to_float32 = to_float32
        self._color_type = color_type
        self._max_elems = max_elems

    def _load_imgs(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """

        filenames = results["img_filename"]
        img = np.stack(
            [mmcv.imread(name, self._color_type) for name in filenames], axis=-1
        )
        if self._to_float32:
            img = img.astype(np.float32)
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __call__(self, results):
        prev_list = results["prev"]
        for i in range(min(len(prev_list), self._max_elems)):
            result = self._load_imgs(prev_list[i])
            prev_list[i] = result
        results["prev"] = prev_list
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self._to_float32, self._color_type
        )


@PIPELINES.register_module()
class AccumulatePointClouds:
    def __init__(
        self,
        coord_type="LIDAR",
        max_sweeps=10,
        use_dim=[0, 1, 2],
        remove_close=False,
        pad_empty=False,
        test_mode=False,
    ):
        """
        This module merges the loaded prev and current pointclouds, use_dim can be used to use augmented point features as well.
        We add a dimension to the points with relative time diff current - prev for the net to identify the points
        """

        self._coord_type = coord_type
        self._max_sweeps = max_sweeps
        self._pad_empty_sweeps = pad_empty
        self._test_mode = test_mode

        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self._use_dim = use_dim
        self._remove_close = remove_close

    def _remove_close_points(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray): Multi-sweep point cloud arrays.
        """

        points = results["points"].tensor
        attributes = results["points"].attribute_dims

        # append a dimension to the points to set the time deltas
        # time delta for the current stamp is 0
        points = torch.cat(
            (
                points[:, self._use_dim],
                torch.zeros(len(points), 1),
            ),
            dim=1,
        )

        # Wrap the points back to the point class
        points_class = get_points_type(self._coord_type)
        if attributes:
            # points.shape[-1] is dimensionality, last dim is time delta
            # -1 for 0 based indexing
            attributes["time_delta"] = points.shape[-1] - 1

        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attributes
        )

        points_list = [self._remove_close_points(points, radius=self._remove_close)]
        ts = results["timestamp"]

        # case if only the current frame has a sweep (no previous frames)
        previous_count = len(results["prev"])
        if self._pad_empty_sweeps and previous_count == 0:
            for i in range(self.sweeps_num):
                if self._remove_close:
                    points_list.append(
                        self._remove_close_points(points, radius=self._remove_close)
                    )
                else:
                    points_list.append(points)
        else:
            # previous sweeps  exist
            if previous_count <= self._max_sweeps:
                # pick the last x sweeps
                choices = np.arange(previous_count)
            elif self._test_mode:
                # pick the last x sweeps
                choices = np.arange(self._max_sweeps)
            else:
                # randomly select sweeps
                choices = np.random.choice(
                    len(results["prev"]), self._max_sweeps, replace=False
                )
            for idx in choices:
                prev = results["prev"][idx]
                prev_points = prev["points"]
                if self._remove_close:
                    prev_points = self._remove_close(
                        prev_points, radius=self._remove_close
                    )

                prev_ts = prev["timestamp"]

                # transform the points this is done in a highly optimized manner:
                # lidar_current_T_lidar_prev
                # regualr for rotation way would be:
                # rot @ p (3x3) @ (3x1) but since
                # (AB)^T = B^T @ A^T
                # points = B^T already (n x 3)
                # transpose A (rot)
                # the result does not need to be transposed back since we want n x 3 not 3 x n

                # rotate
                prev_points[:, :3] = (
                    prev_points[:, :3] @ prev["lidar_current_R_lidar_prev"].T
                )
                # translate
                prev_points[:, :3] += prev["lidar_current_t_lidar_prev"]

                # add time diff (for the net to know which points belong to which sweep)
                deltas = torch.full(
                    (len(prev_points), 1), ts - prev_ts, device=prev_points.device
                )
                print("before \n", prev_points[0:5])
                prev_points = torch.cat((prev_points, deltas))
                print("after \n", prev_points[0:5])
                exit(0)
                prev_points = points.new_point(prev_points)
                points_list.append(prev_points)

        points = points.cat(points_list)
        points = points[:, self.use_dim]
        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"
