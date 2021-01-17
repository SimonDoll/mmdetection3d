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
            self._file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
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
        for i in range(len(prev_list)):
            pointcloud = self._load_points(prev_list[i]["pts_filename"])
            prev_list[i]["points"] = pointcloud
