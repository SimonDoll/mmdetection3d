import numpy as np
import torch
from matplotlib import pyplot as plt


import mmcv

from mmdet.datasets import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type


@PIPELINES.register_module()
class AugmentPointsWithCurrentImageFeatures:
    """Adds image rgb features to a pointcloud"""

    def __init__(self, filter_non_matched=True, coord_type="LIDAR"):
        """
        Args:
            filter_non_matched (bool) wether to remove points that do not lie in the images. Defaults to True
        """
        self._filter_non_matched = filter_non_matched
        self._coord_type = coord_type

    def __call__(self, results):

        points_dim = results["points"].points_dim
        points = results["points"].tensor
        device = points.device

        lidar2imgs = (
            torch.tensor(
                results["lidar2img"],
            )
            .to(device)
            .float()
        )

        # h x w x channels x cameras
        imgs = results["img"]

        # TODO do in torch
        # move the img dimension to front
        imgs = np.moveaxis(imgs, -1, 0)

        # swap width and height
        imgs = np.swapaxes(imgs, 1, 2)

        # cameras x w x h x color channels
        imgs = torch.from_numpy(imgs).to(device)

        # get x y z only
        points = points[:, 0:3]
        # bring the points to homogenous coords
        points = torch.cat((points, torch.ones((len(points), 1))), dim=1)

        # marks points that have a valid color value
        colored_points_mask = torch.zeros((len(points),), dtype=torch.bool)

        # create the result array to store the colored points in
        # n x color channels
        points_colors = torch.zeros((len(points), imgs.shape[-1]), dtype=imgs.dtype)

        # only valid at colored_points_mask
        point_colors = torch.zeros((len(points), imgs.shape[-1]))

        # make points a row vector n x 4 x 1
        # (enables us to use batch matrix multiplication)
        points = torch.unsqueeze(points, dim=2)

        for img_idx in range(len(lidar2imgs)):
            img_mat = lidar2imgs[img_idx]
            img = imgs[img_idx]

            # transform all points on the img plane of the currently selected img
            # expand the img mat to n x 4 x 4
            img_mat = img_mat.expand((len(points), img_mat.shape[0], img_mat.shape[1]))

            # batch matrix mul n x 4 x 4 @ n x 4 x 1 -> n x 4 x 1
            projected_points = torch.bmm(img_mat, points)

            # make points a column vector -> n x 4
            projected_points = torch.squeeze(projected_points, dim=2)

            # normalize the projected coordinates
            projected_points[0] /= projected_points[2]
            projected_points[1] /= projected_points[2]

            # create a mask of valid points
            # valid means that the points lie inside the image x y borders, z is not filtered here
            mask_x = torch.logical_and(
                projected_points[:, 0] > 0, projected_points[:, 0] < img.shape[0]
            )
            mask_y = torch.logical_and(
                projected_points[:, 1] > 0, projected_points[:, 1] < img.shape[1]
            )

            valid_points_mask = torch.logical_and(mask_x, mask_y)

            # use only the points inside the image
            projected_points = projected_points[valid_points_mask]
            # get x y as pixel indices
            img_row_idxs = projected_points[:, 0].long()
            img_col_idxs = projected_points[:, 1].long()

            projected_points_colors = img[img_row_idxs, img_col_idxs]

            # TODO how to handle overlapping images?
            points_colors[valid_points_mask] = projected_points_colors
            colored_points_mask[valid_points_mask] = True

        # augment the points with the colors
        valid_point_colors = point_colors[colored_points_mask]
        valid_points = results["points"].tensor[colored_points_mask]

        points = torch.cat((valid_points, valid_point_colors), dim=1)
        points_class = get_points_type(self._coord_type)
        # TODO attributes
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        # print("new dim =", points.points_dim)
        results["points"] = points

        return results


@PIPELINES.register_module()
class AugmentPointsWithCorrespondingImageFeatures:
    """Adds image rgb features to a pointcloud"""

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
        # TODO additional args such as shift height
        """
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

        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
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
        # load the current point cloud and the current images
        pts_filename = results["pts_filename"]
        points_current = self._load_points(pts_filename)

        print("results =", results.keys())
        exit(0)
        # self._add_img_features_to_cloud(
        #     points_current,
        # )

    def _add_img_features_to_cloud(self, point_cloud, img_T_lidar_list, imgs):

        points = point_cloud.tensor
        device = points.device

        img_T_lidar_tensors = torch.tensor(img_T_lidar_list).to(device).float()

        # TODO do in torch
        # move the img dimension to front
        imgs = np.moveaxis(imgs, -1, 0)

        # swap width and height
        imgs = np.swapaxes(imgs, 1, 2)

        # cameras x w x h x color channels
        imgs = torch.from_numpy(imgs).to(device)

        # get x y z only
        points = points[:, 0:3]
        # bring the points to homogenous coords
        points = torch.cat((points, torch.ones((len(points), 1))), dim=1)

        # marks points that have a valid color value
        colored_points_mask = torch.zeros((len(points),), dtype=torch.bool)

        # create the result array to store the colored points in
        # n x color channels
        points_colors = torch.zeros((len(points), imgs.shape[-1]), dtype=imgs.dtype)

        # only valid at colored_points_mask
        point_colors = torch.zeros((len(points), imgs.shape[-1]))

        # make points a row vector n x 4 x 1
        # (enables us to use batch matrix multiplication)
        points = torch.unsqueeze(points, dim=2)

        for img_idx in range(len(img_T_lidar_tensors)):
            img_T_lidar = img_T_lidar_tensors[img_idx]
            img = imgs[img_idx]

            # transform all points on the img plane of the currently selected img
            # expand the img mat to n x 4 x 4
            img_T_lidar = img_T_lidar.expand(
                (len(points), img_T_lidar.shape[0], img_T_lidar.shape[1])
            )

            # TODO maybe switch to (AB)^T = B^T @ A^T pattern
            # batch matrix mul n x 4 x 4 @ n x 4 x 1 -> n x 4 x 1
            projected_points = torch.bmm(img_T_lidar, points)

            # make points a column vector -> n x 4
            projected_points = torch.squeeze(projected_points, dim=2)

            # normalize the projected coordinates
            projected_points[0] /= projected_points[2]
            projected_points[1] /= projected_points[2]

            # create a mask of valid points
            # valid means that the points lie inside the image x y borders, z is not filtered here
            mask_x = torch.logical_and(
                projected_points[:, 0] > 0, projected_points[:, 0] < img.shape[0]
            )
            mask_y = torch.logical_and(
                projected_points[:, 1] > 0, projected_points[:, 1] < img.shape[1]
            )

            valid_points_mask = torch.logical_and(mask_x, mask_y)

            # use only the points inside the image
            projected_points = projected_points[valid_points_mask]
            # get x y as pixel indices
            img_row_idxs = projected_points[:, 0].long()
            img_col_idxs = projected_points[:, 1].long()

            projected_points_colors = img[img_row_idxs, img_col_idxs]

            # TODO how to handle overlapping images?
            points_colors[valid_points_mask] = projected_points_colors
            colored_points_mask[valid_points_mask] = True

        # augment the points with the colors
        valid_point_colors = point_colors[colored_points_mask]
        valid_points = point_cloud.tensor[colored_points_mask]

        points = torch.cat((valid_points, valid_point_colors), dim=1)
        points_class = get_points_type(self._coord_type)
        # TODO attributes
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        # print("new dim =", points.points_dim)
        point_cloud = points

        return point_cloud
