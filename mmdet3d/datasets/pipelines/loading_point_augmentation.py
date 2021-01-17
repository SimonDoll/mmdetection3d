import numpy as np
import torch
from matplotlib import pyplot as plt


import mmcv

from mmdet.datasets import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type


@PIPELINES.register_module()
class AugmentPointsWithImageFeatures:
    """Adds image rgb features to a pointcloud"""

    def __init__(
        self, filter_non_matched=True, coord_type="LIDAR", use_dim=[0, 1, 2, 3]
    ):
        """
        Args:
            filter_non_matched (bool) wether to remove points that do not lie in the images. Defaults to True
        """
        self._filter_non_matched = filter_non_matched
        self._coord_type = coord_type

        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self._use_dim = use_dim

    def __call__(self, results):
        points = results["points"].tensor
        device = points.device
        lidar2imgs = (
            torch.tensor(
                results["img_T_lidar"],
            )
            .to(device)
            .float()
        )

        print("lidar 2 imgs =", lidar2imgs)
        exit(0)

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

            self._debug_visualize(
                img, img_row_idxs, img_col_idxs, projected_points[:, 2], img_idx
            )

            projected_points_colors = img[img_row_idxs, img_col_idxs]

            # TODO how to handle overlapping images?
            points_colors[valid_points_mask] = projected_points_colors
            colored_points_mask[valid_points_mask] = True

        # augment the points with the colors
        valid_point_colors = point_colors[colored_points_mask]
        valid_points = results["points"].tensor[colored_points_mask]
        valid_points = valid_points[:, self._use_dim]

        points = torch.cat((valid_points, valid_point_colors), dim=1)
        points_class = get_points_type(self._coord_type)
        # TODO attributes
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        # print("new dim =", points.points_dim)
        results["points"] = points

        print("done")
        return results

    def _debug_visualize(self, img, xs, ys, zs, idx):
        xs = xs.cpu().detach().numpy()
        ys = ys.cpu().detach().numpy()
        zs = zs.cpu().detach().numpy()

        img = img.cpu().detach().numpy()
        img = np.swapaxes(img, 0, 1)

        plt.imshow(img, zorder=1)
        plt.scatter(xs, ys, zorder=2, s=0.4, c=zs)

        plt.savefig("/workspace/work_dirs/plot" + str(idx) + ".png")
        plt.clf()


@PIPELINES.register_module()
class AugmentPrevPointsWithImageFeatures:
    """Adds image rgb features to the prev pointclouds"""

    def __init__(
        self,
        filter_non_matched=True,
        coord_type="LIDAR",
        max_sweeps=10,
    ):
        """
        Args:
            filter_non_matched (bool) wether to remove points that do not lie in the images. Defaults to True
        """
        self._filter_non_matched = filter_non_matched
        self._coord_type = coord_type
        self._max_sweeps = max_sweeps

    def __call__(self, results):
        # augment the current frame

        current_pointcloud = results["points"]
        print(results.keys())
        exit()

        # augment all prev frames
        pass

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
