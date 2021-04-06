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
        self,
        filter_non_matched=True,
        coord_type="LIDAR",
        use_dim=[0, 1, 2, 3],
        filter_close=0.5,
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

        if filter_close:
            assert filter_close >= 0.0
        self._filter_close = filter_close

        self.count = 0

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
        points_colors = torch.zeros(
            (len(points), imgs.shape[-1]), dtype=imgs.dtype)

        # make points a row vector n x 4 x 1
        # (enables us to use batch matrix multiplication)
        # points = torch.unsqueeze(points, dim=2)

        for img_idx in range(len(lidar2imgs)):
            img_mat = lidar2imgs[img_idx]
            img = imgs[img_idx]

            # transform all points on the img plane of the currently selected img

            # use the theorem:
            # if points would have been 4 x 1 (single point)
            # (Img_mat . Points)^T = Points^T . Img_mat^T
            # (4 x 4 . 4 x 1)^T = (1 x 4) . (4 x 4) -> 1 x 4
            # this can be generalized for n points
            # points is n x 4 already (no need to transpose)
            projected_points = points @ img_mat.T

            # normalize the projected coordinates
            depth = projected_points[:, 2]
            # add an epsilon to prevent division by zero
            depth[depth == 0.0] = torch.finfo(torch.float32).eps

            projected_points[:, 0] = torch.div(projected_points[:, 0], depth)
            projected_points[:, 1] = torch.div(projected_points[:, 1], depth)

            # create a mask of valid points
            # valid means that the points lie inside the image x y borders, z is not filtered here
            mask_x = torch.logical_and(
                projected_points[:,
                                 0] > 0, projected_points[:, 0] < img.shape[0]
            )
            mask_y = torch.logical_and(
                projected_points[:,
                                 1] > 0, projected_points[:, 1] < img.shape[1]
            )

            if self._filter_close:
                valid_points_mask = torch.logical_and(mask_x, mask_y)

                mask_z = projected_points[:, 2] > self._filter_close
                valid_points_mask = torch.logical_and(
                    valid_points_mask, mask_z)

            else:
                valid_points_mask = torch.logical_and(mask_x, mask_y)

            # use only the points inside the image
            projected_points = projected_points[valid_points_mask]
            # get x y as pixel indices
            img_row_idxs = projected_points[:, 0].long()
            img_col_idxs = projected_points[:, 1].long()

            # print("here")
            self._debug_visualize(
                img,
                projected_points[:, 0].long(),
                projected_points[:, 1].long(),
                projected_points[:, 2],
                img_idx,
            )

            projected_points_colors = img[img_row_idxs, img_col_idxs]

            # TODO how to handle overlapping images?
            points_colors[valid_points_mask] = projected_points_colors
            colored_points_mask[valid_points_mask] = True

        # augment the points with the colors
        valid_point_colors = points_colors[colored_points_mask]
        valid_points = results["points"].tensor[colored_points_mask]
        valid_points = valid_points[:, self._use_dim]

        points = torch.cat((valid_points, valid_point_colors), dim=1)
        points_class = get_points_type(self._coord_type)
        # TODO attributes
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)
        # print("new dim =", points.points_dim)
        results["points"] = points
        return results

    def _debug_visualize(self, img, xs, ys, zs, idx):
        xs = xs.cpu().detach().numpy()
        ys = ys.cpu().detach().numpy()
        zs = zs.cpu().detach().numpy()

        img = img.cpu().detach().numpy()
        img = np.swapaxes(img, 0, 1)

        plt.imshow(img, zorder=1)
        plt.scatter(xs, ys, zorder=2, s=0.4, c=zs)

        self.count += 1
        plt.savefig("/workspace/work_dirs/plot/" + str(self.count) + ".png")
        plt.clf()


@PIPELINES.register_module()
class AugmentPrevPointsWithImageFeatures:
    """Adds image rgb features to the prev pointclouds"""

    def __init__(
        self,
        filter_non_matched=True,
        coord_type="LIDAR",
        max_sweeps=10,
        filter_close=1.0,
        use_dim=[0, 1, 2, 3],
    ):
        """
        Args:
            filter_non_matched (bool) wether to remove points that do not lie in the images. Defaults to True
        """
        self._filter_non_matched = filter_non_matched
        self._coord_type = coord_type
        self._max_sweeps = max_sweeps

        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self._use_dim = use_dim

        if filter_close:
            assert filter_close >= 0.0
        self._filter_close = filter_close

    def __call__(self, results):
        prev_list = results["prev"]
        for i in range(min(len(prev_list), self._max_sweeps)):
            # augment all prev frames
            prev = prev_list[i]
            projected_points = self._add_img_features_to_cloud(
                prev["points"], prev["img_T_lidar"], prev["img"]
            )
            prev_list[i]["points"] = projected_points

            results["prev"] = prev_list
        return results

    def _add_img_features_to_cloud(self, point_cloud, img_T_lidar_list, imgs):

        points = point_cloud.tensor
        device = points.device
        lidar2imgs = (
            torch.tensor(
                img_T_lidar_list,
            )
            .to(device)
            .float()
        )

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
        points_colors = torch.zeros(
            (len(points), imgs.shape[-1]), dtype=imgs.dtype)

        # make points a row vector n x 4 x 1
        # (enables us to use batch matrix multiplication)
        # points = torch.unsqueeze(points, dim=2)

        for img_idx in range(len(lidar2imgs)):
            img_mat = lidar2imgs[img_idx]
            img = imgs[img_idx]

            # transform all points on the img plane of the currently selected img

            # use the theorem:
            # if points would have been 4 x 1 (single point)
            # (Img_mat . Points)^T = Points^T . Img_mat^T
            # (4 x 4 . 4 x 1)^T = (1 x 4) . (4 x 4) -> 1 x 4
            # this can be generalized for n points
            # points is n x 4 already (no need to transpose)
            projected_points = points @ img_mat.T

            # normalize the projected coordinates
            depth = projected_points[:, 2]
            # add an epsilon to prevent division by zero
            depth[depth == 0.0] = torch.finfo(torch.float32).eps

            projected_points[:, 0] = torch.div(projected_points[:, 0], depth)
            projected_points[:, 1] = torch.div(projected_points[:, 1], depth)

            # create a mask of valid points
            # valid means that the points lie inside the image x y borders, z is not filtered here
            mask_x = torch.logical_and(
                projected_points[:,
                                 0] > 0, projected_points[:, 0] < img.shape[0]
            )
            mask_y = torch.logical_and(
                projected_points[:,
                                 1] > 0, projected_points[:, 1] < img.shape[1]
            )

            if self._filter_close:
                valid_points_mask = torch.logical_and(mask_x, mask_y)
                mask_z = projected_points[:, 2] > self._filter_close
                valid_points_mask = torch.logical_and(
                    valid_points_mask, mask_z)

            else:
                valid_points_mask = torch.logical_and(mask_x, mask_y)

            # use only the points inside the image
            projected_points = projected_points[valid_points_mask]
            # get x y as pixel indices
            img_row_idxs = projected_points[:, 0].long()
            img_col_idxs = projected_points[:, 1].long()

            # self._debug_visualize(
            #     img,
            #     projected_points[:, 0].long(),
            #     projected_points[:, 1].long(),
            #     projected_points[:, 2],
            #     img_idx,
            # )

            projected_points_colors = img[img_row_idxs, img_col_idxs]

            # TODO how to handle overlapping images?
            points_colors[valid_points_mask] = projected_points_colors
            colored_points_mask[valid_points_mask] = True

        # augment the points with the colors
        valid_point_colors = points_colors[colored_points_mask]
        valid_points = point_cloud.tensor[colored_points_mask]

        valid_points = valid_points[:, self._use_dim]

        points = torch.cat((valid_points, valid_point_colors), dim=1)
        points_class = get_points_type(self._coord_type)
        # TODO attributes
        point_cloud = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None
        )

        return point_cloud

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
