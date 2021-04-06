import numpy as np
import torch
from matplotlib import pyplot as plt


import mmcv

from mmdet.datasets import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type


@PIPELINES.register_module()
class PointsToDepthMap:
    """Converts a pointcloud to depth maps for all used cameras"""

    def __init__(
        self,
        filter_close=0.5,
    ):
        """

        Args:
            filter_non_matched(bool) wether to remove points that do not lie in the images. Defaults to True
        """

        if filter_close:
            assert filter_close > 0.0
        self._filter_close = filter_close

    def __call__(self, results):
        """Added  keys: depth_maps
        The generated depth maps follow the same order / shape as the camera rgb images.
        """

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

        # imgs x h x w x 1
        depth_maps = torch.empty(
            (imgs.shape[3], imgs.shape[1], imgs.shape[0], 1,))

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

            # set the depth for all unset pixels to 0.0 (convention from sparse2dense)
            # w x h
            depth_map = torch.zeros(img.shape[0:2])
            depth_map[projected_points[:, 0].long(
            ), projected_points[:, 1].long()] = projected_points[:, 2]

            # add channel dimsion to depth map
            depth_map = torch.unsqueeze(depth_map, dim=-1)

            depth_maps[img_idx] = depth_map

        # transform depth maps to be represented similar to the camera images
        # H x W x 1 x imgs amount
        depth_maps = depth_maps.permute(2, 1, 3, 0)

        results['depth_maps'] = depth_maps

        return results

    def _debug_visualize(self, img, xs, ys, zs, idx):
        xs = xs.cpu().detach().numpy()
        ys = ys.cpu().detach().numpy()
        zs = zs.cpu().detach().numpy()

        img = img.cpu().detach().numpy()
        img = np.swapaxes(img, 0, 1)

        plt.imshow(img, zorder=1)
        plt.scatter(xs, ys, zorder=2, s=0.4, c=zs)

        plt.savefig("/workspace/work_dirs/plot/" + str(idx) + ".png")
        plt.clf()
