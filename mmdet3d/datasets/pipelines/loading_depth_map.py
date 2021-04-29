import pathlib

import numpy as np
import torch
from matplotlib import pyplot as plt
import mmcv

from mmdet.datasets import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type

import sparse_to_dense.models
import sparse_to_dense.dataloaders.transforms as transforms


@PIPELINES.register_module()
class PointsToDepthMap:
    """Converts a pointcloud to depth maps for all used cameras"""

    def __init__(
        self,
        filter_close=0.05,
        img_idxs=[0]
    ):
        """

        Args:
            filter_non_matched(bool) wether to remove points that do not lie in the images. Defaults to True
            img_idxs (list): camera idxs to use (0 = front center in carla datasets)
        """

        if filter_close:
            assert filter_close > 0.0
        self._filter_close = filter_close

        assert isinstance(img_idxs, list)

        self._img_idxs = img_idxs

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

        # list cameras [h x w x channels]
        imgs = results["img"]

        # will be list(cameras [h x w x 1]
        depth_maps = []

        # get x y z only
        points = points[:, 0:3]

        # bring the points to homogenous coords
        points = torch.cat((points, torch.ones((len(points), 1))), dim=1)

        for img_idx in self._img_idxs:
            img_mat = lidar2imgs[img_idx]
            # img = imgs[img_idx]

            # h x w
            depth_map = torch.zeros(imgs[img_idx].shape[0:2])

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
                                 0] > 0, projected_points[:, 0] < depth_map.shape[1]
            )
            mask_y = torch.logical_and(
                projected_points[:,
                                 1] > 0, projected_points[:, 1] < depth_map.shape[0]
            )

            if self._filter_close:
                valid_points_mask = torch.logical_and(mask_x, mask_y)
                mask_z = projected_points[:, 2] > self._filter_close
                valid_points_mask = torch.logical_and(
                    valid_points_mask, mask_z)

            else:
                valid_points_mask = torch.logical_and(mask_x, mask_y)

            # keep the depth for all unset pixels at 0.0 (convention from sparse2dense)
            # use only the points inside the image
            projected_points = projected_points[valid_points_mask]

            # h x w
            depth_map[projected_points[:, 1].long(
            ), projected_points[:, 0].long()] = projected_points[:, 2]

            # add channel dimsion to depth map
            depth_map = torch.unsqueeze(depth_map, dim=-1)

            depth_maps.append(depth_map.numpy())

        results['depth_maps'] = depth_maps

        return results


@PIPELINES.register_module()
class DepthMapToPoints:
    """Converts multiple depth maps a point cloud"""

    def __init__(
        self,
        coord_type="LIDAR",
        img_idxs=[0]
    ):
        self._coord_type = coord_type

        assert isinstance(img_idxs, list)
        self._img_idxs = img_idxs

    def __call__(self, results):

        points = results["points"].tensor
        device = points.device
        img_T_lidar_tfs = (
            torch.tensor(
                results["img_T_lidar"],
            )
            .to(device)
            .float()
        )

        # list cameras [h x w x 1]
        depth_maps = results["depth_maps"]

        points = []
        for img_idx in self._img_idxs:
            img_T_lidar = img_T_lidar_tfs[img_idx]

            depth_map = depth_maps[img_idx]
            # remove extra depth channel
            depth_map = depth_map.squeeze()

            # create a point cloud from the pixels
            # get the valid pixel coordinates
            # all pixels with depth != 0 are valid

            depth_map = torch.from_numpy(depth_map)

            valid_idxs = torch.nonzero(depth_map)

            points_img_plane = torch.ones((len(valid_idxs), 3))

            valid_y_idxs = valid_idxs[:, 0]
            valid_x_idxs = valid_idxs[:, 1]

            valid_depth_values = depth_map[valid_y_idxs, valid_x_idxs]

            # create the points for back projection
            points_img_plane = torch.ones((len(valid_depth_values), 4))
            points_img_plane[:, 0] = valid_x_idxs
            points_img_plane[:, 1] = valid_y_idxs

            # for back projection we need points (u,v,1, 1/z)
            # because a point (x,y,z,1) is projected to (u,v,1,1/z)
            # store the inverted depth (valid because depth was forced to non zero values beforehand)
            points_img_plane[:, 3] = 1.0 / valid_depth_values
            lidar_T_img = torch.inverse(img_T_lidar)

            # transform all points on the img plane of the currently selected img

            # use the theorem:
            # if points would have been 4 x 1 (single point)
            # (Img_mat . Points)^T = Points^T . Img_mat^T
            # (4 x 4 . 4 x 1)^T = (1 x 4) . (4 x 4) -> 1 x 4
            # this can be generalized for n points
            # points is n x 4 already (no need to transpose)
            points_lidar = points_img_plane @ lidar_T_img.T
            points_lidar *= torch.unsqueeze(valid_depth_values, dim=-1)

            # TODO how to hanle overlapping images?
            points.append(points_lidar[:, 0:3])

        points = torch.cat(points, dim=0)

        points_class = get_points_type(self._coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)

        results['points'] = points

        return results


@PIPELINES.register_module()
class PointsToFile:
    """Stores the pointcloud to a file (used for precomputes of sparse2dense)"""

    def __init__(
        self,
        points_folder="lidar_upsampled",
        points_prefix="lidar_upsampled",
        points_prefix_to_remove="lidar_top"
    ):
        self._points_folder = points_folder
        self._points_prefix = points_prefix
        self._points_prefix_to_remove = points_prefix_to_remove

    def __call__(self, results):

        orig_path = results['pts_filename']
        # construct the file for the pseudo sensor
        orig_path = pathlib.Path(orig_path)

        # assumes sweeps/sensor_name/sensor.bla
        sensor_data_root = orig_path.parents[1]

        out_root = sensor_data_root.joinpath(self._points_folder)
        out_root.mkdir(exist_ok=True)

        points = results["points"].tensor.detach().cpu().numpy()

        # check that the input file is of known structure
        assert orig_path.stem.startswith(
            self._points_prefix_to_remove), "path to orig sensor is not of known structure {}".format(orig_path)

        out_file_name = self._points_prefix + orig_path.stem[len(
            self._points_prefix_to_remove):] + ".bin"
        out_file_path = out_root.joinpath(out_file_name)

        points.astype(np.float32).tofile(out_file_path)

        return results


@PIPELINES.register_module()
class LoadSparse2DensePrecompute:
    """Augments the lidar pointcloud with the sparse2dense precompute points"""

    def __init__(
        self,
        points_folder="lidar_upsampled",
        points_prefix="lidar_upsampled",
        points_prefix_to_remove="lidar_top"
    ):
        self._points_folder = points_folder
        self._points_prefix = points_prefix
        self._points_prefix_to_remove = points_prefix_to_remove

    def __call__(self, results):

        orig_path = results['pts_filename']
        # construct the file for the pseudo sensor
        orig_path = pathlib.Path(orig_path)

        # assumes sweeps/sensor_name/sensor.bla
        sensor_data_root = orig_path.parents[1]

        augmented_root = sensor_data_root.joinpath(self._points_folder)

        # check that the input file is of known structure
        assert orig_path.stem.startswith(
            self._points_prefix_to_remove), "path to orig sensor is not of known structure {}".format(orig_path)

        augmented_file_name = self._points_prefix + orig_path.stem[len(
            self._points_prefix_to_remove):] + ".bin"
        augmented_file_path = augmented_root.joinpath(augmented_file_name)

        # load the points back
        aug_points = np.fromfile(augmented_file_path, dtype=np.float32)
        # aug points are x y z only
        aug_points = aug_points.reshape((-1, 3))

        aug_points = torch.from_numpy(aug_points)

        # now merge with lidar cloud
        # for now we simply concatenate the clouds
        lidar_points = results['points']

        if lidar_points.shape[-1] == 3:
            # points are x y z
            lidar_points.tensor = torch.cat(
                [lidar_points.tensor, aug_points], dim=0)
        elif lidar_points.shape[-1] == 4:
            # points are x y z i
            # add a 0 for the intensity dim
            aug_points = torch.cat(
                [aug_points, torch.zeros((len(aug_points), 1))], dim=1)
            lidar_points.tensor = torch.cat(
                [lidar_points.tensor, aug_points], dim=0)
        else:
            raise ValueError("points have wrong dimensions, expected 3 or 4, got {}".format(
                lidar_points.shape))

        results['points'] = lidar_points
        return results


@PIPELINES.register_module()
class SparseToDense:
    """Upsamples the point cloud utilizing the sparse to dense approach.
    """

    def __init__(self, checkpoint_path, road_crop=(450, 0, 450, 1600), img_idxs=[0]):
        """Module to apply sparse to dense for point cloud upsampling

        Args:
            checkpoint_path (str): Path to sparse to dense model state dict
            road_crop (tuple, optional): Road crop for images, y0, x0, height, width. Defaults to (450,0,450,1600).
        """
        # h x w
        self._output_size = (road_crop[2], road_crop[3])
        self._road_crop = road_crop

        assert isinstance(img_idxs, list)
        self._img_idxs = img_idxs

        state_dict = torch.load(checkpoint_path)
        self._model = self._build_model(state_dict)

    def _build_model(self, state_dict, in_channels=4, layers=18, decoder="deconv3",):
        model = sparse_to_dense.models.ResNet(
            layers=layers, decoder=decoder, output_size=self._output_size, in_channels=in_channels)

        model = torch.nn.DataParallel(model).cuda()

        model.load_state_dict(state_dict)
        model.eval()
        return model

    def __call__(self, results):

        depth_maps = results["depth_maps"]
        imgs = results["img"]
        upsampled_depth_maps = []
        for cam_idx in self._img_idxs:

            img = imgs[cam_idx]
            depth_map = depth_maps[cam_idx]

            y0 = self._road_crop[0]
            x0 = self._road_crop[1]

            y1 = y0 + self._road_crop[2]
            x1 = x0 + self._road_crop[3]

            # store orignal image size (before crop)
            img_shape = img.shape

            # road crop
            img = img[y0:y1, x0:x1]
            depth_map = depth_map[y0:y1, x0:x1]

            # color [0,1] as rgb
            img = img[:, :, [2, 1, 0]]
            img = np.divide(img, 255.0)

            img = torch.from_numpy(img).float()
            depth_map = torch.from_numpy(depth_map).float()

            # combine to HxWx4 and add batch dimension
            rgbd = torch.cat((img, depth_map), dim=-1).unsqueeze(dim=0)
            # reshape to 1x4xHxW
            rgbd = rgbd.permute(0, 3, 1, 2)
            with torch.no_grad():
                upsampled_depth_map = self._model(rgbd)

            # reshape to default img format (h x w x 1) (drop batch dimension)
            upsampled_depth_map = upsampled_depth_map[0].detach().permute(
                1, 2, 0)

            upsampled_depth_map_full_size = torch.zeros(
                (img_shape[0], img_shape[1], 1))

            upsampled_depth_map_full_size[y0:y1, x0:x1] = upsampled_depth_map

            upsampled_depth_maps.append(
                upsampled_depth_map_full_size.cpu().numpy())

        results["depth_maps"] = upsampled_depth_maps

        return results
