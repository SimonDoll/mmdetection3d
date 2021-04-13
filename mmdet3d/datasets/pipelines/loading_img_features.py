import numpy as np
import torch
from matplotlib import pyplot as plt


import mmcv

from mmdet.datasets import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type


@PIPELINES.register_module()
class RGBA2RGB:
    """Removes transparency dimension from images"""

    def __init__(
        self,
    ):
        pass

    def __call__(self, results):

        for key in results.get('img_fields', ['img']):
            # imgs are loaded
            # h x w x channels
            # -> channels [0:3] are rgb

            img = results[key]
            img = img[:, :, 0:3]

            results[key] = img
        # tuple to list
        img_shape = list(results['img_shape'])
        img_shape[2] = 3  # 3 channels only
        img_shape = tuple(img_shape)

        results['img_shape'] = img_shape

        # tuple to list
        ori_shape = list(results['ori_shape'])
        ori_shape[2] = 3  # 3 channels only
        ori_shape = tuple(ori_shape)

        results['ori_shape'] = ori_shape

        # tuple to list
        pad_shape = list(results['pad_shape'])
        pad_shape[2] = 3  # 3 channels only
        pad_shape = tuple(pad_shape)

        results['pad_shape'] = pad_shape

        # the img norm config also needs to be reduced to 3D
        norm_cfg = results['img_norm_cfg']
        norm_cfg['mean'] = norm_cfg['mean'][0:3]
        norm_cfg['std'] = norm_cfg['mean'][0:3]
        results['img_norm_cfg'] = norm_cfg

        return results


@PIPELINES.register_module()
class MultiViewImagesToList:
    """Collects multi view images (referenced in img_fields) to a list and adds it to resutls["img"]"""

    def __init__(
        self,
    ):
        pass

    def __call__(self, results):

        imgs = []
        for cam_name in results.get('img_fields'):

            # the order of images must be the same as the cameras (because the lidar2img transforms are in the same order as well)
            imgs.append(results[cam_name])

        results["img"] = imgs
        return results


@PIPELINES.register_module()
class NormalizeMultiSweepImages:
    """Normalizes multiple images"""

    def __init__(self, mean, std, to_rgb=True) -> None:

        self._mean = np.array(mean, dtype=np.float32)
        self._std = np.array(std, dtype=np.float32)

        # multi view images have the shape
        # H x W x 3 x imgs
        # adapt mean and std
        self._mean = np.expand_dims(self._mean, axis=(0, 1, -1))
        self._std = np.expand_dims(self._std, axis=(0, 1, -1))

        self._to_rgb = to_rgb

    def __call__(self, results):
        # the images are H x W x 3 x images amount
        imgs = results['img']

        if self._to_rgb:
            # TODO maybe create ascontiguousarray
            imgs = imgs[..., ::-1, :]

        imgs = np.divide((imgs - self._mean), self._std)

        results['img_norm_cfg'] = dict(
            mean=self._mean, std=self._std, to_rgb=self._to_rgb)
        return results


@PIPELINES.register_module()
class ExtractFrontImageToKittiFormat:
    """Adds image rgb features to a pointcloud"""

    def __init__(
        self,
    ):
        pass

    def __call__(self, results):

        # TODO critical this module is for debugging only!!!
        # for the moment simply extract the first image in the list (as it is the center for testing)
        results['img'] = results[results["camera_names"][0]]
        results['img_shape'] = results['img_shape'][0:3]
        results['ori_shape'] = results['img_shape']
        results['pad_shape'] = results['img_shape']
        results['img_filename'] = results['img_filename'][0]
        results['img_fields'] = ['img']

        results['img_T_lidar'] = results['img_T_lidar'][0]
        results['lidar2img'] = results['img_T_lidar']

        return results
