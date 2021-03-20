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

        # imgs are loaded
        # h  x w x channels x cameras
        # -> channels [0:3] are rgb
        results['img'] = results['img'][:, :, 0:3, :]
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
        results['img'] = results['img'][:, :, :, 0]
        results['img_shape'] = results['img_shape'][0:3]
        results['ori_shape'] = results['img_shape']
        results['pad_shape'] = results['img_shape']
        results['img_filename'] = results['img_filename'][0]
        results['img_fields'] = ['img']

        # scale factor should not be in keys
        results.pop('scale_factor')

        results['img_T_lidar'] = results['img_T_lidar'][0]
        results['lidar2img'] = results['img_T_lidar']

        return results
