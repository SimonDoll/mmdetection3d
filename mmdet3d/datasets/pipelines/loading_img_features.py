import numpy as np
import torch
from matplotlib import pyplot as plt


import mmcv

from mmdet.datasets import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type


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

        results['img_T_lidar'] = results['img_T_lidar'][0]
        results['lidar2img'] = results['img_T_lidar']

        return results
