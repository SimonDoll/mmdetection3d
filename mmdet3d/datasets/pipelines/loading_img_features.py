import numpy as np
import torch
from matplotlib import pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa

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


@PIPELINES.register_module()
class ImageAugmentationPipeline:
    """Color based image augmentation pipeline.
        Applies blur (gaus, median), shapren, contrast, grayscale  overlay, hsv colorspace"""

    def __init__(
            self, sometimes_prob=0.5, someof_range=(0, 3)):

        def sometimes(aug): return iaa.Sometimes(sometimes_prob, aug)
        # define the sequence of augmentation strageties

        self._pipeline = sometimes(
            iaa.SomeOf(someof_range,
                       [
                           # converts to HSV
                           # alters Hue in range -50,50°
                           # multiplies saturation
                           # converts back
                           iaa.WithHueAndSaturation([
                               iaa.WithChannels(0, iaa.Add((-50, 50))),
                               iaa.WithChannels(1, [
                                   iaa.Multiply((0.5, 1.5)),
                               ]),
                           ]),
                           # Sharpen each image, overlay the result with the original
                           # image using an alpha between 0 (no sharpening) and 1
                           # (full sharpening effect).
                           iaa.Sharpen(alpha=(0, 1.0),
                                       lightness=(0.75, 1.5)),

                           # Improve or worsen the contrast of images.
                           iaa.LinearContrast(
                               (0.5, 1.5), per_channel=0.5),

                           # Either drop randomly 1 to 10% of all pixels (i.e. set
                           # them to black) or drop them on an image with 2-5% percent
                           # of the original size, leading to large dropped
                           # rectangles.
                           # otherwise apply gaussian blur
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03, 0.15), size_percent=(0.02, 0.05),
                                   per_channel=0.2
                               ),
                               # gaussian blur (sigma between 0 and 3.0),
                               iaa.GaussianBlur((0, 3.0)),
                           ]),

                           # Add a value of -10 to 10 to each pixel.
                           iaa.Add((-10, 10), per_channel=0.5),

                           # Convert each image to grayscale and then overlay the
                           # result with the original with random alpha. I.e. remove
                           # colors with varying strengths.
                           sometimes(iaa.Grayscale(alpha=(0.0, 1.0))),

                       ],
                       # do all of the above augmentations in random order
                       random_order=True)

        )

    def __call__(self, results):

        # we want to apply the same augmentation for each camera
        # -> generate a deterministic pipeline for this batch
        curr_pipeline = self._pipeline.to_deterministic()

        for cam_name in results['camera_names']:
            img = results[cam_name]
            augmented = curr_pipeline.augment_image(img)
            results[cam_name] = augmented

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "".format(
            self.__class__.__name__)
