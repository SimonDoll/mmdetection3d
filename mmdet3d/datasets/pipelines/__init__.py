from mmdet.datasets.pipelines import Compose
from .dbsampler import DataBaseSampler
from .formating import Collect3D, DefaultFormatBundle, DefaultFormatBundle3D
from .loading import (
    LoadAnnotations3D,
    LoadMultiViewImageFromFiles,
    LoadPointsFromFile,
    LoadPointsFromMultiSweeps,
    PointSegClassMapping,
)
from .test_time_aug import MultiScaleFlipAug3D
from .transforms_3d import (
    BackgroundPointsFilter,
    GlobalRotScaleTrans,
    IndoorPointSample,
    ObjectNoise,
    ObjectRangeFilter,
    ObjectSample,
    PointShuffle,
    PointsRangeFilter,
    RandomFlip3D,
    VoxelBasedPointSampler,
)

# own modules
from .loading_point_augmentation import AugmentPointsWithImageFeatures
from .loading_point_augmentation import AugmentPrevPointsWithImageFeatures

from .loading_custom import LoadPrevPointsFromFile

from .loading_img_features import ExtractFrontImageToKittiFormat, RGBA2RGB, MultiViewImagesToList, ImageAugmentationPipeline
from .loading_depth_map import PointsToDepthMap, DepthMapToPoints, SparseToDense

__all__ = [
    "ObjectSample",
    "RandomFlip3D",
    "ObjectNoise",
    "GlobalRotScaleTrans",
    "PointShuffle",
    "ObjectRangeFilter",
    "PointsRangeFilter",
    "Collect3D",
    "Compose",
    "LoadMultiViewImageFromFiles",
    "LoadPointsFromFile",
    "DefaultFormatBundle",
    "DefaultFormatBundle3D",
    "DataBaseSampler",
    "LoadAnnotations3D",
    "IndoorPointSample",
    "PointSegClassMapping",
    "MultiScaleFlipAug3D",
    "LoadPointsFromMultiSweeps",
    "BackgroundPointsFilter",
    "VoxelBasedPointSampler",
    "AugmentPointsWithImageFeatures",
    "AugmentPrevPointsWithImageFeatures",
    "LoadPrevPointsFromFile",
    "LoadPrevMultiViewImagesFromFile",
    "ExtractFrontImageToKittiFormat",
    "RGBA2RGB",
    "MultiViewImagesToList",
    "PointsToDepthMap",
    "DepthMapToPoints",
    "SparseToDense",
    "ImageAugmentationPipeline",
]
