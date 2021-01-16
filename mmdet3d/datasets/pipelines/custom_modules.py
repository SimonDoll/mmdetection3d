import torch
import numpy as np
from matplotlib import pyplot as plt


from mmdet.datasets import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type


@PIPELINES.register_module()
class AugmentPointsWithImageFeats:
    """Adds image rgb features to a pointcloud"""

    def __init__(self, filter_non_matched=True, coord_type="LIDAR"):
        """
        Args:
            filter_non_matched (bool) wether to remove points that do not lie in the images. Defaults to True
        """
        self._filter_non_matched = filter_non_matched
        self._coord_type = coord_type

    def __call__(self, results):

        lidar2imgs = results["lidar2img"]

        # channels x h x w x cameras
        imgs = results["img"]
        # cameras x h x w x channels
        reshaped = []
        for i in range(imgs.shape[-1]):
            img = imgs[:, :, :, i]
            reshaped.append(img)
        reshaped = np.asarray(reshaped)
        imgs = reshaped

        points = results["points"].tensor

        points_dim = results["points"].points_dim
        # print("init dim =", points_dim)

        # marks points that have a valid color value
        points_mask = torch.zeros((len(points),), dtype=torch.bool)

        # only valid at points_mask
        point_colors = torch.zeros((len(points), imgs.shape[-1]))
        for img_idx in range(len(lidar2imgs)):
            img_mat = lidar2imgs[img_idx]

            img_mat = torch.Tensor(img_mat)

            img = imgs[img_idx]

            for p_idx in range(len(points)):
                point4d = torch.cat((points[p_idx][0:3], torch.tensor([1])))

                # project the point onto the image plane
                point_projected = img_mat @ point4d

                x = point_projected[0]
                y = point_projected[1]
                z = point_projected[2]
                x /= z
                y /= z

                if z < 1.0:
                    # skip close points
                    continue

                # only use points that lie inside this image
                if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:

                    # grab the image color value (bgr or rgb)
                    x = int(x)
                    y = int(y)
                    color = img[y][x]

                    # mark this index as valid
                    # TODO check if already set, should not happen? img overlap?
                    points_mask[p_idx] = True
                    point_colors[p_idx] = torch.from_numpy(color)

        # augment the points with the img features
        valid_point_colors = point_colors[points_mask]
        valid_points = points[points_mask]

        points = torch.cat((valid_points, valid_point_colors), dim=1)
        points_class = get_points_type(self._coord_type)
        # TODO attributes
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        # print("new dim =", points.points_dim)
        results["points"] = points

        return results
