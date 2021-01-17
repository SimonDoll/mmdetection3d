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

        points_dim = results["points"].points_dim
        points = results["points"].tensor
        device = points.device

        lidar2imgs = (
            torch.tensor(
                results["lidar2img"],
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
        points_colors = torch.zeros((len(points), imgs.shape[-1]), dtype=imgs.dtype)

        # only valid at colored_points_mask
        point_colors = torch.zeros((len(points), imgs.shape[-1]))

        # def print_tensor(x):
        #     return (
        #         "shape = "
        #         + str(x.shape)
        #         + ", dtype = "
        #         + str(x.dtype)
        #         + ", device = "
        #         + str(x.device)
        #     )

        # print("points:", print_tensor(points))
        # print("colored points mask:", print_tensor(colored_points_mask))
        # print("points_colors:", print_tensor(points_colors))
        # print("imgs:", print_tensor(imgs))

        # print("point_colors:", print_tensor(point_colors))

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

            projected_points_colors = img[img_row_idxs, img_col_idxs]

            # TODO how to handle overlapping images?
            points_colors[valid_points_mask] = projected_points_colors
            colored_points_mask[valid_points_mask] = True

        # augment the points with the colors
        valid_point_colors = point_colors[colored_points_mask]
        valid_points = results["points"].tensor[colored_points_mask]

        points = torch.cat((valid_points, valid_point_colors), dim=1)
        points_class = get_points_type(self._coord_type)
        # TODO attributes
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        # print("new dim =", points.points_dim)
        results["points"] = points

        return results


# def __call__(self, results):

#         lidar2imgs = results["lidar2img"]

#         # channels x h x w x cameras
#         imgs = results["img"]
#         # cameras x h x w x channels
#         reshaped = []
#         for i in range(imgs.shape[-1]):
#             img = imgs[:, :, :, i]
#             reshaped.append(img)
#         reshaped = np.asarray(reshaped)
#         imgs = reshaped

#         points = results["points"].tensor

#         points_dim = results["points"].points_dim
#         # print("init dim =", points_dim)

#         # marks points that have a valid color value
#         points_mask = torch.zeros((len(points),), dtype=torch.bool)

#         # only valid at points_mask
#         point_colors = torch.zeros((len(points), imgs.shape[-1]))
#         for img_idx in range(len(lidar2imgs)):
#             img_mat = lidar2imgs[img_idx]

#             img_mat = torch.Tensor(img_mat)

#             img = imgs[img_idx]

#             xs = []
#             ys = []
#             zs = []
#             for p_idx in range(len(points)):
#                 point4d = torch.cat((points[p_idx][0:3], torch.tensor([1])))

#                 # project the point onto the image plane
#                 point_projected = img_mat @ point4d

#                 x = point_projected[0]
#                 y = point_projected[1]
#                 z = point_projected[2]
#                 x /= z
#                 y /= z

#                 if z < 1.0:
#                     # skip close points
#                     continue

#                 # only use points that lie inside this image
#                 if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:

#                     # grab the image color value (bgr or rgb)
#                     x = int(x.item())
#                     y = int(y.item())

#                     xs.append(x)
#                     ys.append(y)
#                     zs.append(z)
#                     color = img[y][x]

#                     # mark this index as valid
#                     # TODO check if already set, should not happen? img overlap?
#                     points_mask[p_idx] = True
#                     point_colors[p_idx] = torch.from_numpy(color)

#             plt.imshow(img, zorder=1)
#             plt.scatter(xs, ys, zorder=2, s=0.4, c=zs)

#             plt.savefig("/workspace/work_dirs/plot" + str(img_idx) + ".png")
#             plt.clf()

#         # augment the points with the img features
#         valid_point_colors = point_colors[points_mask]
#         valid_points = points[points_mask]

#         points = torch.cat((valid_points, valid_point_colors), dim=1)
#         points_class = get_points_type(self._coord_type)
#         # TODO attributes
#         points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
#         # print("new dim =", points.points_dim)
#         results["points"] = points

#         return results
