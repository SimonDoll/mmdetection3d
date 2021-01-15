import torch
import numpy as np
from matplotlib import pyplot as plt


from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class AugmentPointsWithImageFeats:
    """Adds image rgb features to a pointcloud"""

    def __init__(self, filter_non_matched=True):
        """
        Args:
            filter_non_matched (bool) wether to remove points that do not lie in the images. Defaults to True
        """
        self._filter_non_matched = filter_non_matched

    def __call__(self, results):

        lidar2imgs = results["lidar2img"]

        # channels x h x w x cameras
        imgs = results["img"]
        print("imgs =", imgs.shape)
        # cameras x h x w x channels
        reshaped = []
        for i in range(imgs.shape[-1]):
            img = imgs[:, :, :, i]
            reshaped.append(img)
        reshaped = np.asarray(reshaped)
        print("reshaped =", reshaped.shape)
        imgs = np.moveaxis(imgs, -1, 0)
        imgs = reshaped

        points = results["points"].tensor

        points_4d = points[:, 0:4]

        print("points shape =", points.shape)
        points_dim = results["points"].points_dim

        for idx in range(len(lidar2imgs)):
            img_mat = lidar2imgs[idx]

            img_mat = torch.Tensor(img_mat)

            img = imgs[idx]
            # img mat is 4x4 projection to img plane

            print("img mat =\n", np.asarray(img_mat))

            print(points_4d.shape)

            count = 0
            xs = []
            ys = []
            zs = []
            for p in points_4d:
                point4d = torch.cat((p[0:3], torch.tensor([1])))
                point_projected = img_mat @ point4d
                # print("proj =", point_projected)
                # print("img mat =", img_mat)
                x = point_projected[0]
                y = point_projected[1]
                z = point_projected[2]
                x /= z
                y /= z

                if z < 1.0:
                    # skip close points
                    continue

                # print(x, ",", y)
                if x >= 0 and x < 1600 and y >= 0 and y < 900:
                    xs.append(x.item())
                    ys.append(y.item())
                    zs.append(z.item())
                    count += 1
                    # hits[int(y), int(x)] = np.asarray([255, 255, 255])
                    # img[int(y), int(x)] = np.asarray([255, 0, 0])

            print("count =", count)
            plt.imshow(img, zorder=1)
            plt.scatter(xs, ys, zorder=2, s=0.4, c=zs)

            plt.savefig("/workspace/work_dirs/plot" + str(idx) + ".png")
            plt.clf()

        # raise ValueError("bla")

        # exit(0)
        # print("\n")
        # print("scale factor =", results["scale_factor"])
        # print("img_norm_cfg =", results["img_norm_cfg"])
        # print("lidar2img", results["lidar2img"])
        # print("img_fields", results["img_fields"])
        # print("\n")
        # raise ValueError("bla")
        return results


# @PIPELINES.register_module()
# class AugmentPointsWithImageFeats:
#     """Adds image rgb features to a pointcloud"""

#     def __init__(self, filter_non_matched=True, sweeps=10):
#         """
#         Args:
#             filter_non_matched (bool) wether to remove points that do not lie in the images. Defaults to True
#         """
#         self._filter_non_matched = filter_non_matched
#         self._sweeps_num = sweeps

#     def __call__(self, results):
#         print(results.keys())
#         print("time =", results["timestamp"])
#         print("sweeps =", results["sweeps"])
#         # if len(results["sweeps"]) <= self._sweeps_num:
#         #         choices = np.arange(len(results["sweeps"])-1)
#         #     elif self.test_mode:
#         #         choices = np.arange(self.sweeps_num-1)
#         #     else:
#         #         choices = np.random.choice(
#         #             len(results["sweeps"]), self.sweeps_num-1, replace=False
#         #         )
#         #     # choices is list of idxs for sweeps with len sweeps_num -1
#         #     # always add the current sweep:

#         #     for idx in choices:
#         #         sweep = results["sweeps"][idx]
#         #         points_sweep = self._load_points(sweep["data_path"])
#         #         points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
#         #         if self.remove_close:
#         #             points_sweep = self._remove_close(points_sweep)
#         #         sweep_ts = sweep["timestamp"] / 1e6
#         #         points_sweep[:, :3] = (
#         #             points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
#         #         )
#         #         points_sweep[:, :3] += sweep["sensor2lidar_translation"]
#         #         points_sweep[:, 4] = ts - sweep_ts
#         #         points_sweep = points.new_point(points_sweep)
#         #         sweep_points_list.append(points_sweep)