import torch
from mmcv.cnn import ConvModule, xavier_init
from torch import nn as nn
from torch._C import dtype
from torch.nn import functional as F

from ..registry import FUSION_LAYERS


def point_sample(
    img_features,
    points,
    lidar2img_rt,
    pcd_rotate_mat,
    img_scale_factor,
    img_crop_offset,
    pcd_trans_factor,
    pcd_scale_factor,
    pcd_flip,
    img_flip,
    img_pad_shape,
    img_shape,
    aligned=True,
    padding_mode="zeros",
    align_corners=True,
):
    """Obtain image features using points.

    Args:
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        lidar2img_rt (torch.Tensor): 4x4 transformation matrix.
        pcd_rotate_mat (torch.Tensor): 3x3 rotation matrix of points
            during augmentation.
        img_scale_factor (torch.Tensor): Scale factor with shape of \
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop \
            image during data augmentation with shape of (w_offset, h_offset).
        pcd_trans_factor ([type]): Translation of points in augmentation.
        pcd_scale_factor (float): Scale factor of points during.
            data augmentation
        pcd_flip (bool): Whether the points are flipped.
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """
    # aug order: flip -> trans -> scale -> rot
    # The transformation follows the augmentation order in data pipeline
    if pcd_flip:
        # if the points are flipped, flip them back first
        points[:, 1] = -points[:, 1]

    points -= pcd_trans_factor
    # the points should be scaled to the original scale in velo coordinate
    points /= pcd_scale_factor
    # the points should be rotated back
    # pcd_rotate_mat @ pcd_rotate_mat.inverse() is not exactly an identity
    # matrix, use angle to create the inverse rot matrix neither.
    points = points @ pcd_rotate_mat.inverse()

    # project points from velo coordinate to camera coordinate
    num_points = points.shape[0]
    pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)
    pts_2d = pts_4d @ lidar2img_rt.t()

    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate

    pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        coor_x = orig_w - coor_x

    h, w = img_pad_shape
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1

    # print("points =", points.shape)
    # print("coors x =", coor_x.shape)
    # print("coors y =", coor_y.shape)

    grid = (
        torch.cat([coor_x, coor_y], dim=1).unsqueeze(0).unsqueeze(0)
    )  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = "bilinear" if aligned else "nearest"
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )  # 1xCx1xN feats

    return point_features.squeeze().t()


@FUSION_LAYERS.register_module()
class PointFusionMultiView(nn.Module):
    """Fuse image features from multi-scale features.

    Args:
        img_channels (list[int] | int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (int): Channels of point features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (int, optional): Number of image levels. Defaults to 3.
        conv_cfg (dict, optional): Dict config of conv layers of middle
            layers. Defaults to None.
        norm_cfg (dict, optional): Dict config of norm layers of middle
            layers. Defaults to None.
        act_cfg (dict, optional): Dict config of activatation layers.
            Defaults to None.
        activate_out (bool, optional): Whether to apply relu activation
            to output features. Defaults to True.
        fuse_out (bool, optional): Whether apply conv layer to the fused
            features. Defaults to False.
        dropout_ratio (int, float, optional): Dropout ratio of image
            features to prevent overfitting. Defaults to 0.
        aligned (bool, optional): Whether apply aligned feature fusion.
            Defaults to True.
        align_corners (bool, optional): Whether to align corner when
            sampling features according to points. Defaults to True.
        padding_mode (str, optional): Mode used to pad the features of
            points that do not have corresponding image features.
            Defaults to 'zeros'.
        lateral_conv (bool, optional): Whether to apply lateral convs
            to image features. Defaults to True.
    """

    def __init__(
        self,
        img_channels,
        pts_channels,
        mid_channels,
        out_channels,
        img_levels=3,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        activate_out=True,
        fuse_out=False,
        dropout_ratio=0,
        aligned=True,
        align_corners=True,
        padding_mode="zeros",
        lateral_conv=True,
    ):
        super(PointFusionMultiView, self).__init__()
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.img_levels = img_levels
        self.out_channels = out_channels
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(
                    img_channels[i],
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False,
                )
                self.lateral_convs.append(l_conv)
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels * len(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(sum(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        )

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False),
            )
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule

    def init_weights(self):
        """Initialize the weights of modules."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                xavier_init(m, distribution="uniform")

    def forward(self, img_feats, pts, pts_feats, img_metas):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.

        Returns:
            torch.Tensor: Fused features of each point.
        """

        img_pts = self.obtain_mlvl_feats(img_feats, pts, img_metas)
        img_pre_fuse = self.img_transform(img_pts)
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(pts_feats)

        fuse_out = img_pre_fuse + pts_pre_fuse
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def obtain_mlvl_feats(self, img_feats, pts, img_metas):
        """Obtain multi-level features for each point.

        Args:
            img_feats (list(list(torch.Tensor))): Multi-scale image features produced
                by image backbone in shape (B, C, H, W) for each lvl and each camera.
            pts (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Meta information for each sample in a batch.

        Returns:
            torch.Tensor: Corresponding image features of each point.
        """

        if self.lateral_convs is not None:
            img_ins = []
            for cam_idx in range(len(img_feats)):
                img_feats_c = img_feats[cam_idx]
                img_ins_c = [
                    lateral_conv(img_feats_c[i])
                    for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
                ]
                img_ins.append(img_ins_c)
            img_pts_features = self.out_channels * len(self.img_levels)
        else:
            img_ins = img_feats
            img_pts_features = 0
            # compute output_shape ([0] for first cam)
            for lvl in range(len(self.img_levels)):
                # features are cams x lvls x feats x h x w
                # output is sum of feats over levels
                img_pts_features += img_feats[0][lvl].size(1)

        # Sample multi-level features
        img_feats_per_point = []

        for i in range(len(img_metas)):
            # batch dimension
            # marks all points that have not been processed yet
            pts_to_process_mask = torch.ones(
                len(pts[i]), dtype=torch.bool, device="cuda")

            # stores img features for all points (combined for all cameras)
            img_pts_sample = torch.zeros(
                len(pts[i]), img_pts_features, device="cuda")
            for cam_idx in range(len(img_ins)):
                # for each camera
                img_pts_multi_lvl = []
                for lvl in range(len(self.img_levels)):
                    # feature map level dim
                    img_feats_lvl_c = img_ins[cam_idx][lvl]

                    img_pts_single_lvl = self.sample_single(
                        img_feats_lvl_c[i:i+1], pts[i][:, :3], img_metas[i], cam_idx)
                    img_pts_multi_lvl.append(img_pts_single_lvl)

                # concatenate multi level features
                img_pts_multi_lvl = torch.cat(img_pts_multi_lvl, dim=-1)

                # img_pts_multi_lvl is points x feats (out of image points get zero padded)
                # remove all projected points (to prevent duplicate projection)

                # true -> point lies inside the current camera image
                img_pts_mask = torch.any(img_pts_multi_lvl.bool(), dim=1)

                # remove all points that have been augmented already
                img_pts_mask = torch.logical_and(
                    img_pts_mask, pts_to_process_mask)

                # mark points as processed
                pts_to_process_mask[img_pts_mask] = False

                # store augmented points
                img_pts_sample[img_pts_mask] = img_pts_multi_lvl[img_pts_mask]
            img_feats_per_point.append(img_pts_sample)

        # concatenate all clouds -> points for all batches in one dim
        img_feats_per_point = torch.cat(img_feats_per_point, dim=0)
        return img_feats_per_point

    def sample_single(self, img_feats, pts, img_meta, cam_idx):
        """Sample features from single level image feature map.

        Args:
            img_feats (torch.Tensor): Image feature map in shape
                (N, C, H, W).
            pts (torch.Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.

        Returns:
            torch.Tensor: Single level image features of each point.
        """

        pcd_scale_factor = (
            img_meta["pcd_scale_factor"] if "pcd_scale_factor" in img_meta.keys() else 1
        )
        pcd_trans_factor = (
            pts.new_tensor(img_meta["pcd_trans"])
            if "pcd_trans" in img_meta.keys()
            else 0
        )
        pcd_rotate_mat = (
            pts.new_tensor(img_meta["pcd_rotation"])
            if "pcd_rotation" in img_meta.keys()
            else torch.eye(3).type_as(pts).to(pts.device)
        )

        img_scale_factor = (
            pts.new_tensor(img_meta["scale_factor"][:2])
            if "scale_factor" in img_meta.keys()
            else 1
        )
        pcd_flip = img_meta["pcd_flip"] if "pcd_flip" in img_meta.keys(
        ) else False
        img_flip = img_meta["flip"] if "flip" in img_meta.keys() else False
        img_crop_offset = (
            pts.new_tensor(img_meta["img_crop_offset"])
            if "img_crop_offset" in img_meta.keys()
            else 0
        )

        img_pts = point_sample(
            img_feats,
            pts,
            pts.new_tensor(img_meta["lidar2img"][cam_idx]),
            pcd_rotate_mat,
            img_scale_factor,
            img_crop_offset,
            pcd_trans_factor,
            pcd_scale_factor,
            pcd_flip=pcd_flip,
            img_flip=img_flip,
            img_pad_shape=img_meta["input_shape"][:2],
            img_shape=img_meta["img_shape"][:2],
            aligned=self.aligned,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return img_pts
