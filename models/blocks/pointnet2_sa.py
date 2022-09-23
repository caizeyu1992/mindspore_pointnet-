"""PointNet2SetAbstraction"""

from mindspore import nn
from mindspore import ops
from utils.pointnet2_util import sample_and_group_all, sample_and_group, farthest_point_sample, index_points, \
    query_ball_point


class PointNet2SetAbstraction(nn.Cell):
    """
    SA_ssg  module.
    Input:
        npoint(int):points sampled in farthest point sampling.
        radius(float):search radius in local region,0~1.
        nsample(int): how many points in each local region.
        in_channel(int): Input characters of points.
        mlp(array):output size for MLP on each point.
        group_all(bool): if True choose  pointnet2_group.SampleGroup.sample_and_group_all
                    if False  choose  pointnet2_group.SampleGroup.sample_and_group

        xyz: input points position data, [B, C, N]
        points: input points data, [B, D, N]
    Return:
        new_xyz: sampled points position data, [B, C, S]
        new_points_concat: sample points feature data, [B, D', S]

    Examples:
        >> l1_xyz= Tensor(np.ones((24, 3, 512)),ms.float32)
        >> l1_points= Tensor(np.ones((24,128, 512)),ms.float32)
        >> sa2 = PointNet2SetAbstraction(npoint=128, radius=0.4, nsample=64,
                                        in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)

        >> l2_xyz, l2_points = sa2.construct(l1_xyz, l1_points)
        >> print(l2_xyz.shape, l2_points.shape)

    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNet2SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.CellList()
        self.mlp_bns = nn.CellList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.relu = ops.ReLU()
        self.transpose = ops.Transpose()
        self.reduce_max = ops.ReduceMax()

    def construct(self, xyz, points):
        """SA construct"""
        xyz = self.transpose(xyz, (0, 2, 1))
        if points is not None:
            points = self.transpose(points, (0, 2, 1))
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = self.transpose(new_points, (0, 3, 2, 1))

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.relu(bn(conv(new_points)))

        new_points = self.reduce_max(new_points, 2)
        new_xyz = self.transpose(new_xyz, (0, 2, 1))
        return new_xyz, new_points


class PointNet2SetAbstractionMsg(nn.Cell):
    """
    SA_msg  module.
    Input:
        npoint(int):points sampled in farthest point sampling.
        radius(array):search radius in local region,0~1.
        nsample(array): how many points in each local region.
        in_channel(int): Input characters of points.
        mlp_list(array):output size for MLP on each point.

        xyz: input points position data, [B, C, N]
        points: input points data, [B, D, N]
    Return:
        new_xyz: sampled points position data, [B, C, S]
        new_points_concat: sample points feature data, [B, D', S]

    Examples:
        >> xyz = Tensor(np.ones((4, 6, 1024)), ms.float32)
        >> norm = xyz[:, 3:, :]
        >> xyz = xyz[:, :3, :]
        >> sa1 = PointNet2SetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 6,
                                            [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        >> new_xyz1, new_points_concat1 = sa1.construct(xyz , norm)
        >> print(new_xyz1.shape, new_points_concat1.shape)
    """

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNet2SetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.conv_blocks = nn.CellList()
        self.bn_blocks = nn.CellList()

        for i in range(len(mlp_list)):
            convs = nn.CellList()
            bns = nn.CellList()
            last_channel = in_channel
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

        self.relu = ops.ReLU()
        self.transpose = ops.Transpose()
        self.reduce_max = ops.ReduceMax()

    def construct(self, xyz, points):
        """SA construct"""
        xyz = self.transpose(xyz, (0, 2, 1))
        if points is not None:
            points = self.transpose(points, (0, 2, 1))
        # 最远点采样
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        # 将不同半径下的点云特征保存在new_points_list
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            k = self.nsample_list[i]
            # query_ball_point函数用于寻找球形邻域中的点
            group_idx = query_ball_point(radius, k, xyz, new_xyz)
            # 按照输入的点云数据和索引返回索引的点云数据
            grouped_xyz = index_points(xyz, group_idx)
            new_poi = ops.ExpandDims()(new_xyz, 2)
            grouped_xyz -= new_poi
            if points is not None:
                grouped_points = index_points(points, group_idx)
                # 拼接点特征数据和点坐标数据
                grouped_points = ops.Concat(-1)((grouped_points, grouped_xyz))
            else:
                grouped_points = grouped_xyz

            grouped_points = self.transpose(grouped_points, (0, 3, 2, 1))
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = self.relu(bn(conv(grouped_points)))
            # 最大池化，获得局部区域的全局特征
            new_points = self.reduce_max(grouped_points, 2)
            new_points_list.append(new_points)

        new_points_concat = ops.Concat(1)(new_points_list)
        new_xyz = self.transpose(new_xyz, (0, 2, 1))
        return new_xyz, new_points_concat


"""import numpy as np
from mindspore import Tensor
import mindspore as ms
from mindspore import context
context.set_context(mode=context.GRAPH_MODE,
                            device_target="GPU",
                            max_call_depth=20480)

xyz = Tensor(np.ones((4, 6, 1024)), ms.float32)
norm = xyz[:, 3:, :]
xyz = xyz[:, :3, :]
sa1 = PointNet2SetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3 + 3, [[16, 16, 32], [32, 32, 64]])
new_xyz1, new_points_concat1 = sa1.construct(xyz , norm)
print(new_xyz1.shape, new_points_concat1.shape)"""
