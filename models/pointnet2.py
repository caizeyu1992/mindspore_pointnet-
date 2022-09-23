"""PointNet2 Model"""

import mindspore.ops as ops
import mindspore.nn as nn
from models.blocks.pointnet2_sa import PointNet2SetAbstraction, PointNet2SetAbstractionMsg
from models.blocks.pointnet2_fp import PointNetFeaturePropagation



class Pointnet2clsModel(nn.Cell):
    """
    Constructs a PointnetNet2 architecture from
    PointnetNet2: Deep Hierarchical Feature Learning on Point Sets in a Metric Space <https://arxiv.org/abs/1706.02413>.

    Args:
        normal_channel (bool): Whether to use the channels of points' normal vector. Default: True.
        pretrained (bool): If True, returns a model pre-trained on PointnetNet2. Default: False.

    Inputs:
        - points(Tensor) - Tensor of original points. shape:[batch, channels, npoints].

    Outputs:
        Tensor of shape :[batch, 40].

    Supported Platforms:
        ``GPU``

    Examples:
        >> import numpy as np
        >> import mindspore as ms
        >> from mindspore import Tensor, context
        >> from mindvision.ms3d.models.backbones import pointnet2cls
        >> context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, max_call_depth=2000)
        >> net = Pointnet2clsModel(normal_channel=True)
        >> xyz = Tensor(np.ones((24,6, 1024)),ms.float32)
        >> output = net(xyz)
        >> print(output.shape)
        (24, 40)

    About PointNet2Cls:

    This architecture is based on PointNet2 classfication SSG,
    compared with PointNet, PointNet2_SSG added local feature extraction.

    Citation

    .. code-block::

        @article{qi2017pointnetplusplus,
          title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
          author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1706.02413},
          year={2017}
        }
    """

    def __init__(self, normal_channel=False):
        super(Pointnet2clsModel, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # 512 npoint = points sampled in farthest point sampling
        # 0.2 radius = search radius in local region
        # 32 nsample = how many points in each local region
        # [64,64,128] mlp = output size for MLP on each point
        # + 3 = xyz 3-dim coordinates
        self.sa1 = PointNet2SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],
                                           group_all=False)
        self.sa2 = PointNet2SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                           group_all=False)
        self.sa3 = PointNet2SetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                           mlp=[256, 512, 1024], group_all=True)

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

        head = []
        mid_channel = [512, 256]
        input_channel = 1024
        length = len(mid_channel)
        for i in range(length):
            head.append(nn.Dense(input_channel, mid_channel[i]))
            head.append(nn.BatchNorm1d(mid_channel[i]))
            head.append(nn.ReLU())
            head.append(nn.Dropout(0.4))
            input_channel = mid_channel[i]

        self.classifier = nn.SequentialCell(head)
        self.fc = nn.Dense(mid_channel[-1], 40)
        self.logsoftmax = nn.LogSoftmax(axis=1)

    def construct(self, xyz):
        """construct method"""
        xyz = self.transpose(xyz, (0, 2, 1))
        if self.normal_channel:
            norm = xyz[:, 3:, :]
        else:
            norm = None
        xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        x = self.reshape(l3_points, (-1, 1024))
        x = self.classifier(x)
        x = self.fc(x)
        x = self.logsoftmax(x)
        return x


class Pointnet2clsModelMSG(nn.Cell):
    """
    Constructs a PointnetNet2 architecture from
    PointnetNet2: Deep Hierarchical Feature Learning on Point Sets in a Metric Space <https://arxiv.org/abs/1706.02413>.

    Args:
        normal_channel (bool): Whether to use the channels of points' normal vector. Default: True.
        pretrained (bool): If True, returns a model pre-trained on PointnetNet2. Default: False.

    Inputs:
        - points(Tensor) - Tensor of original points. shape:[batch, channels, npoints].

    Outputs:
        Tensor of shape :[batch, 40].

    Supported Platforms:
        ``GPU``

    Examples:
        >> import numpy as np
        >> import mindspore as ms
        >> from mindspore import Tensor, context
        >> from mindvision.ms3d.models.backbones import pointnet2cls
        >> context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, max_call_depth=2000)
        >> net = Pointnet2clsModel(normal_channel=True)
        >> xyz = Tensor(np.ones((24,6, 1024)),ms.float32)
        >> output = net(xyz)
        >> print(output.shape)
        (24, 40)

    About PointNet2Cls:

    This architecture is based on PointNet2 classfication SSG,
    compared with PointNet, PointNet2_SSG added local feature extraction.

    Citation

    .. code-block::

        @article{qi2017pointnetplusplus,
          title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
          author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1706.02413},
          year={2017}
        }
    """

    def __init__(self, normal_channel=False):
        super(Pointnet2clsModelMSG, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # 512 npoint = points sampled in farthest point sampling
        # [0.1,0.2,0.4] radius = search radius in local region
        # [16,32,128] nsample = how many points in each local region
        # [[32,32,64], [64,64,128], [64,96,128]] mlp = output size for MLP on each point
        # + 3 = xyz 3-dim coordinates
        self.sa1 = PointNet2SetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNet2SetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320 + 3, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNet2SetAbstraction(None, None, None, 256 + 256 + 128 + 3, [256, 512, 1024], True)

        # fc1 input:1024
        self.fc1 = nn.Dense(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        # fc2 input:512
        self.fc2 = nn.Dense(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        # fc3 input:256
        self.fc3 = nn.Dense(256, 40)

        self.relu = ops.ReLU()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.logsoftmax = nn.LogSoftmax(axis=1)

    def construct(self, xyz):
        """construct method"""
        xyz = self.transpose(xyz, (0, 2, 1))
        if self.normal_channel:
            norm = xyz[:, 3:, :]
        else:
            norm = None
        xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        x = self.reshape(l3_points, (-1, 1024))
        x = self.drop1(self.relu(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = self.logsoftmax(x)
        return x


class Pointnet2segModel(nn.Cell):
    """
    Constructs a PointnetNet2 architecture from
    PointnetNet2: Deep Hierarchical Feature Learning on Point Sets in a Metric Space <https://arxiv.org/abs/1706.02413>.

    Args:
        num_classes (int): Number of classes.Default: 20.
        use_color (bool): Whether to use the channels of points' color vector. Default: False.
        use_normal (bool): Whether to use the channels of points' normal vector. Default: False.

    Inputs:
        - points(Tensor) - Tensor of original points. shape:[batch, channels, npoints].

    Outputs:
        Tensor of shape :[batch, npoints, 20].

    Supported Platforms:
        ``GPU``

    Examples:
        >> import numpy as np
        >> import mindspore as ms
        >> from mindspore import Tensor, context
        >> context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, max_call_depth=2000)
        >> net = Pointnet2segModel(use_color=False, use_normal=False)
        >> xyz = Tensor(np.ones((24,3,8192)),ms.float32)
        >> output = net(xyz)
        >> print(output.shape)
        (24, 8192, 20)

    About PointNet2Cls:

    This architecture is based on PointNet2 segamentation SSG,
    compared with PointNet, PointNet2_SSG added local feature extraction.

    Citation

    .. code-block::

        @article{qi2017pointnetplusplus,
          title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
          author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1706.02413},
          year={2017}
        }
    """

    def __init__(self, num_classes=20, use_color=False, use_normal=False):
        super(Pointnet2segModel, self).__init__()
        self.use_color = use_color
        self.use_normal = use_normal

        if use_color and use_normal:
            in_channel = 9
        elif use_color or use_normal:
            in_channel = 6
        else:
            in_channel = 3
        # Set Abstraction layers
        self.sa1 = PointNet2SetAbstraction(1024, 0.1, 32, in_channel + 3, [32, 32, 64], False)
        self.sa2 = PointNet2SetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNet2SetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNet2SetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        # Feature Propagation layers
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv2d(128, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(128, num_classes, 1)

        self.relu = ops.ReLU()
        self.reshape = ops.Reshape()
        self.log_softmax = ops.LogSoftmax()
        self.transpose = ops.Transpose()

    def construct(self, xyz):
        "seg construct"
        if self.use_color and self.use_normal:
            l0_points = xyz[:, :9, :]
        elif self.use_color or self.use_normal:
            l0_points = xyz[:, :6, :]
        else:
            l0_points = xyz[:, :3, :]

        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = ops.ExpandDims()(l0_points, 2)

        x = self.drop1(self.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = ops.Squeeze(2)(x)
        x = self.transpose(x, (0, 2, 1))
        a, b, c = x.shape
        x = ops.LogSoftmax(1)(x.reshape(a * b, c))
        return x.reshape(a, b, c)


class Pointnet2segModelMSG(nn.Cell):
    """
    Constructs a PointnetNet2 architecture from
    PointnetNet2: Deep Hierarchical Feature Learning on Point Sets in a Metric Space <https://arxiv.org/abs/1706.02413>.

    Args:
        num_classes (int): Number of classes.Default: 20.
        use_color (bool): Whether to use the channels of points' color vector. Default: False.
        use_normal (bool): Whether to use the channels of points' normal vector. Default: False.

    Inputs:
        - points(Tensor) - Tensor of original points. shape:[batch, channels, npoints].

    Outputs:
        Tensor of shape :[batch, npoints, 20].

    Supported Platforms:
        ``GPU``

    Examples:
        >> import numpy as np
        >> import mindspore as ms
        >> from mindspore import Tensor, context
        >> context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, max_call_depth=2000)
        >> net = Pointnet2segModel(use_color=False, use_normal=False)
        >> xyz = Tensor(np.ones((24,3,8192)),ms.float32)
        >> output = net(xyz)
        >> print(output.shape)
        (24, 8192, 20)

    About PointNet2Cls:

    This architecture is based on PointNet2 segamentation MSG,
    compared with PointNet, PointNet2_SSG added local feature extraction.

    Citation

    .. code-block::

        @article{qi2017pointnetplusplus,
          title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
          author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1706.02413},
          year={2017}
        }
    """

    def __init__(self, num_classes=20, use_color=False, use_normal=False):
        super(Pointnet2segModelMSG, self).__init__()
        self.use_color = use_color
        self.use_normal = use_normal

        if use_color and use_normal:
            in_channel = 9
        elif use_color or use_normal:
            in_channel = 6
        else:
            in_channel = 3
        # Set Abstraction layers
        self.sa1 = PointNet2SetAbstractionMsg(1024, [0.05, 0.1], [16, 32], in_channel + 3, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNet2SetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64+ 3, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNet2SetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128+ 3, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNet2SetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256+ 3, [[256, 256, 512], [256, 384, 512]])

        # Feature Propagation layers
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv2d(128, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(128, num_classes, 1)

        self.relu = ops.ReLU()
        self.reshape = ops.Reshape()
        self.log_softmax = ops.LogSoftmax()
        self.transpose = ops.Transpose()

    def construct(self, xyz):
        "seg construct"
        if self.use_color and self.use_normal:
            l0_points = xyz[:, :9, :]
        elif self.use_color or self.use_normal:
            l0_points = xyz[:, :6, :]
        else:
            l0_points = xyz[:, :3, :]

        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = ops.ExpandDims()(l0_points, 2)

        x = self.drop1(self.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = ops.Squeeze(2)(x)
        x = self.transpose(x, (0, 2, 1))
        a, b, c = x.shape
        x = ops.LogSoftmax(1)(x.reshape(a * b, c))
        return x.reshape(a, b, c)


"""import numpy as np
import mindspore as ms
from mindspore import Tensor, context
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, max_call_depth=20000)
net = Pointnet2segModelMSG(use_color=False, use_normal=False)
xyz = Tensor(np.ones((4,12,8192)),ms.float32)
output = net(xyz)
print(output.shape)"""
