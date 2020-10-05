from utils.mls import MLS
from torch import nn
import torch.nn.functional as F
import torch

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = F.relu(bn(conv(weights)))

        return weights


class SurfaceConv(nn.Module):
    def __init__(self, npoint, in_channel,  mlp):
        super(SurfaceConv, self).__init__()

        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.in_channel = in_channel
        self.npoint = npoint

    def forward(self, xyz, points, local_coordinates, neighbor_lists, data_idx):
        """needs more modification"""
        B, N, C = points.shape

        # fps sampling

        weights = self.weightnet(self.local_coordinates)
        data = points[neighbor_lists]
        new_points = torch.matmul(input=data, other=weights.permute(0, 3, 2, 1)).view(B,
                                                                                                                self.npoint,
                                                                                                                    -1)
        new_points = new_points[data_idx]

        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        return xyz, new_points


class SurfacePool(nn.Module):
    def __init__(self, points, mlp):
        super(SurfacePool, self).__init__()

    def forward(self, xyz, points):
        return xyz