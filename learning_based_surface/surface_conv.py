from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch


class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_len=3):
        super(WeightNet, self).__init__()
        '''hidden_unit = []
        steps = (math.log(out_channel) - 4)/hidden_len

        for i in range(hidden_len):
            hidden_unit.append(out_channel//(2**int(round(steps*(hidden_len-i)))))'''
        hidden_unit = [16, 16]
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
            weights = F.relu(bn(conv(weights.contiguous())))
        return weights


class WeightNet2(nn.Module):

    def __init__(self, in_channel, pre_channel, out_channel, hidden_unit=[8, 8, 16]):
        super(WeightNet2, self).__init__()
        '''for i in range(len(hidden_unit)):
            hidden_unit[i] = out_channel//(2**(i+1))'''
        self.out_channel = out_channel
        self.pre_channel = pre_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Linear(in_channel, hidden_unit[0]*pre_channel))
        self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[0]*pre_channel))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Linear(hidden_unit[i - 1]*pre_channel, hidden_unit[i]*pre_channel))
            self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[i]*pre_channel))
        self.mlp_convs.append(nn.Linear(hidden_unit[-1]*pre_channel, out_channel*pre_channel, 1))
        self.mlp_bns.append(nn.BatchNorm1d(out_channel*pre_channel))

    def forward(self, localized_xyz):
        # input: xyz : BxCxKxN -> BxNxKxC1
        # output: weight : BxNxKxPrexC2
        localized_xyz = localized_xyz.permute(0, 3, 2, 1)
        B, N, K, C = localized_xyz.shape
        localized_xyz = localized_xyz.reshape(-1, C)
        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = F.relu(bn(conv(weights)))
        weights = weights.view(B, N, K, self.pre_channel, self.out_channel)  # BxNxKxPreXC2
        return weights


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    # if points.shape[1] == idx.shape[1]:
    #    return points

    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class AxisConv(nn.Module):
    def __init__(self, in_channel,  mlp, merge=False):
        super(AxisConv, self).__init__()
        self.merge = merge
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.fcs = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(weight_norm(nn.Conv2d(last_channel, out_channel, [3, 1], padding=[1, 0])))  #
            # self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if merge:
            self.fc = nn.Linear(24, 1)


    def forward(self, xyz, grouped_points):
        """needs more modification
            grouped_points: B C K N
        """

        if grouped_points is None:
            grouped_points = xyz
        else:
            grouped_points = torch.cat([grouped_points, xyz], 1)

        B, _, K, N = grouped_points.shape
        for i, conv in enumerate(self.mlp_convs):
            # bn = self.mlp_bns[i]
            # fc = self.fcs[i]
            # grouped_points = F.relu(fc(grouped_points.permute(0,1,3,2).reshape(-1, K)))
            # grouped_points = grouped_points.reshape(B,-1,N,K).permute(0,1,3,2)
            grouped_points = F.relu(conv(grouped_points))
        if self.merge:
            grouped_points = grouped_points.permute(0,3,1,2).reshape(-1, K)
            grouped_points = self.fc(grouped_points).squeeze().reshape(B, N, -1)
            # grouped_points = grouped_points.mean(2).squeeze()
        return grouped_points


class SurfaceConvPurity(nn.Module):
    def __init__(self, npoint, in_channel,  out_channel):
        super(SurfaceConvPurity, self).__init__()
        weight_channel = out_channel
        self.bn_conv = nn.BatchNorm1d(out_channel *in_channel)
        self.linear = nn.Linear(in_channel*out_channel, out_channel)
        self.bn_linear = nn.BatchNorm1d(out_channel)
        self.in_channel = in_channel
        self.npoint = npoint
        self.weightnet = WeightNet(3, out_channel)
    def forward(self, xyz, points, local_coordinates, neighbor_lists, data_idx):
        """needs more modification
            neighbor_lists: B, N, Neighbor_num
        """
        B, N, _ = xyz.shape
        # fps sampling index
        new_xyz = index_points(xyz, data_idx)
        if points is not None:
            points = torch.cat((points, xyz), 2)
            grouped_points = index_points(points, neighbor_lists).permute(0, 3, 2, 1)
        else:
            grouped_points = index_points(xyz, neighbor_lists).permute(0, 3, 2, 1)  # (local_coordinates + new_xyz.unsqueeze(2)).permute(0, 3, 2, 1)

        grouped_points = grouped_points.permute(0, 3, 1, 2)  # BNCK

        local_coordinates = local_coordinates.reshape(B, self.npoint, -1, 3).permute(0, 3, 2, 1)  # BCKN

        weights = self.weightnet(local_coordinates[:, :, :, :]).permute(0, 3, 2, 1)  # BNKO
        new_points = torch.matmul(input=grouped_points, other=weights).view(B, self.npoint, -1)
        new_points = F.relu(self.bn_conv(new_points.permute(0, 2, 1)).permute(0, 2, 1))

        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1)).permute(0, 2, 1)
        new_points = F.relu(new_points)

        return new_xyz, new_points

class SurfaceConv2(nn.Module):
    def __init__(self, npoint, in_channel,  mlp):
        super(SurfaceConv2, self).__init__()
        self.mlp = mlp
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp[0:-1]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # self.bn_conv = nn.BatchNorm1d(weight_channel * mlp[-1])
        self.linear = nn.Linear(mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.in_channel = in_channel
        self.npoint = npoint
        self.weightnet = WeightNet2(3, mlp[-2], mlp[-1])
    def forward(self, xyz, points, local_coordinates, neighbor_lists, data_idx):
        """needs more modification
            neighbor_lists: B, N, Neighbor_num
        """

        if points is not None:
            points = torch.cat((points, xyz), 2)
        else:
            points = xyz

        B, N, C = points.shape

        # fps sampling
        new_xyz = index_points(xyz, data_idx)
        grouped_points = index_points(points, neighbor_lists).permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))
        grouped_points = grouped_points.permute(0, 3, 1, 2)  # BNIK
        grouped_points = grouped_points.reshape(B, self.npoint, -1, 1)

        local_coordinates = local_coordinates.reshape(B, self.npoint, -1, 3).permute(0, 3, 2, 1)  # BCKN

        weights = self.weightnet(local_coordinates[:,:,:,:])  # BNKIO
        weights = weights.permute(0, 1, 4, 3, 2)  # BNOIK
        weights = weights.reshape(B, self.npoint, self.mlp[-1], -1)
        new_points = torch.matmul(input=weights, other=grouped_points).view(B, self.npoint, -1) #
        # new_points = self.bn_conv(new_points.permute(0, 2, 1)).permute(0, 2, 1)

        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1)).permute(0, 2, 1)
        new_points = F.relu(new_points)

        return new_xyz, new_points

class Merge(nn.Module):
    def __init__(self, npoint, in_channel,  mlp):
        super(Merge, self).__init__()
        weight_channel = 16
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.linear = nn.Linear(weight_channel * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.in_channel = in_channel
        self.npoint = npoint
        self.weightnet = WeightNet(3, weight_channel)
    def forward(self, xyz, points):
        """needs more modification
            neighbor_lists: B, N, Neighbor_num
        """

        # fps sampling
        B, N, C = xyz.shape
        # new_xyz = torch.zeros(B, 1, C).to(device)
        new_xyz = xyz.mean(dim=1, keepdim=True)
        grouped_xyz = xyz.view(B, 1, N, C) - new_xyz.view(B, 1, 1, C)
        points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)  # [B, 1, npoint, C+D]
        # points = points.view(B, 1, N, -1)
        # Conv
        points = points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            points = F.relu(bn(conv(points.contiguous())))

        grouped_xyz = grouped_xyz.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=points.permute(0, 3, 1, 2), other=weights.permute(0, 3, 2, 1)).view(B,
                                                                                                      self.npoint, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_points = new_points.permute(0, 2, 1)

        return new_xyz, new_points


class SurfacePool(nn.Module):
    def __init__(self, points, mlp):
        super(SurfacePool, self).__init__()

    def forward(self, xyz, points):
        return xyz
