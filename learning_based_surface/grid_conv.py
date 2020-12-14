from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm
import math
import numpy as np
from torch_scatter import scatter_mean, scatter_add
from learning_based_surface.surface_conv import WeightNet

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cut(x, upbound):
    new_x = -F.relu(-x+upbound)+upbound
    # new_x = F.relu(x-lowbound)+lowbound
    return new_x


def circle_conv(local_coordinate, feature, radius=0.1, partition_num=5):
    """input: local_coordinate： BNKC
    ---------------------------------
    output: index:BNK"""
    feature = feature.permute(0, 1, 3, 2)
    B, N, K, C = feature.shape
    result = torch.zeros((B, N, partition_num, C), device=feature.device, dtype=feature.dtype)
    distance = local_coordinate[:, :, :, 0:2].norm(dim=3)
    distance = cut(distance/radius, 0.99)
    distance = (distance*partition_num).floor().long()
    # result = scatter_mean(feature, distance.unsqueeze(3).repeat(1,1,1,C), dim=2, dim_size=partition_num)
    result.scatter_add_(2, distance.unsqueeze(3).repeat(1,1,1,C), feature)
    # result = result/(result.norm(dim=3, keepdim=True)+1e-7)
    '''norm = result.norm(dim=3)
    res = (norm<1e-9).float().sum()
    rate = res/(B*N*partition_num)
    print(rate)'''
    return result


def get_index(local_coordinate, partition_num, radius=0.1):
    """input: local_coordinate： BNKC
    feature:BNKC
    ---------------------------------
    output: index:BNK"""
    partition_num = partition_num - 1
    x_coordinate = local_coordinate[:, :, :, 0]  # BNK
    y_coordinate = local_coordinate[:, :, :, 1]  # BNK
    filter_index = torch.where((x_coordinate**2 > radius**2)|(y_coordinate**2 > radius**2))  # (z_coordinate**2 > radius**2))
    x_index = ((x_coordinate + radius) / (2 * radius) * (partition_num-0.01))
    y_index = ((y_coordinate + radius) / (2 * radius) * (partition_num-0.01))

    left_down_x = x_index.floor()
    left_down_y = y_index.floor()

    left_up_x = x_index.floor() + 1
    left_up_y = y_index.floor()

    right_down_x = x_index.floor()
    right_down_y = y_index.floor() + 1

    right_up_x = x_index.floor() + 1
    right_up_y = y_index.floor() + 1

    index0 = (left_down_x * (partition_num + 1) + left_down_y) + 1
    index0[filter_index] = 0

    '''
    index1 = (left_up_x * (partition_num + 1) + left_up_y) + 1
    index1[filter_index] = 0

    index2 = (right_down_x * (partition_num + 1) + right_down_y) + 1
    index2[filter_index] = 0

    index3 = (right_up_x * (partition_num + 1) + right_up_y) + 1
    index3[filter_index] = 0

    left_down_weight = 1/(((x_index-left_down_x)**2 + (y_index-left_down_y)**2).sqrt()+1e-7)
    left_up_weight = 1/(((x_index-left_up_x)**2 + (y_index-left_up_y)**2).sqrt()+1e-7)
    right_down_weight = 1/(((x_index-right_down_x)**2 + (y_index-right_down_y)**2).sqrt()+1e-7)
    right_up_weight = 1/(((x_index-right_up_x)**2 + (y_index-right_up_y)**2).sqrt()+1e-7)

    #normalize_term = left_down_weight + left_up_weight + right_down_weight + right_up_weight
    #left_down_weight = left_down_weight/normalize_term
    #left_up_weight = left_up_weight / normalize_term
    #right_down_weight = right_down_weight / normalize_term
    #right_up_weight = right_up_weight / normalize_term
    '''
    # return [torch.cat((index0.long(), index1.long(), index2.long(), index3.long()), 2)], [torch.cat((left_down_weight, left_up_weight, right_down_weight, right_up_weight), 2)]
    return [index0.long()], None


def get_feature(indexes, weight, feature, partition_num):
    """input index:BNK
    feature:BNKC
    --------------------
    output: BNGC"""

    B, N, K, C = feature.shape
    grid_feature = torch.zeros((B, N, partition_num ** 2 + 1, C), device=feature.device, dtype=feature.dtype)
    grid_feature_norm = torch.zeros((B, N, partition_num ** 2 + 1), device=feature.device, dtype=feature.dtype)
    grid_feature_temp = torch.zeros((B, N, partition_num ** 2 + 1, C), device=feature.device, dtype=feature.dtype)

    # grid_feature_norm.scatter_add_(2, indexes[0], weight[0])  #BNG
    # grid_feature.scatter_add_(2, indexes[0].unsqueeze(3).repeat(1, 1, 1, C), feature.repeat(1, 1, 4, 1))  #BNGC
    # grid_feature = grid_feature[:, :, 1:, :]
    #
    # grid_feature_norm = grid_feature_norm[:,:,1:]
    # grid_feature = grid_feature/(grid_feature_norm.unsqueeze(3)+1e-9)

    grid_feature.scatter_add_(2, indexes[0].unsqueeze(3).repeat(1, 1, 1, C), feature)  # BNGC
    # grid_feature_temp.scatter_add_(2, indexes[0].unsqueeze(3).repeat(1, 1, 1, C), torch.ones_like(feature))
    # grid_feature_temp = grid_feature_temp[:,:,0,0].long()
    # lll = torch.where(grid_feature_temp==0)
    # grid_feature = grid_feature[:, :, 1:, :]
    # # grid_feature = grid_feature/(grid_feature.norm(dim=3, keepdim=True)+1e-9)
    return grid_feature


class GridConv(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, grid_with, out_features, bias=True):
        super(GridConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_with = grid_with
        self.weight = torch.nn.parameter.Parameter(torch.Tensor(out_features, (grid_with*grid_with)*in_features))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = self.weight.reshape(self.out_features, self.grid_with, self.grid_with, self.in_features)
        weight1 = torch.rot90(weight, 1, [1,2])
        weight1 = weight1.reshape(self.out_features, self.grid_with*self.grid_with*self.in_features)
        x1 = F.linear(input, weight1, self.bias)

        weight2 = torch.rot90(weight, 2, [1, 2])
        weight2 = weight2.reshape(self.out_features, self.grid_with * self.grid_with * self.in_features)
        x2 = F.linear(input, weight2, self.bias)

        weight3 = torch.rot90(weight, 3, [1, 2])
        weight3 = weight3.reshape(self.out_features, self.grid_with * self.grid_with * self.in_features)
        x3 = F.linear(input, weight3, self.bias)

        weight0 = weight.reshape(self.out_features, self.grid_with * self.grid_with * self.in_features)
        x0 = F.linear(input, weight0, self.bias)
        m = nn.MaxPool1d(4, stride=4)
        x = m(torch.stack([x0, x1, x2, x3]).permute(1, 2, 0)).squeeze()
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def get_rotate_weight(weight, rotate, grid_width):
    rng_origin = (grid_width - 1) / (2 * 1.4142135623730951)
    rng = torch.linspace(-rng_origin, rng_origin, grid_width, dtype=weight.dtype, device=weight.device)
    x_coordinate, y_coordinate = torch.meshgrid(rng, rng)
    x_coordinate = x_coordinate.reshape(-1)
    y_coordinate = y_coordinate.reshape(-1)
    rotate_matrix = torch.tensor([[math.cos(rotate), math.sin(-rotate)],
                                  [math.sin(rotate), math.cos(rotate)]], dtype=weight.dtype,
                                 device=weight.device)
    new_xy = torch.matmul(rotate_matrix, torch.stack([x_coordinate, y_coordinate]))
    new_xy = new_xy + (grid_width - 1) / 2
    new_xy_left_down = new_xy.floor()

    residual_xy = new_xy - new_xy_left_down
    left_down_x = new_xy_left_down.long()[0, :]
    left_down_y = new_xy_left_down.long()[1, :]
    left_down_weight = weight[:, left_down_x, left_down_y, :]
    left_up_weight = weight[:, left_down_x, left_down_y + 1, :]
    right_down_weight = weight[:, left_down_x + 1, left_down_y, :]
    right_up_weight = weight[:, left_down_x + 1, left_down_y + 1, :]
    residual_xy = residual_xy.unsqueeze(2).unsqueeze(1)
    weight = (residual_xy[0] * right_down_weight + (1 - residual_xy[0]) * left_down_weight) * (1 - residual_xy[1]) + \
             (residual_xy[0] * right_up_weight + (1 - residual_xy[0]) * left_up_weight) * residual_xy[1]
    weight = weight.reshape(weight.shape[0], -1)
    return weight


class RotateConv(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, grid_width, out_features, bias=True):
        super(RotateConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_width = grid_width
        self.weight = torch.nn.parameter.Parameter(torch.Tensor(out_features, (grid_width*grid_width+1)*in_features))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        x = F.linear(input, self.weight, self.bias)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class SurfaceRotateConv(nn.Module):
    def __init__(self, npoint, in_channel, out_channel, radius, partition_num=5):
        super(SurfaceRotateConv, self).__init__()
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.npoint = npoint
        self.partition_num = partition_num
        self.conv_linear = weight_norm(RotateConv(in_channel, partition_num, out_channel))
        # self.conv_bn = nn.BatchNorm1d(out_channel)
        # self.linear = nn.Linear(out_channel, out_channel)
        # self.bn_linear = nn.BatchNorm1d(out_channel)
        self.radius = radius


    def forward(self, xyz, points, local_coordinates, neighbor_lists, parameter_list,  data_idx):
        """needs more modification
            neighbor_lists: B, N, Neighbor_num
        """
        B, N, _ = xyz.shape
        new_xyz = index_points(xyz, data_idx)
        # fps sampling index
        # parameter_list = parameter_list.squeeze()
        # xyz = torch.cat((xyz, parameter_list), 2)

        if points is not None:
            # points = torch.cat((points, xyz), 2)
            grouped_points = index_points(points, neighbor_lists)
            grouped_points = torch.cat([local_coordinates, grouped_points], 3).permute(0, 3, 2, 1)  # [:,:,:,2].unsqueeze(3)
        else:
            # K = local_coordinates.shape[2]

            grouped_points = torch.cat((local_coordinates, new_xyz.unsqueeze(2).repeat(1,1,local_coordinates.shape[2],1)), 3).permute(0, 3, 2, 1) #index_points(xyz, neighbor_lists).permute(0, 3, 2, 1)  #

        grouped_points = grouped_points.permute(0, 3, 1, 2)  # BNCK

        local_coordinates = local_coordinates.reshape(B, self.npoint, -1, 3)  # .permute(0, 3, 2, 1)  # BNKC

        # grid conv
        index, weight = get_index(local_coordinates, self.partition_num, radius=self.radius)  # BNG
        feature = get_feature(index, weight, grouped_points.permute(0, 1, 3, 2), self.partition_num)  # BNKC

        # circle conv
        # feature = circle_conv(local_coordinates, grouped_points, radius=self.radius, partition_num=self.partition_num)

        feature = feature.reshape(B*self.npoint, -1)

        new_points = self.conv_linear(feature)
        new_points = F.relu(new_points).view(B, self.npoint, -1)

        # new_points = self.linear(new_points)
        # new_points = F.relu(self.bn_linear(new_points.permute(0, 2, 1))).permute(0, 2, 1)
        '''
        local_coordinates = local_coordinates.reshape(B, self.npoint, -1, 3).permute(0, 3, 2, 1)  # BCKN
        weights = self.weightnet(local_coordinates[:, 0:2, :, :]).permute(0, 3, 2, 1)
        new_points = torch.matmul(input=grouped_points, other=weights).view(B, self.npoint, -1)
        # new_points = self.bn_conv(new_points.permute(0, 2, 1)).permute(0, 2, 1)

        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1)).permute(0, 2, 1)
        new_points = F.relu(new_points)'''

        return new_xyz, new_points


class SurfaceCircleConv(nn.Module):
    def __init__(self, npoint, in_channel, out_channel, radius, partition_num=5):
        super(SurfaceCircleConv, self).__init__()
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.npoint = npoint
        self.partition_num = partition_num
        self.conv_linear = nn.Linear(in_channel*partition_num, out_channel)
        self.conv_bn = nn.BatchNorm1d(out_channel)
        self.linear = nn.Linear(out_channel, out_channel)
        self.bn_linear = nn.BatchNorm1d(out_channel)
        self.radius = radius

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in [out_channel]*3:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        '''self.weightnet = WeightNet(2, 16)
        self.linear = nn.Linear(16 * self.out_channel, self.out_channel)
        self.bn_linear = nn.BatchNorm1d(self.out_channel)'''

    def forward(self, xyz, points, local_coordinates, neighbor_lists, parameter_list,  data_idx):
        """needs more modification
            neighbor_lists: B, N, Neighbor_num
        """
        B, N, _ = xyz.shape
        new_xyz = index_points(xyz, data_idx)
        # fps sampling index
        # parameter_list = parameter_list.squeeze()
        # xyz = torch.cat((xyz, parameter_list), 2)

        if points is not None:
            points = torch.cat((points, xyz), 2)
            grouped_points = index_points(points, neighbor_lists).permute(0, 3, 2, 1)
        else:
            # K = local_coordinates.shape[2]
            grouped_points = index_points(xyz, neighbor_lists).permute(0, 3, 2, 1)  #  torch.cat((local_coordinates, new_xyz.unsqueeze(2).repeat(1,1,local_coordinates.shape[2],1)), 3).permute(0, 3, 2, 1) #
        '''
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))'''

        grouped_points = grouped_points.permute(0, 3, 1, 2)  # BNCK

        local_coordinates = local_coordinates.reshape(B, self.npoint, -1, 3)  # .permute(0, 3, 2, 1)  # BNKC

        # circle conv
        feature = circle_conv(local_coordinates, grouped_points, radius=self.radius, partition_num=self.partition_num)

        feature = feature.reshape(B*self.npoint, -1)

        new_points = self.conv_linear(feature)
        new_points = F.relu(self.conv_bn(new_points)).view(B, self.npoint, -1)

        new_points = self.linear(new_points)
        new_points = F.relu(self.bn_linear(new_points.permute(0, 2, 1))).permute(0, 2, 1)
        '''
        local_coordinates = local_coordinates.reshape(B, self.npoint, -1, 3).permute(0, 3, 2, 1)  # BCKN
        weights = self.weightnet(local_coordinates[:, 0:2, :, :]).permute(0, 3, 2, 1)
        new_points = torch.matmul(input=grouped_points, other=weights).view(B, self.npoint, -1)
        # new_points = self.bn_conv(new_points.permute(0, 2, 1)).permute(0, 2, 1)

        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1)).permute(0, 2, 1)
        new_points = F.relu(new_points)'''

        return new_xyz, new_points