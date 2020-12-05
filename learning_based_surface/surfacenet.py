"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.pointconv_util import PointConvDensitySetAbstraction
from learning_based_surface.surface_conv import AxisConv, Merge, index_points  # , SurfaceConvPurity
from learning_based_surface.grid_conv import SurfaceRotateConv as SurfaceConvPurity

# [512, 256, 128, 64, 32, 16] auxiliary6
# [512, 128, 32] auxiliary3


def lc_std_loss(lc):
    """
    lc: BNK3
    """
    B, N, K, _ = lc.shape
    lc = lc.reshape(B*N, K, 3)
    x = lc[:, :, 0]  # BN, K
    y = lc[:, :, 1]  # BN, K
    x_std = torch.std(x, 1).mean()
    y_std = torch.std(y, 1).mean()
    return x_std + y_std

class AxisNet(nn.Module):
    def __init__(self, regression_dim=6):
        super(AxisNet, self).__init__()
        scale = 2
        self.sa1 = AxisConv(in_channel=3, mlp=[3*scale, 8*scale, 8*scale])
        self.sa2 = AxisConv(in_channel=8*scale+3, mlp=[8*scale, 8*scale, 16*scale])
        self.sa3 = AxisConv(in_channel=16*scale+3,  mlp=[16*scale, 16*scale, 32*scale], merge=True)
        self.fc1 = nn.Linear(32*scale, 16*scale)
        self.bn1 = nn.BatchNorm1d(16*scale)
        self.drop1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(16*scale, 16*scale)
        self.bn2 = nn.BatchNorm1d(16*scale)
        self.drop2 = nn.Dropout(0.0)
        self.fc3 = nn.Linear(16*scale, regression_dim)

    def forward(self, xyz, neighbors):
        # input : B N K C
        B, N, _ = xyz.shape
        xyz = index_points(xyz, neighbors).permute(0, 3, 2, 1)  # BCKN
        l1_points = self.sa1(xyz, None)
        l2_points = self.sa2(xyz, l1_points)
        l3_points = self.sa3(xyz, l2_points)  # BCKN
        x = l3_points.view(B*N, -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        # orthogonalization and normalization
        alpha1 = x[:, 0:3]  # BN X 3
        alpha2 = x[:, 3:]
        alpha1_norm = alpha1.norm(dim=1) + 1e-9  # BN
        k = torch.matmul(alpha1.unsqueeze(1), alpha2.unsqueeze(2)).squeeze()/alpha1_norm**2  # BN
        beta2 = alpha2-k.unsqueeze(1)*alpha1  # BN 3
        x_axis = beta2/(beta2.norm(dim=1, keepdim=True)+1e-9)  # BN 3
        z_axis = alpha1/alpha1_norm.unsqueeze(1)  # BN 3
        y_axis = z_axis.cross(x_axis)  # BN 3
        # self.fc1.weight.retain_grad()
        return [x_axis.reshape(B, N, -1), y_axis.reshape(B, N, -1), z_axis.reshape(B, N, -1)]


class SurfaceNet(nn.Module):
    def __init__(self, num_classes=40, point_num=[2048, 512, 512, 128]):
        super(SurfaceNet, self).__init__()
        scale = 2
        pad = 3 + 0
        r = 0.15
        self.axis_net = AxisNet()
        self.sa0 = SurfaceConvPurity(npoint=point_num[0], in_channel=pad + 3, out_channel=16*scale, radius=r)
        self.sa02 = SurfaceConvPurity(npoint=point_num[0], in_channel=16*scale+pad, out_channel=16*scale, radius=r)
        self.sa03 = SurfaceConvPurity(npoint=point_num[0], in_channel=16*scale+pad, out_channel=16*scale, radius=r)
        self.sa04 = SurfaceConvPurity(npoint=point_num[0], in_channel=16*scale+pad, out_channel=16*scale, radius=r)
        self.fc_res0 = nn.Linear(16*scale, 16*scale)
        self.bn_res0 = nn.BatchNorm1d(16*scale)

        self.sa1 = SurfaceConvPurity(npoint=point_num[1], in_channel=16*scale+pad, out_channel=64*scale, radius=r)
        self.sa12 = SurfaceConvPurity(npoint=point_num[2], in_channel=64*scale+pad, out_channel=64*scale, radius=r)
        self.sa13 = SurfaceConvPurity(npoint=point_num[2], in_channel=64*scale+pad, out_channel=64*scale, radius=2*r)
        self.sa14 = SurfaceConvPurity(npoint=point_num[2], in_channel=64*scale+pad, out_channel=64*scale, radius=2*r)
        self.fc_res1 = nn.Linear(64*scale, 64*scale)
        self.bn_res1 = nn.BatchNorm1d(64*scale)

        self.sa2 = SurfaceConvPurity(npoint=point_num[3], in_channel=64*scale+pad, out_channel=128*scale, radius=2*r)
        # self.sa3 = SurfaceConvPurity(npoint=point_num[4], in_channel=64*scale+pad, out_channel=128 * scale,
        #                            radius=0.25)
        # self.sa31 = SurfaceConvPurity(npoint=point_num[4], in_channel=64 * scale + pad, out_channel=64 * scale,
        #                             radius=0.25)
        # self.sa21 = SurfaceConv(npoint=point_num[2], in_channel=64 + 3, mlp=[64, 64, 128])
        # self.sa22 = SurfaceConv(npoint=point_num[2], in_channel=32 + 3, mlp=[32, 32, 64])
        # self.sa3 = SurfaceConv(npoint=point_num[2], in_channel=256 + 3, mlp=[256, 256, 512])
        self.sa4 = Merge(npoint=1, in_channel=128*scale+3,  mlp=[128*scale, 256*scale])

        self.fc1 = nn.Linear(256*scale, 128*scale)
        self.bn1 = nn.BatchNorm1d(128*scale)
        self.drop1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(128*scale, 64*scale)
        self.bn2 = nn.BatchNorm1d(64*scale)
        self.drop2 = nn.Dropout(0.0)
        self.fc3 = nn.Linear(64*scale, num_classes)
        self.point_num = point_num
        self.scale = scale

    def forward(self, xyz, neighbors, data_idxes):
        B, N, C = xyz.shape

        # seperate the neighbor information into layers
        neighbors_layer = []
        local_coordinates_layer = []
        data_idxes_layer = []
        parameters_layer = [0]*5
        lc_std = 0
        cid = 0
        current_xyz = xyz
        for idx, point_num in enumerate(self.point_num):
            current_xyz = index_points(current_xyz, data_idxes[:, cid:cid+point_num].squeeze())
            x_axis, y_axis, z_axis = self.axis_net(current_xyz, neighbors[:, cid:cid+point_num, :].squeeze())  # BNC
            # print(self.axis_net.fc1.weight)
            axis = torch.stack([x_axis, y_axis, z_axis]).permute(1,2,3,0)  # BNC3
            grouped_xyz = index_points(current_xyz, neighbors[:, cid:cid+point_num, :].squeeze())
            grouped_xyz = grouped_xyz- current_xyz.unsqueeze(2)  # BNKC
            lc = torch.matmul(grouped_xyz, axis)  # BNK3
            local_coordinates_layer.append(lc)

            lc_std += lc_std_loss(lc)

            # local_coordinates_layer.append(local_coordinates[:, cid:cid+point_num,:, :].squeeze())
            neighbors_layer.append(neighbors[:, cid:cid+point_num, :].squeeze())
            data_idxes_layer.append(data_idxes[:, cid:cid+point_num].squeeze())
            cid += point_num
        # 2048 -> 2048

        # projection

        l0_xyz, l0_points_res = self.sa0(xyz, None, local_coordinates_layer[0],
                                     neighbors_layer[0], parameters_layer[0], data_idxes_layer[0])
        l0_xyz, l0_points = self.sa02(l0_xyz, l0_points_res, local_coordinates_layer[0],
                                      neighbors_layer[0], parameters_layer[0], data_idxes_layer[0])
        #l0_xyz, l0_points = self.sa03(l0_xyz, l0_points, local_coordinates_layer[0],
        #                              neighbors_layer[0], parameters_layer[0], data_idxes_layer[0])
        #l0_xyz, l0_points = self.sa04(l0_xyz, l0_points, local_coordinates_layer[0],
        #                              neighbors_layer[0], parameters_layer[0], data_idxes_layer[0])
        # res0 = F.relu(self.bn_res0(self.fc_res0(l0_points_res).permute(0, 2, 1)).permute(0, 2, 1))
        # l0_points = l0_points + l0_points_res

        # 2048 -> 512
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points, local_coordinates_layer[1],
                                     neighbors_layer[1], parameters_layer[0], data_idxes_layer[1])
        # 512 -> 512
        l2_xyz, l2_points = self.sa12(l1_xyz, l1_points, local_coordinates_layer[2],
                                      neighbors_layer[2], parameters_layer[1], data_idxes_layer[2])
        #l2_xyz, l2_points = self.sa13(l2_xyz, l2_points, local_coordinates_layer[2],
        #                               neighbors_layer[2], parameters_layer[1], data_idxes_layer[2])
        #l2_xyz, l2_points = self.sa14(l2_xyz, l2_points, local_coordinates_layer[2],
        #                              neighbors_layer[2], parameters_layer[1], data_idxes_layer[2])
        # res1 = F.relu(self.bn_res1(self.fc_res1(l1_points).permute(0, 2, 1)).permute(0, 2, 1))
        # l2_points = res1 + l2_points

        # 512 -> 128
        l2_xyz, l2_points = self.sa2(l2_xyz, l2_points, local_coordinates_layer[3],
                                     neighbors_layer[3], parameters_layer[2], data_idxes_layer[3])

        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, local_coordinates_layer[4],
        #                             neighbors_layer[4], data_idxes_layer[4])
        # l3_xyz, l3_points = self.sa31(l3_xyz, l3_points, local_coordinates_layer[4],
        #                             neighbors_layer[4], data_idxes_layer[4])
        # l2_xyz, l2_points = self.sa21(l2_xyz, l2_points, local_coordinates_layer[3],
        #                             neighbors_layer[3], data_idxes_layer[3])
        # l2_xyz, l2_points = self.sa22(l2_xyz, l2_points, local_coordinates_layer[3],
        #                              neighbors_layer[3], data_idxes_layer[3])
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, local_coordinates_layer[2],
        #                            neighbors_layer[2], data_idxes_layer[2])

        l4_xyz, l4_points = self.sa4(l2_xyz, l2_points)
        x = l4_points.view(B, self.scale * 256)
        # x = l2_points.mean(1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, lc_std


class SurfaceNet2(nn.Module):
    def __init__(self, num_classes=40, point_num=[2048, 512, 512, 128]):
        super(SurfaceNet2, self).__init__()
        scale = 4
        pad = 3
        self.sa0 = SurfaceConvPurity(npoint=point_num[0], in_channel=3, out_channel=8*scale, radius=0.1)

        self.sa1 = SurfaceConvPurity(npoint=point_num[1], in_channel=8*scale+pad, out_channel=16*scale, radius=0.1)

        self.sa12 = SurfaceConvPurity(npoint=point_num[2], in_channel=16*scale+pad, out_channel=16*scale, radius=0.2)

        self.sa2 = SurfaceConvPurity(npoint=point_num[3], in_channel=16*scale+pad, out_channel=64*scale, radius=0.2)

        self.sa4 = Merge(npoint=1, in_channel=64*scale+pad,  mlp=[64*scale, 128*scale, 256*scale])
        '''self.sa4 = SurfaceConv(npoint=point_num[3], in_channel=256 + 3, mlp=[256, 256, 256])
        self.sa5 = SurfaceConv(npoint=point_num[4], in_channel=256 + 3, mlp=[256, 512, 512]) 
        self.sa6 = SurfaceConv(npoint=point_num[5], in_channel=512 + 3, mlp=[512, 512, 1024])
        '''

        self.fc1 = nn.Linear(256*scale, 128*scale)
        self.bn1 = nn.BatchNorm1d(128*scale)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128*scale, 64*scale)
        self.bn2 = nn.BatchNorm1d(64*scale)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64*scale, num_classes)
        self.point_num = point_num
        self.scale = scale

    def forward(self, xyz, local_coordinates, neighbors, data_idxes):
        B, N, C = xyz.shape

        # seperate the neighbor information into layers
        neighbors_layer = []
        local_coordinates_layer = []
        data_idxes_layer = []
        cid = 0
        for idx, point_num in enumerate(self.point_num):
            local_coordinates_layer.append(local_coordinates[:, cid:cid+point_num,:, :].squeeze())
            neighbors_layer.append(neighbors[:, cid:cid+point_num, :].squeeze())
            data_idxes_layer.append(data_idxes[:, cid:cid+point_num].squeeze())
            cid += point_num
        # 2048 -> 2048
        l0_xyz, l0_points = self.sa0(xyz, None, local_coordinates_layer[0],
                                     neighbors_layer[0], data_idxes_layer[0])

        # 2048 -> 512
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points, local_coordinates_layer[1],
                                     neighbors_layer[1], data_idxes_layer[1])
        # 512 -> 512
        l2_xyz, l2_points = self.sa12(l1_xyz, l1_points, local_coordinates_layer[2],
                                      neighbors_layer[2], data_idxes_layer[2])
        # 512 -> 128
        l2_xyz, l2_points = self.sa2(l2_xyz, l2_points, local_coordinates_layer[3],
                                     neighbors_layer[3], data_idxes_layer[3])

        l4_xyz, l4_points = self.sa4(l2_xyz, l2_points)
        x = l4_points.view(B, self.scale * 256)

        # x = l2_points.mean(1).squeeze()

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x


if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    label = torch.randn(8,16)
    model = SurfaceNet(num_classes=40)
    output= model(input)
    print(output.size())

