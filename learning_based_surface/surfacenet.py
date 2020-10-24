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
from learning_based_surface.surface_conv import SurfaceConv, Merge, SurfaceConv2, SurfaceConvPurity

# [512, 256, 128, 64, 32, 16] auxiliary6
# [512, 128, 32] auxiliary3
class SurfaceNet(nn.Module):
    def __init__(self, num_classes=40, point_num=[2048, 512, 512, 128]):
        super(SurfaceNet, self).__init__()
        self.sa0 = SurfaceConvPurity(npoint=point_num[0], in_channel=3, out_channel=16)
        self.sa02 = SurfaceConvPurity(npoint=point_num[0], in_channel=16 + 3, out_channel=32)
        self.sa03 = SurfaceConvPurity(npoint=point_num[0], in_channel=32 + 3, out_channel=32)
        self.sa04 = SurfaceConvPurity(npoint=point_num[0], in_channel=32 + 3, out_channel=32)
        self.fc_res0 = nn.Linear(16, 32)
        self.bn_res0 = nn.BatchNorm1d(32)

        self.sa1 = SurfaceConvPurity(npoint=point_num[1], in_channel=3, out_channel=64)
        self.sa12 = SurfaceConvPurity(npoint=point_num[2], in_channel=64 + 3, out_channel=64)
        self.sa13 = SurfaceConvPurity(npoint=point_num[2], in_channel=64 + 3, out_channel=64)
        self.sa14 = SurfaceConvPurity(npoint=point_num[2], in_channel=64 + 3, out_channel=128)
        self.fc_res1 = nn.Linear(64, 128)
        self.bn_res1 = nn.BatchNorm1d(128)

        self.sa2 = SurfaceConvPurity(npoint=point_num[3], in_channel=64 + 3, out_channel=256)
        # self.sa21 = SurfaceConv(npoint=point_num[2], in_channel=64 + 3, mlp=[64, 64, 128])
        # self.sa22 = SurfaceConv(npoint=point_num[2], in_channel=32 + 3, mlp=[32, 32, 64])
        # self.sa3 = SurfaceConv(npoint=point_num[2], in_channel=256 + 3, mlp=[256, 256, 512])
        self.sa4 = Merge(npoint=1, in_channel=256+3,  mlp=[256, 256])
        '''self.sa4 = SurfaceConv(npoint=point_num[3], in_channel=256 + 3, mlp=[256, 256, 256])
        self.sa5 = SurfaceConv(npoint=point_num[4], in_channel=256 + 3, mlp=[256, 512, 512]) 
        self.sa6 = SurfaceConv(npoint=point_num[5], in_channel=512 + 3, mlp=[512, 512, 1024])
        '''

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.0)
        #self.fc2 = nn.Linear(128, 128)
        #self.bn2 = nn.BatchNorm1d(128)
        #self.drop2 = nn.Dropout(0.0)
        self.fc3 = nn.Linear(128, num_classes)
        self.point_num = point_num

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
        '''l0_xyz, l0_points_res = self.sa0(xyz, None, local_coordinates_layer[0],
                                     neighbors_layer[0], data_idxes_layer[0])
        l0_xyz, l0_points = self.sa02(l0_xyz, l0_points_res, local_coordinates_layer[0],
                                      neighbors_layer[0], data_idxes_layer[0])
        l0_xyz, l0_points = self.sa03(l0_xyz, l0_points, local_coordinates_layer[0],
                                      neighbors_layer[0], data_idxes_layer[0])
        l0_xyz, l0_points = self.sa04(l0_xyz, l0_points, local_coordinates_layer[0],
                                      neighbors_layer[0], data_idxes_layer[0])
        res0 = F.relu(self.bn_res0(self.fc_res0(l0_points_res).permute(0, 2, 1)).permute(0, 2, 1))
        l0_points = res0 + l0_points'''

        # 2048 -> 512
        l1_xyz, l1_points = self.sa1(xyz, None, local_coordinates_layer[1],
                                     neighbors_layer[1], data_idxes_layer[1])
        # 512 -> 512
        '''l2_xyz, l2_points = self.sa12(l1_xyz, l1_points, local_coordinates_layer[2],
                                      neighbors_layer[2], data_idxes_layer[2])
        l2_xyz, l2_points = self.sa13(l2_xyz, l2_points, local_coordinates_layer[2],
                                      neighbors_layer[2], data_idxes_layer[2])
        l2_xyz, l2_points = self.sa14(l2_xyz, l2_points, local_coordinates_layer[2],
                                      neighbors_layer[2], data_idxes_layer[2])
        res1 = F.relu(self.bn_res1(self.fc_res1(l1_points).permute(0, 2, 1)).permute(0, 2, 1))
        l2_points = res1 + l2_points'''

        # 512 -> 128
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, local_coordinates_layer[3],
                                     neighbors_layer[3], data_idxes_layer[3])
        # l2_xyz, l2_points = self.sa21(l2_xyz, l2_points, local_coordinates_layer[3],
        #                             neighbors_layer[3], data_idxes_layer[3])
        # l2_xyz, l2_points = self.sa22(l2_xyz, l2_points, local_coordinates_layer[3],
        #                              neighbors_layer[3], data_idxes_layer[3])
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, local_coordinates_layer[2],
        #                            neighbors_layer[2], data_idxes_layer[2])
        l4_xyz, l4_points = self.sa4(l2_xyz, l2_points)
        '''
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points, local_coordinates_layer[4],
                                     neighbors_layer[4], data_idxes_layer[4])
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points, local_coordinates_layer[5],
                                     neighbors_layer[5], data_idxes_layer[5])'''
        # x = l3_points.mean(1).squeeze()
        x = l4_points.view(B, 256)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x


class SurfaceNet2(nn.Module):
    def __init__(self, num_classes=40, point_num=[2048, 512, 128]):
        super(SurfaceNet2, self).__init__()
        self.sa1 = SurfaceConv(npoint=point_num[0], in_channel=3, mlp=[16, 16, 32])
        #self.sa11 = SurfaceConv(npoint=point_num[0], in_channel=3+32, mlp=[32, 32, 32])
        #self.sa12 = SurfaceConv(npoint=point_num[0], in_channel=3+32, mlp=[32, 32, 64])
        self.sa2 = SurfaceConv(npoint=point_num[1], in_channel=32 + 3, mlp=[32, 64, 128])
        # self.sa21 = SurfaceConv2(npoint=point_num[1], in_channel=64 + 3, mlp=[64, 64, 128])
        self.sa3 = SurfaceConv(npoint=point_num[2], in_channel=128 + 3, mlp=[128, 256, 256])
        self.sa4 = Merge(npoint=1, in_channel=256+3,  mlp=[256, 512, 1024])
        '''self.sa4 = SurfaceConv(npoint=point_num[3], in_channel=256 + 3, mlp=[256, 256, 256])
        self.sa5 = SurfaceConv(npoint=point_num[4], in_channel=256 + 3, mlp=[256, 512, 512])
        self.sa6 = SurfaceConv(npoint=point_num[5], in_channel=512 + 3, mlp=[512, 512, 1024])
        '''
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)
        self.point_num = point_num

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

        l1_xyz, l1_points = self.sa1(xyz, None, local_coordinates_layer[0],
                                     neighbors_layer[0], data_idxes_layer[0])
        #l1_xyz, l1_points = self.sa11(xyz, l1_points, local_coordinates_layer[0],
        #                              neighbors_layer[0], data_idxes_layer[0])
        #l1_xyz, l1_points = self.sa12(xyz, l1_points, local_coordinates_layer[0],
        #                              neighbors_layer[0], data_idxes_layer[0])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, local_coordinates_layer[1],
                                     neighbors_layer[1], data_idxes_layer[1])
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, local_coordinates_layer[2],
                                     neighbors_layer[2], data_idxes_layer[2])
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        '''
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points, local_coordinates_layer[4],
                                     neighbors_layer[4], data_idxes_layer[4])
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points, local_coordinates_layer[5],
                                     neighbors_layer[5], data_idxes_layer[5])'''
        # x = l3_points.mean(1).squeeze()
        x = l4_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
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

