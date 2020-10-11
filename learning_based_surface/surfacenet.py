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
from learning_based_surface.surface_conv import SurfaceConv, Merge

# [512, 256, 128, 64, 32, 16] auxiliary6
# [512, 128, 32] auxiliary3
class SurfaceNet(nn.Module):
    def __init__(self, num_classes=40, point_num=[512, 128, 32]):
        super(SurfaceNet, self).__init__()
        self.sa1 = SurfaceConv(npoint=point_num[0], in_channel=3, mlp=[16, 16, 32])
        self.sa2 = SurfaceConv(npoint=point_num[1], in_channel=32 + 3, mlp=[32, 64, 128])
        # self.sa3 = SurfaceConv(npoint=point_num[2], in_channel=256 + 3, mlp=[256, 256, 512])
        self.sa4 = Merge(npoint=1, in_channel=128+3,  mlp=[128, 128, 256])
        '''self.sa4 = SurfaceConv(npoint=point_num[3], in_channel=256 + 3, mlp=[256, 256, 256])
        self.sa5 = SurfaceConv(npoint=point_num[4], in_channel=256 + 3, mlp=[256, 512, 512])
        self.sa6 = SurfaceConv(npoint=point_num[5], in_channel=512 + 3, mlp=[512, 512, 1024])
        '''
        self.fc1 = nn.Linear(256, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_classes)
        self.point_num = point_num

    def forward(self, xyz, local_coordinates, neighbors, data_idxes):
        B, N, C = xyz.shape
        KNN = 15
        # seperate the neighbor information into layers
        neighbors_layer = []
        local_coordinates_layer = []
        data_idxes_layer = []
        cid = 0
        for point_num in self.point_num:
            local_coordinates_layer.append(local_coordinates[:, cid*KNN:(cid+point_num)*KNN, :].squeeze())
            neighbors_layer.append(neighbors[:, cid:cid+point_num, :].squeeze())
            data_idxes_layer.append(data_idxes[:, cid:cid+point_num].squeeze())
            cid += point_num

        l1_xyz, l1_points = self.sa1(xyz, None, local_coordinates_layer[0],
                                     neighbors_layer[0], data_idxes_layer[0])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, local_coordinates_layer[1],
                                     neighbors_layer[1], data_idxes_layer[1])
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

