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
from learning_based_surface.surface_conv import SurfaceConv

class SurfaceNet(nn.Module):
    def __init__(self, num_classes=40, point_num=[512, 128, 1]):
        super(SurfaceNet, self).__init__()
        self.sa1 = SurfaceConv(npoint=point_num[0], in_channel=3, mlp=[64, 64, 128])
        self.sa2 = SurfaceConv(npoint=point_num[1], in_channel=128 + 3, mlp=[128, 128, 256])
        self.sa3 = SurfaceConv(npoint=point_num[2], in_channel=256 + 3, mlp=[256, 512, 1024])
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)
        self.point_num = point_num

    def forward(self, xyz, local_coordinates, neighbor_lists, data_idx_lists):
        B, _, _ = xyz.shape
        cid = 0
        l1_xyz, l1_points = self.sa1(xyz, None, local_coordinates[:, cid:self.point_num[0]*15, :].unsqueeze(),
                                     neighbor_lists[:, cid:self.point_num[0], :].unsqueeze(), data_idx_lists[:, cid:self.point_num[0]].unsqueeze())
        cid += self.point_num[0]
        l2_xyz, l2_points = self.sa2(xyz, None, local_coordinates[:, cid:self.point_num[1]*15, :].unsqueeze(),
                                     neighbor_lists[:, cid:self.point_num[1], :].unsqueeze(), data_idx_lists[:, cid:self.point_num[1]].unsqueeze())
        cid += self.point_num[1]
        l3_xyz, l3_points = self.sa3(xyz, None, local_coordinates[:, cid:self.point_num[2]*15, :].unsqueeze(),
                                     neighbor_lists[:, cid:self.point_num[2], :].unsqueeze(), data_idx_lists[:, cid:self.point_num[2]].unsqueeze())
        x = l3_points.view(B, 1024)
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

