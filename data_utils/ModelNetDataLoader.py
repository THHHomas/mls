import numpy as np
import warnings
import h5py
from time import time
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def num_per_class_cal(train_label):
    train_label_vis = torch.from_numpy(train_label).squeeze().long()
    num_per_class = torch.zeros(40)
    others = torch.ones_like(train_label_vis).float()
    num_per_class.scatter_add_(0, train_label_vis, others)
    return num_per_class.numpy()

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = []
    return (data, label, seg)

def svd_points(data):
    # data: B, N, C
    data = torch.from_numpy(data)
    normal = data[:, :, 3:]
    points = data[:, :, 0:3]
    A = torch.matmul(points.permute(0,2,1), points)
    E, U = torch.symeig(A, eigenvectors=True)  # U->BCC
    new_point = torch.matmul(points, U)
    normal = torch.matmul(normal, U)
    new_point = torch.cat([new_point, normal], 2)
    return new_point.numpy()

def load_normal_data(dir):
    data_train0, label_train0,Seglabel_train0  = load_h5(dir + 'train.h5')
    data_test0, label_test0,Seglabel_test0 = load_h5(dir + 'test.h5')
    # data_test0 = svd_points(data_test0)
    # data_train0 = svd_points(data_train0)
    return data_train0, label_train0, data_test0, label_test0

def load_data(dir, classification=False):
    data_train0, label_train0,Seglabel_train0  = load_h5(dir + 'ply_data_train0.h5')
    data_train1, label_train1,Seglabel_train1 = load_h5(dir + 'ply_data_train1.h5')
    data_train2, label_train2,Seglabel_train2 = load_h5(dir + 'ply_data_train2.h5')
    data_train3, label_train3,Seglabel_train3 = load_h5(dir + 'ply_data_train3.h5')
    data_train4, label_train4,Seglabel_train4 = load_h5(dir + 'ply_data_train4.h5')
    data_test0, label_test0,Seglabel_test0 = load_h5(dir + 'ply_data_test0.h5')
    data_test1, label_test1,Seglabel_test1 = load_h5(dir + 'ply_data_test1.h5')
    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])
    train_Seglabel = np.concatenate([Seglabel_train0,Seglabel_train1,Seglabel_train2,Seglabel_train3,Seglabel_train4])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    test_Seglabel = np.concatenate([Seglabel_test0,Seglabel_test1])

    #num_per_class_train = num_per_class_cal(train_label)
    #num_per_class_test = num_per_class_cal(test_label)
    #plt.bar([x for x in range(40)] ,num_per_class_test)
    #plt.title('test data num per class')
    #plt.show()


    if classification:
        return train_data, train_label, test_data, test_label
    else:
        return train_data, train_Seglabel, test_data, test_Seglabel

class ModelNetDataLoader(Dataset):
    def __init__(self, data, labels, neighbor_lists=None, data_idx_lists=None, local_axises=None, rotation=None, npoint=1024):
        self.data = data
        self.labels = labels
        self.npoint = npoint
        self.rotation = rotation
        self.neighbor_lists = neighbor_lists
        self.data_idx_lists = data_idx_lists
        self.local_axises = local_axises

    def __len__(self):
        return len(self.data)

    def rotate_point_cloud_by_angle(self, data, rotation_angle):
        """
        Rotate the point cloud along up direction with certain angle.
        :param batch_data: Nx3 array, original batch of point clouds
        :param rotation_angle: range of rotation
        :return:  Nx3 array, rotated batch of point clouds
        """
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]], dtype = np.float32)
        data[:, 0:3] = np.dot(data[:, 0:3], rotation_matrix)
        return data

    def __getitem__(self, index):
        if self.rotation is not None:
            pointcloud = self.data[index]
            angle = np.random.randint(self.rotation[0], self.rotation[1]) * np.pi / 180
            pointcloud = self.rotate_point_cloud_by_angle(pointcloud, angle)
            if self.neighbor_lists is not None:
                return pointcloud, self.labels[index], self.neighbor_lists[index], self.data_idx_lists[index], self.local_axises[index]
            else:
                return pointcloud[:,0:3], self.labels[index]
        else:
            if self.neighbor_lists is not None:
                return self.data[index], self.labels[index], self.neighbor_lists[index], self.data_idx_lists[index], self.local_axises[index]
            else:
                return self.data[index], self.labels[index]