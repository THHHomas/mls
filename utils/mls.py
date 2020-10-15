from utils.pointconv_util import knn_point
import numpy as np
import torch
from time import time

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

eps = 1e-10


def visualize(x_coordinate, y_coordinate, f_i, parameter):
    x_range = torch.linspace(x_coordinate.min(), x_coordinate.max(), steps=40)
    y_range = torch.linspace(y_coordinate.min(), y_coordinate.max(), steps=40)
    grid_x, grid_y = torch.meshgrid(x_range, y_range)
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)
    new_base = torch.stack([torch.ones_like(grid_x), grid_x, grid_y, grid_x * grid_y,
                            grid_x ** 2, grid_y ** 2, grid_x ** 2 * grid_y, grid_y ** 2 * grid_x,
                            grid_x ** 3, grid_y ** 3])
    # predict_f_i = parameter.T.matmul(base).squeeze()
    grid_predict_f_i = parameter.T.matmul(new_base).squeeze()
    fig = plt.figure(1, figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_top_view()
    ax.plot_wireframe(grid_x.reshape(40, 40), grid_y.reshape(40, 40), grid_predict_f_i.reshape(40, 40), rstride=1,
                    cstride=1)
    ax.scatter3D(x_coordinate, y_coordinate, f_i, c='r', marker='o')
    plt.show()

def inverse_distance(vector, h=0.02):
    dis = (vector*vector).sum(1)
    return torch.exp(-dis/(h*h))


def MLS(points, data_idx, KNN_num):

    start_time = time()
    points = points.unsqueeze(0)
    neighbor_lists = knn_point(KNN_num, points, points).squeeze()
    points = points.squeeze()
    n_list = []
    error_rate_list = []
    projected_points = []
    rate = 0
    local_coordinate = []
    filtered_neighbor_list = []

    svd_time = 0
    solve_time = 0

    for i in data_idx:
        # Fit the plane
        r = points[i, :]
        neighbors = points[neighbor_lists[i]]
        relative_shift = neighbors - r
        theta_i = inverse_distance(relative_shift)
        A = torch.matmul(torch.matmul(relative_shift.T, torch.diag(theta_i)), relative_shift)
        local_start = time()
        res = torch.eig(A, eigenvectors=True)
        svd_time += time()-local_start
        init_n = res.eigenvectors[:, 2]
        # Powell iteration (optional)
        n_list.append(init_n)

        # Fit polynomial function
        nTr = init_n.matmul(r)
        x_axis = torch.tensor([0, 0, nTr/(init_n[2]+eps)]) - r
        x_axis = x_axis/(x_axis.norm()+eps)
        y_axis = init_n.cross(x_axis)
        f_i = relative_shift.matmul(init_n)
        local_vector = relative_shift - f_i.repeat(3,1).T*init_n
        x_coordinate = local_vector.matmul(x_axis)
        y_coordinate = local_vector.matmul(y_axis)
        # coordinate = torch.stack((x_coordinate, y_coordinate, f_i)).T
        # local_coordinate.append(coordinate)
        # minimize()

        base = torch.stack([torch.ones_like(x_coordinate), x_coordinate, y_coordinate, x_coordinate*y_coordinate,
                            x_coordinate**2, y_coordinate**2, x_coordinate**2*y_coordinate, y_coordinate**2*x_coordinate,
                            x_coordinate**3, y_coordinate**3])#, x_coordinate**4, x_coordinate**3*y_coordinate,
                            #x_coordinate**2*y_coordinate**2, x_coordinate*y_coordinate**3, y_coordinate**4])
        #B = base.matmul(torch.diag(theta_i)).matmul(base.T)
        B = base.matmul(torch.diag(theta_i)).matmul(base.T)
        lamd = 1e-10
        # B =B + torch.eye(B.shape[0])*lamd
        F = base.matmul(f_i*theta_i).unsqueeze(1)
        try:
            local_start = time()
            parameter, LU = torch.solve(F*1000, B*1000)
            # visualize(x_coordinate, y_coordinate, f_i, parameter)
            solve_time += time()-local_start

            # partition with radius
            predict_f_i = parameter.T.matmul(base).squeeze()
            temp = torch.stack((x_coordinate, y_coordinate, predict_f_i)).T
            local_coordinate.append(temp)
        except:
            print("no surface", i)
            local_coordinate.append(torch.stack((x_coordinate, y_coordinate, f_i)).T)

        #projected_point = (r + local_vector + predict_f_i.repeat(3, 1).T * init_n)
        #projected_point = projected_point[indices, :]
        #origin_projected_point = parameter[0]*init_n + r
        #projected_points.append(projected_point)
        # projected_points.append(origin_projected_point.unsqueeze(0).numpy())'''

    #projected_points = torch.cat(projected_points)
    local_coordinate = torch.cat(local_coordinate)
    # points2pcd(projected_points, "projected")
    # print("plane time is: ", time()-start_time)
    neighbor_lists = neighbor_lists[data_idx]
    if local_coordinate.shape[0] != len(data_idx)*KNN_num:
        print("hello")
    return neighbor_lists, local_coordinate


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    if N == npoint:
        centroids = torch.range(0, N-1).unsqueeze(0).repeat(B, 1)
        return centroids
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


if __name__ == "__main__":
    points = np.load("../sample.npy").transpose(1, 0)
    points = torch.from_numpy(points)
    filtered_neighbor_list, coordinate = MLS(points)
