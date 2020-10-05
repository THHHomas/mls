from utils.pointconv_util import knn_point
import numpy as np
import torch
from time import time


eps = 1e-10


def inverse_distance(vector, h=0.1):
    dis = (vector*vector).sum(1)
    return torch.exp(-dis/(h*h))


def MLS(points):

    start_time = time()
    points = points.unsqueeze(0)
    neighbor_lists = knn_point(15, points, points).squeeze()
    points = points.squeeze()
    n_list = []
    error_rate_list = []
    projected_points = []
    rate = 0
    local_coordinate = []
    filtered_neighbor_list = []

    svd_time = 0
    solve_time = 0

    for i in range(points.shape[0]):
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
        x_axis = torch.tensor([-nTr/(init_n[2]+eps), 0, nTr/(init_n[1]+eps)])
        x_axis = x_axis/(x_axis.norm()+eps)
        y_axis = init_n.cross(x_axis)
        f_i = relative_shift.matmul(init_n)
        local_vector = relative_shift - f_i.repeat(3,1).T*init_n
        x_coordinate = local_vector.matmul(x_axis)
        y_coordinate = local_vector.matmul(y_axis)
        coordinate = torch.stack((x_coordinate, y_coordinate)).T
        local_coordinate.append(coordinate)
        # minimize()
        base = torch.stack([torch.ones_like(x_coordinate), x_coordinate, y_coordinate, x_coordinate*y_coordinate,
                            x_coordinate**2, y_coordinate**2, x_coordinate**2*y_coordinate, y_coordinate**2*x_coordinate,
                            x_coordinate**3, y_coordinate**3])#, x_coordinate**4, x_coordinate**3*y_coordinate,
                            #x_coordinate**2*y_coordinate**2, x_coordinate*y_coordinate**3, y_coordinate**4])
        B = base.matmul(torch.diag(theta_i)).matmul(base.T)
        F = base.matmul(f_i*theta_i).unsqueeze(1)
        try:
            local_start = time()
            parameter, LU = torch.solve(F, B)
            solve_time += time()-local_start
        except:
            continue

        predict_f_i = parameter.T.matmul(base).squeeze()

        f_std = f_i.std()
        indices = torch.where((predict_f_i - f_i).abs() < f_std)[0]
        filtered_neighbor_list.append(neighbor_lists[i, indices])
        #projected_point = (r + local_vector + predict_f_i.repeat(3, 1).T * init_n)
        #projected_point = projected_point[indices, :]
        #origin_projected_point = parameter[0]*init_n + r
        #projected_points.append(projected_point)
        # projected_points.append(origin_projected_point.unsqueeze(0).numpy())

    #print("svd time is:", svd_time, ", solve time is:", solve_time)
    rate = rate / points.shape[0]
    #projected_points = torch.cat(projected_points)
    local_coordinate = torch.cat(local_coordinate)
    # points2pcd(projected_points, "projected")
    # print("plane time is: ", time()-start_time)
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
