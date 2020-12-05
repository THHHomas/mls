from utils.pointconv_util import knn_point
import numpy as np
import torch
from time import time

import matplotlib.pyplot as plt
from scipy.spatial import KDTree as kdtree
from mpl_toolkits.mplot3d import Axes3D

eps = 1e-10


def visualize(x_coordinate, y_coordinate, f_i, parameter):
    steps = 20
    x_range = torch.linspace(x_coordinate.min(), x_coordinate.max(), steps=steps)
    y_range = torch.linspace(y_coordinate.min(), y_coordinate.max(), steps=steps)
    grid_x, grid_y = torch.meshgrid(x_range, y_range)
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)
    new_base = torch.stack([torch.ones_like(grid_x), grid_x, grid_y, grid_x * grid_y,
                            grid_x ** 2, grid_y ** 2])  # , grid_x ** 2 * grid_y, grid_y ** 2 * grid_x,
    # grid_x ** 3, grid_y ** 3])
    # predict_f_i = parameter.T.matmul(base).squeeze()
    if parameter is not None:
        grid_predict_f_i = parameter.T.matmul(new_base).squeeze()
    else:
        grid_predict_f_i = torch.zeros_like(grid_x)

    fig = plt.figure(1, figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_top_view()
    ax.plot_wireframe(grid_x.reshape(steps, steps), grid_y.reshape(steps, steps), grid_predict_f_i.reshape(steps, steps), rstride=1,
                      cstride=1)
    ax.scatter3D(x_coordinate, y_coordinate, f_i, c='r', marker='o')
    plt.show()


def inverse_distance(vector, h=0.2):
    dis = (vector * vector).sum(-1)
    return torch.exp(-dis / (h * h))


def MLS(points, data_idx, KNN_num, radius=0.15):
    start_time = time()

    points = points.unsqueeze(0)
    neighbor_lists = knn_point(KNN_num, points, points).squeeze()
    points = points.squeeze()

    n_list = []
    chosen_data_num = KNN_num
    chosen_neighbor_lists = []
    error_rate_list = []
    projected_points = []
    rate = 0
    local_coordinate = []
    filtered_neighbor_list = []

    svd_time = 0
    solve_time = 0
    no_surface_num = 0
    for i in data_idx:
        # Fit the plane

        r = points[i, :]
        neighbors = points[neighbor_lists[i]]
        local_neighbor_index = neighbor_lists[i]
        if isinstance(local_neighbor_index, list):
            local_neighbor_index = torch.tensor(local_neighbor_index)
        relative_shift = neighbors - r
        theta_i = inverse_distance(relative_shift)
        A = torch.matmul(torch.matmul(relative_shift.T, torch.diag(theta_i)), relative_shift)
        local_start = time()
        res, _, _ = torch.svd(A)  # svd support batch
        svd_time += time() - local_start
        init_n = res[:, 2]
        # Powell iteration (optional)
        n_list.append(init_n)

        # Fit polynomial function
        nTr = init_n.matmul(r)
        x_axis = torch.tensor([0, 0, nTr / (init_n[2] + eps)]) - r
        x_axis = x_axis / (x_axis.norm() + eps)
        y_axis = init_n.cross(x_axis)
        f_i = relative_shift.matmul(init_n)
        local_vector = relative_shift - f_i.repeat(3, 1).T * init_n
        x_coordinate = local_vector.matmul(x_axis)
        y_coordinate = local_vector.matmul(y_axis)
        # coordinate = torch.stack((x_coordinate, y_coordinate, f_i)).T
        # local_coordinate.append(coordinate)
        # minimize()

        base = torch.stack([torch.ones_like(x_coordinate), x_coordinate, y_coordinate, x_coordinate * y_coordinate,
                            x_coordinate ** 2,
                            y_coordinate ** 2])  # , x_coordinate**2*y_coordinate, y_coordinate**2*x_coordinate,
        # x_coordinate**3, y_coordinate**3])#, x_coordinate**4, x_coordinate**3*y_coordinate,
        # x_coordinate**2*y_coordinate**2, x_coordinate*y_coordinate**3, y_coordinate**4])
        # B = base.matmul(torch.diag(theta_i)).matmul(base.T)
        B = base.matmul(torch.diag(theta_i)).matmul(base.T)
        lamd = 1e-5
        B = B + torch.eye(B.shape[0]) * lamd
        F = base.matmul(f_i * theta_i).unsqueeze(1)
        try:
            local_start = time()
            current_idx = i
            parameter, LU = torch.solve(F * 1000, B * 1000)

            solve_time += time() - local_start

            # partition with radius
            predict_f_i = parameter.T.matmul(base).squeeze()
            chosen_neighbor = np.where(abs(predict_f_i - f_i) < 100)[0]
            visualize(x_coordinate[chosen_neighbor], y_coordinate[chosen_neighbor], predict_f_i[chosen_neighbor],
                      parameter)

            temp = torch.stack((x_coordinate, y_coordinate, f_i)).T
            chosen_neighbor_list = []
            if len(chosen_neighbor) == 0:
                chosen_neighbor = torch.tensor([0] * chosen_data_num)
                temp = temp[chosen_neighbor]
                chosen_neighbor_list = local_neighbor_index[chosen_neighbor]
            elif len(chosen_neighbor) >= chosen_data_num:
                chosen_neighbor = chosen_neighbor[0:chosen_data_num]
                temp = temp[chosen_neighbor]
                chosen_neighbor_list = local_neighbor_index[chosen_neighbor]
            else:
                appended_data_num = chosen_data_num - len(chosen_neighbor)
                temp = temp[chosen_neighbor]
                appended_data_index = np.random.choice([x for x in range(len(chosen_neighbor))], appended_data_num)
                chosen_neighbor_list = torch.cat((local_neighbor_index[chosen_neighbor],
                                                  local_neighbor_index[appended_data_index]), 0)
                temp = torch.cat((temp, temp[appended_data_index, :]), 0)
            chosen_neighbor_lists.append(chosen_neighbor_list)
            local_coordinate.append(temp)
        except:
            no_surface_num += 1
            local_coordinate.append(torch.stack((x_coordinate, y_coordinate, f_i)).T[0:chosen_data_num])
            chosen_neighbor_lists.append(local_neighbor_index[0:chosen_data_num])

        # projected_point = (r + local_vector + predict_f_i.repeat(3, 1).T * init_n)
        # projected_point = projected_point[indices, :]
        # origin_projected_point = parameter[0]*init_n + r
        # projected_points.append(projected_point)
        # projected_points.append(origin_projected_point.unsqueeze(0).numpy())'''
    if no_surface_num > 0:
        print("no surface number is:", no_surface_num)
    # projected_points = torch.cat(projected_points)
    local_coordinate = torch.cat(local_coordinate)
    chosen_neighbor_lists = torch.stack(chosen_neighbor_lists)
    # points2pcd(projected_points, "projected")
    # print("plane time is: ", time()-start_time)
    if local_coordinate.shape[0] != len(data_idx) * chosen_data_num:
        print("hello")
    return chosen_neighbor_lists, local_coordinate


def MLS_batch(points, data_idx, KNN_num, radius=0.15):
    start_time = time()
    origin_points = points[:, 3:]
    points = points[:, 0:3]
    points = points[data_idx]
    origin_points = origin_points[data_idx]
    points = points.unsqueeze(0)
    neighbor_lists = knn_point(KNN_num, points, points).squeeze()
    '''points = points.squeeze()

    r = points  # NC
    neighbors = index_points(points.unsqueeze(0), neighbor_lists.unsqueeze(0)).squeeze()  # N, K, C
    relative_shift = neighbors - r.unsqueeze(1)  # N, K, C
    theta_i = inverse_distance(relative_shift)  # N, K

    A = torch.matmul((relative_shift * (theta_i.unsqueeze(2))).permute(0, 2, 1), relative_shift)
    U, S, V = torch.svd(A)
    # viewpoint at x axis
    init_n = U[:, :, 2].squeeze()  # N, C
    init_n_dir = origin_points  # N, C
    nTx = (init_n*init_n_dir).sum(1)
    # nTx = init_n.matmul(torch.tensor([1.0, 0.0, 0.0], device=init_n.device))
    dir = (nTx > 0).float() * 2 - 1
    init_n = init_n * (dir.unsqueeze(1))  # N C

    nTr = (init_n * r).sum(1)  # N

    # x_axis = torch.stack((torch.zeros_like(nTr), torch.zeros_like(nTr), nTr / (init_n[:, 2] + eps))).T - r  # N C
    x_axis = torch.stack([init_n[:, 2]/(init_n[:, 0] + eps), init_n[:, 2]/(init_n[:, 1] + eps), -2*torch.ones_like(nTr)]).T
    # dir = (x_axis[:, 2] < 0).float() * 2 - 1
    # x_axis = x_axis * (dir.unsqueeze(1))  # N C

    x_axis = x_axis / (x_axis.norm(dim=1, keepdim=True) + eps)
    y_axis = init_n.cross(x_axis)  # N C

    f_i = relative_shift.matmul(init_n.unsqueeze(2))  # N, K, 1
    local_vector = relative_shift - f_i.repeat(1, 1, 3) * init_n.unsqueeze(1)  # NKC
    x_coordinate = local_vector.matmul(x_axis.unsqueeze(2)).squeeze()  # NK
    y_coordinate = local_vector.matmul(y_axis.unsqueeze(2)).squeeze()  # NK
    local_coordinate = torch.stack((x_coordinate, y_coordinate, f_i.squeeze())).permute(1, 2, 0)  # NKC
    # local_coordinate = relative_shift
    base = torch.stack([torch.ones_like(x_coordinate), x_coordinate, y_coordinate, x_coordinate * y_coordinate,
                        x_coordinate ** 2,
                        y_coordinate ** 2])  # , x_coordinate**2*y_coordinate, y_coordinate**2*x_coordinate,
    # x_coordinate**3, y_coordinate**3])#, x_coordinate**4, x_coordinate**3*y_coordinate,
    # x_coordinate**2*y_coordinate**2, x_coordinate*y_coordinate**3, y_coordinate**4])
    base = base.permute(1, 0, 2)  # NDK
    B = torch.matmul(base * (theta_i.unsqueeze(1)), base.permute(0, 2, 1))
    F = base.matmul(f_i * (theta_i.unsqueeze(2)))
    parameter, LU = torch.solve(F, B)  # ND
    predict_f_i = torch.matmul(parameter.permute(0, 2, 1), base).squeeze()

    # local_coordinate = torch.stack((x_coordinate, y_coordinate, f_i)).permute(1, 2, 0)   # NKC
    # local_coordinate = x_axis.unsqueeze(1) * x_coordinate.unsqueeze(2) + y_axis.unsqueeze(1) * y_coordinate.unsqueeze(
    #    2) + init_n.unsqueeze(1) * predict_f_i.unsqueeze(2)
    # for i in range(10):
    #    visualize(x_coordinate[i], y_coordinate[i], predict_f_i[i], parameter[i])'''
    return neighbor_lists  # , local_coordinate, parameter


def MLS_grid(points, data_idx, KNN_num, radius=0.15):
    start_time = time()
    points = points[data_idx]
    points = points.unsqueeze(0)
    neighbor_lists = knn_point(KNN_num, points, points).squeeze()
    points = points.squeeze()

    r = points  # NC
    neighbors = index_points(points.unsqueeze(0), neighbor_lists.unsqueeze(0)).squeeze()  # N, K, C
    relative_shift = neighbors - r.unsqueeze(1)  # N, K, C
    theta_i = inverse_distance(relative_shift)  # N, K

    A = torch.matmul((relative_shift * (theta_i.unsqueeze(2))).permute(0, 2, 1), relative_shift)
    U, S, V = torch.svd(A)
    init_n = U[:, :, 2].squeeze()  # N, C
    nTx = init_n.matmul(torch.tensor([1.0, 0.0, 0.0], device=init_n.device))
    dir = (nTx > 0).float() * 2 - 1
    init_n = init_n * (dir.unsqueeze(1))  # N K C
    nTr = (init_n * r).sum(1)  # N
    x_axis = torch.stack((torch.zeros_like(nTr), torch.zeros_like(nTr), nTr / (init_n[:, 2] + eps))).T - r  # N C
    x_axis = x_axis / (x_axis.norm(dim=1, keepdim=True) + eps)
    y_axis = init_n.cross(x_axis)  # N C
    f_i = relative_shift.matmul(init_n.unsqueeze(2))  # N, K, 1
    local_vector = relative_shift - f_i.repeat(1, 1, 3) * init_n.unsqueeze(1)  # NKC
    x_coordinate = local_vector.matmul(x_axis.unsqueeze(2)).squeeze()  # NK
    y_coordinate = local_vector.matmul(y_axis.unsqueeze(2)).squeeze()  # NK
    # local_coordinate = torch.stack((x_coordinate, y_coordinate, f_i.squeeze())).permute(1, 2, 0)  # NKC
    # local_coordinate = relative_shift
    base = torch.stack([torch.ones_like(x_coordinate), x_coordinate, y_coordinate, x_coordinate * y_coordinate,
                        x_coordinate ** 2,
                        y_coordinate ** 2])  # , x_coordinate**2*y_coordinate, y_coordinate**2*x_coordinate,
    # x_coordinate**3, y_coordinate**3])#, x_coordinate**4, x_coordinate**3*y_coordinate,
    # x_coordinate**2*y_coordinate**2, x_coordinate*y_coordinate**3, y_coordinate**4])
    base = base.permute(1, 0, 2)  # NDK
    B = torch.matmul(base * (theta_i.unsqueeze(1)), base.permute(0, 2, 1))
    F = base.matmul(f_i * (theta_i.unsqueeze(2)))
    parameter, LU = torch.solve(F * 1000, B * 1000)  # ND
    predict_f_i = torch.matmul(parameter.permute(0, 2, 1), base).squeeze()
    # local_coordinate = x_axis.unsqueeze(1) * x_coordinate.unsqueeze(2) + y_axis.unsqueeze(1) * y_coordinate.unsqueeze(
    #    2) + init_n.unsqueeze(1) * predict_f_i.unsqueeze(2)
    # grid generation
    step_num = 4
    scope_scale = 2
    x_min, x_max = x_coordinate.min(1)[0]/scope_scale, x_coordinate.max(1)[0]/scope_scale  # N
    y_min, y_max = y_coordinate.min(1)[0]/scope_scale, y_coordinate.max(1)[0]/scope_scale  # N
    x_step = (x_max - x_min)/step_num
    y_step = (y_max - y_min)/step_num
    x_grid = torch.stack([x_min + x_step*step for step in range(step_num+1)]).permute(1, 0)  # NS
    y_grid = torch.stack([y_min + y_step*step for step in range(step_num+1)]).permute(1, 0)  # NS
    N = x_grid.shape[0]
    x_grid = x_grid.unsqueeze(1).repeat(1, step_num+1, 1).reshape(N, -1)  # N S*S
    y_grid = y_grid.unsqueeze(2).repeat(1, 1, step_num+1).reshape(N, -1)  # N S*S
    base_grid = torch.stack([torch.ones_like(x_grid), x_grid, y_grid, x_grid * y_grid, x_grid ** 2, y_grid ** 2])
    base_grid = base_grid.permute(1, 0, 2)  # ND S*S
    grid_f_i = torch.matmul(parameter.permute(0, 2, 1), base_grid).squeeze()
    # qqq0, www0 = f_i.min(), f_i.max()
    # qqq, www = grid_f_i.min(), grid_f_i.max()

    # local_coordinate = torch.stack((x_coordinate, y_coordinate, predict_f_i)).permute(1, 2, 0)   # NKC
    local_coordinate = x_axis.unsqueeze(1) * x_grid.unsqueeze(2) + y_axis.unsqueeze(1) * y_grid.unsqueeze(
        2) + init_n.unsqueeze(1) * grid_f_i.unsqueeze(2)  # NKC

    return local_coordinate


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    if N == npoint:
        centroids = torch.range(0, N - 1).unsqueeze(0).repeat(B, 1)
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


if __name__ == "__main__":
    points = np.load("../sample.npy").transpose(1, 0)
    points = torch.from_numpy(points)
    filtered_neighbor_list, coordinate = MLS(points)
