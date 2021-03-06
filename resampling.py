# -*- coding: utf-8 -*-
# Smoothing and normal estimation based on polynomial reconstruction
# http://pointclouds.org/documentation/tutorials/resampling.php#moving-least-squares

import numpy as np
from scipy.spatial import KDTree as kdtree
import pcl
from utils.pointconv_util import knn_point
from utils.utils import points2pcd
import torch
from time import time
import os


from scipy.optimize import minimize, rosen
import random
eps =1e-10

def inverse_distance(vector, h=0.1):
    dis = (vector*vector).sum(1)
    return torch.exp(-dis/(h*h))


'''def powell_loss(x):
    """x[0:3] is n, x[3] is t"""
'''


def MLS(points):

    points2pcd(points, "origin")
    #x, y = np.mgrid[0:5, 2:8]
    #tree = kdtree(list(zip(x.ravel(), y.ravel())))
    #pts = np.array([[0, 0], [2.1, 2.9]])
    #res = tree.query(pts)
    start_time = time()

    #tree = kdtree(points)
    #neighbor_lists = tree.query_ball_tree(tree, r=0.1)

    #print("kdtree ball query time is :", time()-start_time)

    start_time = time()
    points = torch.from_numpy(points).unsqueeze(0)
    neighbor_lists = knn_point(25, points, points).squeeze()
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


        L_o = ((predict_f_i - f_i).abs()).mean()
        f_std = f_i.std()
        if L_o < f_std:
            rate += 1
        error_rate_list.append(L_o)

        indices = torch.where((predict_f_i - f_i).abs() < f_std)[0]
        filtered_neighbor_list.append(neighbor_lists[i, indices])
        projected_point = (r + local_vector + predict_f_i.repeat(3, 1).T * init_n)
        projected_point = projected_point[indices, :]
        origin_projected_point = parameter[0]*init_n + r
        projected_points.append(projected_point)
        # projected_points.append(origin_projected_point.unsqueeze(0).numpy())

    print("svd time is:", svd_time, ", solve time is:", solve_time)
    rate = rate / points.shape[0]
    projected_points = torch.cat(projected_points)
    local_coordinate = torch.cat(local_coordinate)
    # points2pcd(projected_points, "projected")
    print("plane time is: ", time()-start_time)
    return filtered_neighbor_list, local_coordinate


def main():
    # // Load input file into a PointCloud<T> with an appropriate type
    # pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    # // Load bun0.pcd -- should be available with the PCL archive in test
    # pcl::io::loadPCDFile ("bun0.pcd", *cloud);
    points = np.load("sample.npy").transpose(1, 0)
    filtered_neighbor_list, coordinate = MLS(points)
    cloud = pcl.load('pcd/origin.pcd')
    print('cloud(size) = ' + str(cloud.size))

    # // Create a KD-Tree
    # pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

    start = time()
    tree = cloud.make_kdtree()
    # tree = cloud.make_kdtree_flann()
    # blankCloud = pcl.PointCloud()
    # tree = blankCloud.make_kdtree()

    # // Output has the PointNormal type in order to store the normals calculated by MLS
    # pcl::PointCloud<pcl::PointNormal> mls_points;
    # mls_points = pcl.PointCloudNormal()
    # // Init object (second point type is for the normals, even if unused)
    # pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    # mls.setComputeNormals (true);
    #
    # // Set parameters
    # mls.setInputCloud (cloud);
    # mls.setPolynomialFit (true);
    # mls.setSearchMethod (tree);
    # mls.setSearchRadius (0.03);
    #
    # // Reconstruct
    # mls.process (mls_points);
    mls = cloud.make_moving_least_squares()
    # print('make_moving_least_squares')
    mls.set_Compute_Normals(True)
    mls.set_polynomial_fit(True)
    mls.set_polynomial_order(3)
    mls.set_Search_Method(tree)
    mls.set_search_radius(0.1)
    print('set parameters')
    mls_points = mls.process()
    projected = np.asarray(mls_points)
    points2pcd(projected, "projected_mls")
    print("standard time is: ", time() - start)
    # Save output
    # pcl::io::savePCDFile ("bun0-mls.pcd", mls_points);
    pcl.save_PointNormal(mls_points, 'bun0-mls.pcd')


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
