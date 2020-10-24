# -*- coding: utf-8 -*-
# Spatial Partitioning and Search Operations with Octrees
# http://pointclouds.org/documentation/tutorials/octree.php#octree-search

import pcl
import numpy as np
import random
from utils.mls import MLS, farthest_point_sample, MLS_batch, MLS_grid
import torch
import time

def main():
    points = np.load("sample.npy").transpose(1, 0)
    cloud = pcl.load('pcd/origin.pcd')
    print('cloud(size) = ' + str(cloud.size))

    # // Create a KD-Tree
    # pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

    start = time.time()
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
    mls.set_polynomial_order(2)
    mls.set_Search_Method(tree)
    mls.set_search_radius(0.25)
    print('set parameters')
    mls_points = mls.process()
    print("standard time is: ", time.time() - start)


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
    points = np.load("./sample.npy").transpose(1, 0)
    points = torch.from_numpy(points)
    data_idx = farthest_point_sample(points.unsqueeze(0), 512).squeeze().long()
    # points = points[data_idx]
    # data_idx = farthest_point_sample(points.unsqueeze(0), 128).squeeze().long()
    ss = time.time()
    data_idx = torch.arange(0, points.shape[0], device=points.device)
    for i in range(1):
        filtered_neighbor_list, coordinate = MLS_batch(points, data_idx, 32)
    print("comsuming time is:", (time.time()-ss)/10)

