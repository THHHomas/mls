# tag: openmp
# You can ignore the previous line.
# It's for internal testing of the cython documentation.

# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
cimport cython
from scipy.spatial import KDTree as kdtree
from cython.parallel import prange
eps =1e-10

ctypedef fused my_type:
    int
    double
    long long


# We declare our plain c function nogil
cdef my_type clip(my_type a, my_type min_value, my_type max_value) nogil:
    return min(max(a, min_value), max_value)


cdef inverse_distance(vector, h=0.1):
    dis = (vector*vector).sum(1)
    return np.exp(-dis/(h*h))


def MLS(points, list neighbor_lists):

    n_list = []
    projected_points = []
    rate = 0

    svd_time = 0
    solve_time = 0
    points_num = points.shape[0]
    for i in range(points_num):
        # Fit the plane
        r = points[i, :]
        neighbors = points[neighbor_lists[i],:]
        relative_shift = neighbors - r
        theta_i = inverse_distance(relative_shift)
        A = np.matmul(np.matmul(relative_shift.T, np.diag(theta_i)), relative_shift)
        eigvalue, eigvector = np.linalg.eig(A)
        init_n = eigvector[:, 2]
        # Powell iteration (optional)
        n_list.append(init_n)

        # Fit polynomial function
        nTr = np.matmul(init_n, r)
        x_axis = np.array([-nTr/(init_n[2]+eps), 0, nTr/(init_n[1]+eps)])
        x_axis = x_axis/(np.linalg.norm(x_axis)+eps)
        y_axis = np.cross(init_n, x_axis)
        f_i = np.matmul(relative_shift, init_n)
        local_vector = relative_shift - np.repeat(np.expand_dims(f_i, 0), 3, axis=0).T*init_n
        x_coordinate = np.matmul(local_vector, x_axis)
        y_coordinate = np.matmul(local_vector, y_axis)
        # minimize()
        base = np.stack([np.ones_like(x_coordinate), x_coordinate, y_coordinate, x_coordinate*y_coordinate,
                            x_coordinate**2, y_coordinate**2, x_coordinate**2*y_coordinate, y_coordinate**2*x_coordinate,
                            x_coordinate**3, y_coordinate**3])#, x_coordinate**4, x_coordinate**3*y_coordinate,
                            #x_coordinate**2*y_coordinate**2, x_coordinate*y_coordinate**3, y_coordinate**4])
        B = np.matmul(np.matmul(base, np.diag(theta_i)), base.T)
        F = np.expand_dims(np.matmul(base, f_i*theta_i), 1)
        try:
            parameter = np.linalg.solve(B, F)
        except:
            continue

        predict_f_i = np.matmul(parameter.T, base).squeeze()


        L_o = (np.abs(predict_f_i - f_i)).mean()
        f_std = np.std(f_i)
        if L_o < f_std:
            rate += 1

        indices = np.where(np.abs(predict_f_i - f_i) < f_std)[0]
        projected_point = (r + local_vector +  np.repeat(np.expand_dims(predict_f_i, 0), 3, axis=0).T * init_n)
        projected_point = projected_point[indices, :]
        origin_projected_point = parameter[0]*init_n + r
        projected_points.append(projected_point)
        projected_points.append(np.expand_dims(origin_projected_point, 0))

    projected_points = np.concatenate(projected_points)
    # points2pcd(projected_points, "projected")
    return 0
