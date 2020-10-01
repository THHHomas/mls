from MLS import MLS
import numpy as np
import time
import torch
from scipy.spatial import KDTree as kdtree
from utils.pointconv_util import knn_point

points = np.random.random((3000, 3)).astype(np.float32)


#tree = kdtree(points)
#neighbor_lists = tree.query_ball_tree(tree, r=0.1)
neighbor_lists = knn_point(25, torch.from_numpy(points).unsqueeze(0),
                               torch.from_numpy(points).unsqueeze(0)).squeeze().numpy().tolist()
s = time.time()
num = 1
for i in range(num):
    res = MLS(points, neighbor_lists)
print("loop time:", (time.time()-s)/num)


'''
array_1 = np.random.uniform(0, 1000, size=(3000, 2000)).astype(np.intc)
array_2 = np.random.uniform(0, 1000, size=(3000, 2000)).astype(np.intc)
s= time.time()
for i in range(10):
    compute(array_1, array_2, 2, 3, 4, set([]))
print("Cython time:", (time.time()-s)/10)


def clip(a, min_value, max_value):
    return min(max(a, min_value), max_value)


def compute_python(array_1, array_2, a, b, c):
    """
    This function must implement the formula
    np.clip(array_1, 2, 10) * a + array_2 * b + c

    array_1 and array_2 are 2D.
    """
    x_max = array_1.shape[0]
    y_max = array_1.shape[1]

    assert array_1.shape == array_2.shape

    result = np.zeros((x_max, y_max), dtype=array_1.dtype)

    for x in range(x_max):
        for y in range(y_max):
            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result[x, y] = tmp + c

    return result

s = time.time()
for i in range(10):
    print(i)
    compute_python(np.clip(array_1, 2, 10), array_2, 2, 3, 4)

print("python for time:", (time.time()-s)/10)
'''