import os
import h5py
import numpy as np
import json
import shutil
import torch

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def load_total(txt_path):
    with open(txt_path) as f:
        file_list = f.read().split()
    return file_list


def load_class(txt_path):
    with open(txt_path) as f:
        lists = f.read().split()
    num2str = dict()
    str2num = dict()
    num2class = dict()
    class2num = dict()
    for i in range(len(lists) // 2):
        num2str[lists[2 * i + 1]] = lists[2 * i]
        str2num[lists[2 * i]] = lists[2 * i + 1]
        num2class[lists[2 * i + 1]] = i
        class2num[i] = lists[2 * i + 1]
    class_num = np.zeros(16)
    for key in str2num.keys():
        value = str2num[key]
        cnum = seg_classes[key]
        index = num2class[value]
        class_num[index] = len(cnum)
    class2str = dict()
    for key in class2num.keys():
        class2str[key] = num2str[class2num[key]]

    return num2str, num2class, class2num


def load_shape(root="../f_shape/hdf5_data/", phase="train"):
    total_data = []
    total_label = []

    for i in range(16):
        total_data.append([])
        total_label.append([])

    setting = json.load(open(os.path.join(root, "catid_partid_to_overallid.json"), encoding='utf-8'))
    setting = dict([val, key] for key, val in setting.items())

    num2str, num2class, class2num = load_class(os.path.join(root, "all_object_categories.txt"))

    pid2local = np.zeros(50, dtype=np.int8)
    for i in range(50):
        num, local = setting[i].split("_")
        pid2local[i] = eval(local)

    points = []
    labels = []
    cates = []

    file_list = load_total(os.path.join(root, phase + "_hdf5_file_list.txt"))
    for h5_file in file_list:
        with h5py.File(os.path.join(root, h5_file), 'r') as data_file:
            label = np.array(data_file["label"][:])
            point = np.array(data_file["data"][:])
            pid = np.array(data_file["pid"][:])

            points.append(point)
            labels.append(pid)
            cates.append(label)
            for i in range(label.shape[0]):
                ss = label[i][0]
                total_data[ss].append(point[i])
                # total_label[ss].append(pid[i])
                sss = pid2local[pid[i]]
                total_label[ss].append(sss)
    points = np.concatenate(points, 0)
    labels = np.concatenate(labels, 0)
    cates = np.concatenate(cates, 0).squeeze()
    weight = torch.zeros(16)
    weight = weight.scatter_add(0, torch.from_numpy(cates).long(),
                                torch.ones_like(torch.from_numpy(cates), dtype=torch.float32))
    weight = weight / weight.sum()
    return total_data, total_label, num2class, weight


def load_shape_total(root="../f_shape/hdf5_data/", phase="train"):
    total_data = []
    total_label = []

    for i in range(16):
        total_data.append([])
        total_label.append([])

    setting = json.load(open(os.path.join(root, "catid_partid_to_overallid.json"), encoding='utf-8'))
    setting = dict([val, key] for key, val in setting.items())

    num2str, num2class, class2num = load_class(os.path.join(root, "all_object_categories.txt"))
    pid2local = np.zeros(50, dtype=np.int8)
    for i in range(50):
        num, local = setting[i].split("_")
        pid2local[i] = eval(local) - 1

    points = []
    labels = []
    cates = []

    file_list = load_total(os.path.join(root, phase + "_hdf5_file_list.txt"))
    for h5_file in file_list:
        with h5py.File(os.path.join(root, h5_file), 'r') as data_file:
            label = np.array(data_file["label"][:])
            point = np.array(data_file["data"][:])
            # pid = pid2local[np.array(data_file["pid"][:])]
            pid = pid2local[np.array(data_file["pid"][:])]

            points.append(point)
            labels.append(pid)
            cates.append(label)
            for i in range(label.shape[0]):
                ss = label[i][0]
                total_data[ss].append(point[i])
                # total_label[ss].append(pid[i])
                sss = pid2local[pid[i]]
                total_label[ss].append(sss)
    points = np.concatenate(points, 0)
    labels = np.concatenate(labels, 0)
    cates = np.concatenate(cates, 0).squeeze()
    weight = torch.zeros(16)
    weight = weight.scatter_add(0, torch.from_numpy(cates).long(),
                                torch.ones_like(torch.from_numpy(cates), dtype=torch.float32))
    weight = weight / weight.sum()
    return points, labels, cates, weight


def to_version2():
    folders = ['04099429', '03467517', '02773838', '02958343', '02691156', '04379243', '02954340', '03642806', \
               '03001627', '03624134', '03636649', '04225987', '03790512', '03948459', '03797390', '03261776']

    train_data, train_label, num2class, weight_train = load_shape(root="../f_shape/hdf5_data/")
    test_data, test_label, _, weight_test = load_shape(root="../f_shape/hdf5_data/", phase="test")
    val_data, val_label, _, weight_val = load_shape(root="../f_shape/hdf5_data/", phase="val")
    if os.path.exists("../f_shape/train"):
        shutil.rmtree("../f_shape/train")
    os.mkdir("../f_shape/train")
    if os.path.exists("../f_shape/test"):
        shutil.rmtree("../f_shape/test")
    os.mkdir("../f_shape/test")
    if os.path.exists("../f_shape/val"):
        shutil.rmtree("../f_shape/val")
    os.mkdir("../f_shape/val")

    for target_str in folders:
        datas = train_data[num2class[target_str]]
        labels = train_label[num2class[target_str]]
        with h5py.File("../f_shape/train/" + target_str + ".h5", 'w') as data_file:
            grp_point = data_file.create_group("points")
            grp_label = data_file.create_group("points_label")
            for idx, points in enumerate(datas):
                grp_point.create_dataset(str(idx), data=points)
                grp_label.create_dataset(str(idx), data=labels[idx])

        datas = test_data[num2class[target_str]]
        labels = test_label[num2class[target_str]]
        with h5py.File("../f_shape/test/" + target_str + ".h5", 'w') as data_file:
            grp_point = data_file.create_group("points")
            grp_label = data_file.create_group("points_label")
            for idx, points in enumerate(datas):
                grp_point.create_dataset(str(idx), data=points)
                grp_label.create_dataset(str(idx), data=labels[idx])

        datas = val_data[num2class[target_str]]
        labels = val_label[num2class[target_str]]
        with h5py.File("../f_shape/val/" + target_str + ".h5", 'w') as data_file:
            grp_point = data_file.create_group("points")
            grp_label = data_file.create_group("points_label")
            for idx, points in enumerate(datas):
                grp_point.create_dataset(str(idx), data=points)
                grp_label.create_dataset(str(idx), data=labels[idx])


if __name__ == "__main__":
    # points, labels, cates, weight = load_shape_total()
    '''cates = [4, 3, 0, 15, 15, 0, 11, 12, 3, 4, 6, 0]
    func = eval("to_categorical")
    y = func(torch.tensor(cates).cuda(), 16)
    print(y.shape)'''
    to_version2()

