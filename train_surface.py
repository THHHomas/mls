import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader, load_data
import datetime
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.utils import test, save_checkpoint, construct_planes
from model.pointconv import PointConvDensityClsSsg as PointConvClsSsg
from learning_based_surface.surfacenet import SurfaceNet
from utils.mls import MLS, farthest_point_sample
import h5py

from tensorboardX import SummaryWriter
import shutil

if os.path.exists('log/exp'):
    shutil.rmtree('log/exp')
os.mkdir('log/exp')
writer1 = SummaryWriter('log/exp/1')
writer2 = SummaryWriter('log/exp/2')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size in training')
    parser.add_argument('--epoch',  default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='2', help='specify gpu device')
    parser.add_argument('--train_metric', type=str, default=True, help='whether evaluate on training dataset')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--model_name', default='pointconv', help='model name')
    return parser.parse_args()

def construct_MLS(points, path, point_nums, phase='train'):
    data_file = h5py.File(os.path.join(path, phase + '.h5'), 'w')

    points = torch.from_numpy(points)
    point_nums = point_nums
    KNN_nums = [25, 25]
    for i, point_num in enumerate(point_nums):
        KNN_num = KNN_nums[i]
        local_coordinates = []
        neighbor_lists = []
        data_idx_lists = []
        neighbor_grp = data_file.create_group("neighbor"+str(i))
        coordinate_grp = data_file.create_group("coordinate"+str(i))
        idx_grp = data_file.create_group("idx"+str(i))
        new_points =[]
        print("layer ", i)
        for idx in range(points.shape[0]):
            if idx%100 == 0:
                print("complete %f"%(float(idx)/float(points.shape[0])) )
            data = points[idx, :, :]
            data_idx = farthest_point_sample(data.unsqueeze(0), point_num).squeeze().long()
            filtered_neighbor_list, local_coordinate = MLS(data, data_idx, KNN_num)
            local_coordinates.append(local_coordinate)
            data_idx_lists.append(data_idx)
            neighbor_lists.append(filtered_neighbor_list)

            data = data[data_idx]
            new_points.append(data)

        points = torch.stack(new_points)
        neighbor_lists = torch.stack(neighbor_lists)
        local_coordinates = torch.stack(local_coordinates)
        data_idx_lists = torch.stack(data_idx_lists)
        neighbor_grp.create_dataset('neighbor_lists', data=neighbor_lists)
        coordinate_grp.create_dataset('local_coordinates', data=local_coordinates)
        idx_grp.create_dataset('data_idx_lists', data=data_idx_lists)


def loadAuxiliaryInfo(auxiliarypath, point_nums, phase='train'):
    data_file = h5py.File(os.path.join(auxiliarypath, phase + '.h5'), 'r')
    local_coordinates = []
    neighbor_lists = []
    data_idx_lists = []
    point_nums = point_nums
    for i, point_num in enumerate(point_nums):

        neighbor_grp = data_file["neighbor" + str(i)]["neighbor_lists"][:].astype(np.long)
        coordinate_grp = data_file["coordinate" + str(i)]["local_coordinates"][:].astype(np.float32)
        idx_grp = data_file["idx" + str(i)]["data_idx_lists"][:].astype(np.long)

        neighbor_lists.append(neighbor_grp)
        local_coordinates.append(coordinate_grp)
        data_idx_lists.append(idx_grp)
    neighbor_lists = np.concatenate(neighbor_lists, 1) #([B, N0, content], [B, N1, content], [B, N1, content])
    local_coordinates = np.concatenate(local_coordinates, 1)
    data_idx_lists = np.concatenate(data_idx_lists, 1)

    return torch.from_numpy(local_coordinates), torch.from_numpy(neighbor_lists), torch.from_numpy(data_idx_lists)


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    datapath = './data/ModelNet/'
    auxiliarypath = os.path.join(datapath, 'auxiliary')
    if ~os.path.exists(auxiliarypath):
        try:
            os.mkdir(auxiliarypath)
        except:
            pass

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sModelNet40-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''MODEL LOADING'''
    num_class = 40
    classifier = SurfaceNet(num_class).cuda()
    if args.pretrain is not None:
        print('Use pretrain model...')
        logger.info('Use pretrain model')
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    blue = lambda x: '\033[94m' + x + '\033[0m'

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    train_data, train_label, test_data, test_label = load_data(datapath, classification=True)
    '''logger.info('construct_MLS for train data...')
    construct_MLS(train_data, auxiliarypath, classifier.point_num)
    logger.info('construct_MLS for test data...')
    construct_MLS(test_data, auxiliarypath, classifier.point_num, phase='test')'''

    train_local_coordinates, train_neighbor_lists, train_data_idx_lists = loadAuxiliaryInfo(auxiliarypath, classifier.point_num)
    logger.info("The number of training data is: %d",train_data.shape[0])
    logger.info("The number of test data is: %d", test_data.shape[0])
    trainDataset = ModelNetDataLoader(train_data, train_label, train_local_coordinates, train_neighbor_lists, train_data_idx_lists)
    test_local_coordinates, test_neighbor_lists, test_data_idx_lists = loadAuxiliaryInfo(auxiliarypath, classifier.point_num, phase='test')
    testDataset = ModelNetDataLoader(test_data, test_label, test_local_coordinates, test_neighbor_lists, test_data_idx_lists)
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchsize, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchsize, shuffle=False)

    '''TRANING'''
    logger.info('Start training...')
    first_time = True
    for epoch in range(start_epoch,args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        logger.info('Epoch %d (%d/%s):' ,global_epoch + 1, epoch + 1, args.epoch)

        scheduler.step()
        losses = 0
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target, local_coordinates, neighbor_lists, data_idx_lists = data
            target = target[:, 0]
            # points = points.transpose(2, 1)
            points, target, local_coordinates, neighbor_lists, data_idx_lists = \
                points.cuda(), target.cuda(), local_coordinates.cuda(), neighbor_lists.cuda(), data_idx_lists.cuda()

            optimizer.zero_grad()
            classifier = classifier.train()
            pred = classifier(points, local_coordinates, neighbor_lists, data_idx_lists)
            loss = F.nll_loss(pred, target.long())

            loss.backward()
            optimizer.step()
            global_step += 1
            losses += loss.item()

        if epoch%10 == 0:
            train_acc = test(classifier.eval(), trainDataLoader) if args.train_metric else None
        acc = test(classifier, testDataLoader)
        writer1.add_scalar('quadratic', losses / (batch_id + 1), global_step=epoch)
        writer2.add_scalar('quadratic', acc, global_step=epoch)

        print('\r Loss: %f' % float(losses/(batch_id+1)))
        logger.info('Loss: %.2f', losses/(batch_id+1))
        if args.train_metric and epoch%10==0:
            print('Train Accuracy: %f' % train_acc)
            logger.info('Train Accuracy: %f', (train_acc))
        print('\r Test %s: %f   ***  %s: %f' % (blue('Accuracy'),acc, blue('Best Accuracy'),best_tst_accuracy))
        logger.info('Test Accuracy: %f  *** Best Test Accuracy: %f', acc, best_tst_accuracy)

        if (acc >= best_tst_accuracy) and epoch > 5:
            best_tst_accuracy = acc
            logger.info('Save model...')
            save_checkpoint(
                global_epoch + 1,
                train_acc if args.train_metric else 0.0,
                acc,
                classifier,
                optimizer,
                str(checkpoints_dir),
                args.model_name)
            print('Saving model....')
        global_epoch += 1
    print('Best Accuracy: %f'%best_tst_accuracy)

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
