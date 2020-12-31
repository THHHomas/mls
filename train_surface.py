import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader, load_data, load_normal_data
import datetime
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.mls import construct_MLS_multi, loadAuxiliaryInfo
from utils.utils import test, save_checkpoint, construct_planes
from model.pointconv import PointConvDensityClsSsg as PointConvClsSsg
from learning_based_surface.surfacenet import SurfaceNet, MainNet

import h5py

from tensorboardX import SummaryWriter
import shutil

if os.path.exists('log/exp'):
    shutil.rmtree('log/exp')
os.mkdir('log/exp')
writer1 = SummaryWriter('log/exp/1')
writer2 = SummaryWriter('log/exp/2')
writer3 = SummaryWriter('log/exp/3')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size in training')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--train_metric', type=str, default=True, help='whether evaluate on training dataset')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None, help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--with_normal', type=bool, default=True, help='whether the input containing point normal')
    parser.add_argument('--model_name', default='pointconv', help='model name')
    parser.add_argument('--random_sample', default=True, help='model name')
    parser.add_argument('--random_neighbor', default=False, help='model name')
    return parser.parse_args()


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
    classifier = MainNet(num_class, normal=args.with_normal, random_sp=args.random_sample, random_nb=args.random_neighbor).cuda()
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
    train_data, train_label, test_data, test_label = load_normal_data('./data/modelnet40_normal_resampled/')
    # logger.info('construct_MLS for train data...')
    # construct_MLS_multi(train_data, auxiliarypath, classifier.point_num)
    # logger.info('construct_MLS for test data...')
    # construct_MLS_multi(test_data, auxiliarypath, classifier.point_num, phase='test')

    train_neighbor_lists, train_data_idx_lists, train_local_axises = loadAuxiliaryInfo(auxiliarypath, classifier.point_num)
    logger.info("The number of training data is: %d",train_data.shape[0])
    logger.info("The number of test data is: %d", test_data.shape[0])
    trainDataset = ModelNetDataLoader(train_data, train_label, train_neighbor_lists, train_data_idx_lists, train_local_axises)
    test_neighbor_lists, test_data_idx_lists, test_local_axises = loadAuxiliaryInfo(auxiliarypath, classifier.point_num, phase='test')
    testDataset = ModelNetDataLoader(test_data, test_label, test_neighbor_lists, test_data_idx_lists, test_local_axises)
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchsize, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchsize, shuffle=False)

    '''TRANING'''
    logger.info('Start training...')
    first_time = True
    for epoch in range(start_epoch,args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        logger.info('Epoch %d (%d/%s):', global_epoch + 1, epoch + 1, args.epoch)

        scheduler.step()
        losses = 0
        lc_stds = 0
        lc_consistences = 0
        mean_correct = []
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target, neighbor_lists, data_idx_lists, local_axises = data
            # target = target[:, 0]
            # points = points.transpose(2, 1)
            if args.with_normal:
                points, target, neighbor_lists, data_idx_lists, local_axises = \
                    points.float().cuda(), target.cuda(), neighbor_lists.cuda(),  data_idx_lists.cuda(), local_axises.cuda()
            else:
                points, target, neighbor_lists, data_idx_lists, local_axises = \
                    points[:,:,0:3].float().cuda(), target.cuda(), neighbor_lists.cuda(), data_idx_lists.cuda(), local_axises.cuda()

            optimizer.zero_grad()
            classifier = classifier.train()
            pred, lc_std, lc_consistence = classifier(points, neighbor_lists, data_idx_lists, local_axises)
            loss = F.nll_loss(pred, target.long())
            total_loss = loss - lc_std - lc_consistence
            total_loss.backward()
            # print(classifier.axis_net.fc1.weight.grad)
            optimizer.step()
            global_step += 1
            losses += loss.item()
            lc_stds += lc_std.item()
            lc_consistences += lc_consistence.item()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
        if epoch%10 == 9:
            train_acc = test(classifier.eval(), trainDataLoader, args.with_normal) if args.train_metric else None
            writer3.add_scalar('quadratic', train_acc, global_step=epoch)
        acc = test(classifier.eval(), testDataLoader, args.with_normal)
        writer1.add_scalar('quadratic', losses / (batch_id + 1), global_step=epoch)
        writer2.add_scalar('quadratic', acc, global_step=epoch)

        print('\r Loss: %f' % float(losses/(batch_id+1)), 'lc_std: %f' % float(lc_stds/(batch_id+1)),
              'lc_consistences: %f' % float(lc_consistences/(batch_id+1)), 'acc: %f' %np.mean(mean_correct))
        logger.info('Loss: %.2f', losses/(batch_id+1))
        if args.train_metric and epoch % 10 == 9:
            print('Train Accuracy: %f' % train_acc)
            logger.info('Train Accuracy: %f', (train_acc))
        print('\r Test %s: %f   ***  %s: %f' % (blue('Accuracy'),acc, blue('Best Accuracy'),best_tst_accuracy))
        logger.info('Test Accuracy: %f  *** Best Test Accuracy: %f', acc, best_tst_accuracy)

        if (acc >= best_tst_accuracy) and epoch > 10:
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
