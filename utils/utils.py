
import torch
import heapq
import MDAnalysis.lib.transformations as MDA
import numpy as np
# import pytorch3d.transforms
import pandas as pd
import torch.nn.functional as F
import os
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from utils.loss import PointDistance
from utils.loader import SSFrameDataset
import numpy as np
from sklearn.metrics import f1_score,accuracy_score


def pair_samples(num_samples, num_pred, single_interval):
    """
    :param num_samples:
    :param num_pred: number of the (last) samples, for which the transformations are predicted
        For each "pred" frame, pairs are formed with every one previous frame 
    :param single_interval: 0 - use all interval predictions
                            1,2,3,... - use only specific intervals
    """

    if single_interval == 0:
        return torch.tensor([[n0,n1] for n1 in range(num_samples-num_pred,num_samples) for n0 in range(n1)])
    else:
        return torch.tensor([[n1-single_interval,n1] for n1 in range(single_interval,num_samples,single_interval) ])


def type_dim(label_pred_type, num_points=None, num_pairs=1):
    type_dim_dict = {
        "transform": 12,
        "parameter": 6,
        "point": num_points*3
    }
    return type_dim_dict[label_pred_type] * num_pairs  # num_points=self.image_points.shape[1]), num_pairs=self.pairs.shape[0]


def reference_image_points(image_size, density=2):
    """
    :param image_size: (x, y), used for defining default grid image_points
    :param density: (x, y), point sample density in each of x and y, default n=2
    """
    if isinstance(density,int):
        density=(density,density)

    # image_points = torch.cartesian_prod(
    #     torch.linspace(-image_size[0]/2,image_size[0]/2,density[0]),
    #     torch.linspace(-image_size[1]/2,image_size[1]/2,density[1])
    #     ).t()  # transpose to 2-by-n

    image_points = torch.cartesian_prod(
        torch.linspace(0, image_size[0] , density[0]),
        torch.linspace(0, image_size[1], density[1])
    ).t()
    
    image_points = torch.cat([
        image_points, 
        torch.zeros(1,image_points.shape[1])*image_size[0]/2,
        torch.ones(1,image_points.shape[1])
        ], axis=0)
    
    return image_points



def tran624_np(parameter,seq= 'rzyx'):
    '''
    for numpy use: 6 parameter --> 4*4 transformation matrix
    :param parameter: numpy type, [6], angle_x, angle_y, angle_z, x, y, z
    :param seq: e.g.,'rxyz'
    :return: transform: 4*4 transformation matrix
    '''

    transform =MDA.euler_matrix(parameter[0], parameter[1], parameter[2], seq)

    transform[0,3]=parameter[3]
    transform[1,3]=parameter[4]
    transform[2,3]=parameter[5]
    transform[3,3]=1

    return transform

def tran426_np(transform,seq= 'rzyx'):
    '''
    :param transform: numpy type, 4*4 transformation matrix
    :param seq: e.g.,'rxyz'
    :return: parameter: [6], angle_x, angle_y, angle_z, x, y, z
    '''
    parameter = np.zeros(6)
    parameter[0:3]=MDA.euler_from_matrix(transform,seq)
    parameter[3]=transform[0,3]
    parameter[4]=transform[1,3]
    parameter[5]=transform[2,3]

    return parameter

# def tran624_tensor(parameter, seq = 'ZYX'):
#     '''
#     # for tensor use: 6 parameter --> 4*4 transformation matrix
#     # this can preserve grad in tensor which need to be backforward
#     :param parameter: tensor type, [6], angle_x, angle_y, angle_z, x, y, z
#     :param seq: e.g.,'XYZ'
#     :return: transform: 4*4 transformation matrix
#     '''
#     Rotation = pytorch3d.transforms.euler_angles_to_matrix(parameter[0:3], seq)
#     transform = torch.row_stack((torch.column_stack((Rotation, torch.t(parameter[3:6]))), torch.tensor([0, 0, 0, 1])))

#     return transform

# def tran426_tensor(transform,seq = 'ZYX'):
#     '''
#     # this can preserve grad in tensor which need to be backforward
#     :param transform:tensor type, 4*4 transformation matrix
#     :param seq: e.g.,'XYZ'
#     :return: parameter: [6], angle_x, angle_y, angle_z, x, y, z
#     '''
#     Rotation = pytorch3d.transforms.matrix_to_euler_angles(transform[0:3, 0:3], seq)
#     parameter = torch.cat((Rotation, transform[0:3, 3]))
#     return parameter




def sample_adjacent_pair(start, step, data_pairs):
    adjacent_pair = []
    while 1:
        adjacent_pair.append(start)
        start = start + step
        step = step + 1
        if start >= data_pairs.shape[0]:
            break
    return adjacent_pair # data_pairs[adjacent_pair]

def add_scalars(writer,epoch, loss_dists,preds_dist_all_train, label_dist_all_train,preds_dist_all_val,label_dist_all_val,data_pairs,opt,data_pairs_samples_index):
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss']
    train_epoch_dist = loss_dists['train_epoch_dist']
    epoch_loss_val = loss_dists['epoch_loss_val']
    epoch_dist_val = loss_dists['epoch_dist_val']

    # train_epoch_loss_total = loss_dists['train_epoch_loss_total']
    # epoch_loss_val_total = loss_dists['val_epoch_loss_total']


    for i in range(len(preds_dist_all_train)):
        dist_train=(((preds_dist_all_train[str(i)]-label_dist_all_train[str(i)])**2).sum(dim=1).sqrt().mean())
        dist_val=(((preds_dist_all_val[str(i)]-label_dist_all_val[str(i)])**2).sum(dim=1).sqrt().mean())
        writer.add_scalars('accumulated_dists', {'train_%d' % i: dist_train.item()}, epoch)
        writer.add_scalars('accumulated_dists', {'val_%d' % i: dist_val.item()}, epoch)

    writer.add_scalars('loss_rec', {'train_loss': train_epoch_loss},epoch)
    writer.add_scalars('loss_rec', {'val_loss': epoch_loss_val},epoch)
    writer.add_scalars('dist_rec', {'train_dist': train_epoch_dist.sum().item()}, epoch)
    writer.add_scalars('dist_rec', {'val_dist': epoch_dist_val.sum().item()}, epoch)

    # writer.add_scalars('loss_total', {'train_loss_total': train_epoch_loss_total},epoch)
    # writer.add_scalars('loss_total', {'val_loss_total': epoch_loss_val_total},epoch)


    # add dists to scalars, seperatatly dist and each dist is divided by the interval
    # random selected a number of dist to monitor
    # random_sample=sorted(random.sample(range(len(train_epoch_dist)),10))
         
    # for i in random_sample:
    #     writer.add_scalars('dists', {'train_%s' % str(data_pairs[i][0].item())+'_'+str(data_pairs[i][1].item()): train_epoch_dist[i]/(data_pairs[i][1]-data_pairs[i][0])},epoch)
    #     writer.add_scalars('dists', {'val_%s' % str(data_pairs[i][0].item())+'_'+str(data_pairs[i][1].item()): epoch_dist_val[i]/(data_pairs[i][1]-data_pairs[i][0])},epoch)

        








def add_scalars_loss(writer, epoch, loss_dists):
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss']
    train_epoch_dist = loss_dists['train_epoch_dist']
    epoch_loss_val = loss_dists['epoch_loss_val']
    epoch_dist_val = loss_dists['epoch_dist_val']
    writer.add_scalars('loss_average_dists', {'train_loss': train_epoch_loss.item(), 'val_loss': epoch_loss_val.item()},epoch)
    writer.add_scalars('loss_average_dists',{'train_dists': train_epoch_dist.item(), 'val_dists': epoch_dist_val.item()}, epoch)


def str2list(string):
    string = ''.join(string)
    string = string[1:-1]
    token = string.split(',')
    list = [int(token_i) for token_i in token]
    return list

def load_json(opt,json_fn):
    if os.path.isfile(opt.SAVE_PATH+'/'+json_fn+".json"):
        with open(opt.SAVE_PATH + '/' + json_fn+".json", "r", encoding='utf-8') as f:
            rmse_intervals_each_scan= json.load(f)
    else:
        rmse_intervals_each_scan= {}

    return rmse_intervals_each_scan



