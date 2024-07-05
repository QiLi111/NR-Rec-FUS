
import torch
import heapq
import MDAnalysis.lib.transformations as MDA
import numpy as np
# import pytorch3d.transforms
import pandas as pd
import os
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from utils.loss import PointDistance
from utils.loader import SSFrameDataset
from torch import linalg as LA


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


def add_scalars_rec_volume(writer,epoch, loss_dists):
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss_all']
    train_epoch_dist = loss_dists['train_epoch_dist']
    epoch_loss_val = loss_dists['epoch_loss_val_all']
    epoch_dist_val = loss_dists['epoch_dist_val']
    train_epoch_loss_reg = loss_dists['train_epoch_loss_reg']
    epoch_loss_val_reg = loss_dists['epoch_loss_val_reg']
    train_epoch_loss_rec = loss_dists['train_epoch_loss_rec']
    epoch_loss_val_rec = loss_dists['epoch_loss_val_rec']


    
    writer.add_scalars('loss_rec_all', {'train_loss': train_epoch_loss},epoch)
    writer.add_scalars('loss_rec_all', {'val_loss': epoch_loss_val},epoch)
    writer.add_scalars('dist', {'train_dist': train_epoch_dist}, epoch)
    writer.add_scalars('dist', {'val_dist': epoch_dist_val}, epoch)

    writer.add_scalars('loss_rec_volume', {'train_loss': train_epoch_loss_reg},epoch)
    writer.add_scalars('loss_rec_volume', {'val_loss': epoch_loss_val_reg},epoch)

    writer.add_scalars('loss_rec', {'train_loss': train_epoch_loss_rec},epoch)
    writer.add_scalars('loss_rec', {'val_loss': epoch_loss_val_rec},epoch)

    

def add_scalars_reg(writer,epoch, loss_dists,model_name):
    # loss in training and val
    train_epoch_loss = loss_dists['train_epoch_loss_reg_only']
    epoch_loss_val = loss_dists['epoch_loss_val_reg_only']

    writer.add_scalars('loss_reg_only', {'train_loss_'+model_name: train_epoch_loss},epoch)
    writer.add_scalars('loss_reg_only', {'val_loss_'+model_name: epoch_loss_val},epoch)

def add_scalars_reg_T(writer,epoch, loss_dists,model_name):
    # loss in training and val
    train_epoch_loss = loss_dists['train_dist_reg_T']
    epoch_loss_val = loss_dists['val_dist_reg_T']

    writer.add_scalars('T_dist_in_R', {'train_loss_'+model_name: train_epoch_loss},epoch)
    writer.add_scalars('T_dist_in_R', {'val_loss_'+model_name: epoch_loss_val},epoch)

def add_scalars_wrap_dist(writer,epoch, loss_dists,model_name):
    # loss in training and val
    train_epoch_loss = loss_dists['train_wrap_dist']
    epoch_loss_val = loss_dists['val_wrap_dist']

    writer.add_scalars('wrap_dist_'+model_name, {'train_wrap_dist_'+model_name: train_epoch_loss},epoch)
    writer.add_scalars('wrap_dist_'+model_name, {'val_wrap_dist_'+model_name: epoch_loss_val},epoch)


def compute_plane_normal(pts):
    # Create vectors from the points
    vector1 = pts[:,:,:,1]-pts[:,:,:,0]
    vector2 = pts[:,:,:,2]-pts[:,:,:,0]
    
    # Compute the cross product of vector1 and vector2
    cross_product = torch.linalg.cross(vector1, vector2)
    
    # Normalize the cross product to get the plane's normal vector
    matrix_norm = LA.norm(cross_product, dim= 2)
    normal_vector = cross_product / matrix_norm.unsqueeze(2).repeat(1, 1, 3)
    
    return normal_vector



def angle_between_planes(normal_vector1, normal_vector2):
    # compute the cos value between two norm vertorc of two planes
   
    # Calculate the dot product of the two normal vectors
    normal_vector1 = normal_vector1.to(torch.float)
    normal_vector2 = normal_vector2.to(torch.float)

    dot_product = torch.sum(normal_vector1 * normal_vector2, dim=(2))

    # dot_product = torch.dot(normal_vector1, normal_vector2)
    
    # Calculate the magnitudes of the two normal vectors
   
    magnitude1 = LA.norm(normal_vector1, dim= 2)
    magnitude2 = LA.norm(normal_vector2, dim= 2)
    
    # Calculate the cos value using the dot product and magnitudes
    cos_value = dot_product / (magnitude1 * magnitude2)
    # np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    
    return cos_value


def save_best_network(opt, model, epoch_label, running_loss_val, running_dist_val, val_loss_min, val_dist_min):
    '''

    :param opt: parameters of this projects
    :param model: model that need to be saved
    :param epoch_label: epoch of this model
    :param running_loss_val: validation loss of this epoch
    :param running_dist_val: validation distance of this epoch
    :param val_loss_min: min of previous validation loss
    :param val_dist_min: min of previous validation distance
    :return:
    '''

    if running_loss_val < val_loss_min:
        val_loss_min = running_loss_val
        file_name = os.path.join(opt.SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ best validation loss result - epoch %s: -------------\n' % (str(epoch_label)))
        if opt.multi_gpu:
            torch.save(model.module.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))

        else:
            torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))
        print('Best validation loss parameters saved.')
    else:
        val_loss_min = val_loss_min

    if running_dist_val < val_dist_min:
        val_dist_min = running_dist_val
        file_name = os.path.join(opt.SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ best validation dist result - epoch %s: -------------\n' % (str(epoch_label)))
        
        if opt.multi_gpu:
            torch.save(model.module.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))
        else:
            torch.save(model.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_dist_model'))
        print('Best validation dist parameters saved.')
    else:
        val_dist_min = val_dist_min

    return val_loss_min, val_dist_min


def save_best_network_reg(opt,VoxelMorph_net, epoch_label, running_loss_val, val_loss_min,count_non_improved_loss):
    '''

    :param opt: parameters of this projects
    :param model: model that need to be saved
    :param epoch_label: epoch of this model
    :param running_loss_val: validation loss of this epoch
    :param running_dist_val: validation distance of this epoch
    :param val_loss_min: min of previous validation loss
    :param val_dist_min: min of previous validation distance
    :return:
    '''

    if running_loss_val < val_loss_min:
        val_loss_min = running_loss_val
        file_name = os.path.join(opt.SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------reg - best validation loss result - epoch %s: -------------\n' % (str(epoch_label)))
        if opt.multi_gpu:
            torch.save(VoxelMorph_net.module.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model_reg' ))
        else:
            torch.save(VoxelMorph_net.state_dict(), os.path.join(opt.SAVE_PATH, 'saved_model', 'best_validation_loss_model_reg' ))

        print('Best reg validation loss parameters saved.')
        count_non_improved_loss = 0
    else:
        val_loss_min = val_loss_min
        count_non_improved_loss += 1

    

    return val_loss_min, count_non_improved_loss