

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch
import json
from torch.nn import MSELoss
import torch.nn as nn
from utils.loader import SSFrameDataset
from utils.network import build_model
from data.calib import read_calib_matrices
from utils.transform import LabelTransform, PredictionTransform
from utils.utils_ori import reference_image_points, compute_plane_normal,angle_between_planes
from options.train_options import TrainOptions
from utils.utils_ori import add_scalars_rec_volume,add_scalars_reg,save_best_network,save_best_network_reg,add_scalars_wrap_dist
from utils.utils_grid_data import *

# from monai.networks.nets.voxelmorph import VoxelMorphUNet, VoxelMorph
from utils.monai.networks.nets import VoxelMorph
from utils.monai.losses import BendingEnergyLoss
from utils.funcs import *

def compute_dimention(label_pred_type,num_points_each_frame=None,num_frames=None,type_option=None):
    if type_option == 'pred':
        num_frames = num_frames-1

    type_dim_dict = {
        "transform": 12*num_frames,
        "parameter": 6*num_frames,
        "point": 3*4*num_frames,  # predict four corner points, and then intepolete the other points in a frame
        "quaternion": 7*num_frames
    }
    return type_dim_dict[label_pred_type]   # num_points=self.image_points.shape[1]), num_pairs=self.pairs.shape[0]



def data_pairs_adjacent(num_frames):
# obtain the data_pairs to compute the tarnsfomration between adjacent frames

    # return torch.tensor([[n0,n0+1] for n0 in range(num_frames-1)])# [0,1],[1,2],...[n-1,n]

    return torch.tensor([[0,n0] for n0 in range(num_frames)])

def scatter_plot_3D(data,save_folder,save_name):
    # plot 3D scatter points

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:,0,:], data[:,1,:], data[:,2,:], marker='o')
    plt.show()
    # plt.savefig(save_folder+'/'+save_name)



def union_volome(gt_volume_position,pred_volume,pred_volume_position):
    # crop the volume2 based on volume 1
    # get the boundary of ground truth volume
    #  not completed
    gt_X = gt_volume_position[0]
    gt_Y = gt_volume_position[1]
    gt_Z = gt_volume_position[2]
    min_x = torch.min(gt_X)
    max_x = torch.max(gt_X)
    min_y = torch.min(gt_Y)
    max_y = torch.max(gt_Y)
    min_z = torch.min(gt_Z)
    max_z = torch.max(gt_Z)

    #  the length of each dimention is larger than the volume, because of the torch.ceil operation, we need to
    #  add one additional length for each dimention to allow the torch.ceil opration
    pred_X = torch.zeros((pred_volume.shape[0],pred_volume.shape[1],pred_volume.shape[2])) 

    pred_X = pred_volume_position[0] + pred_X[:-1,:-1,:-1]
    pred_Y = pred_volume_position[1]
    pred_Z = pred_volume_position[2]


    inside_min_x = torch.where(pred_X > min_x, 1.0, 0.0)
    inside_max_x = torch.where(pred_X < max_x, 1.0, 0.0)
    inside_min_y = torch.where(pred_Y > min_y, 1.0, 0.0)
    inside_max_y = torch.where(pred_Y < max_y, 1.0, 0.0)
    inside_min_z = torch.where(pred_Z > min_z, 1.0, 0.0)
    inside_max_z = torch.where(pred_Z < max_z, 1.0, 0.0)

    return pred_volume*inside_min_x*inside_max_x*inside_min_y*inside_max_y*inside_min_z*inside_max_z

def calculateConvPose_batched(pts_batched,option,device):
    for i_batch in range(pts_batched.shape[0]):
        
        ConvR = calculateConvPose(pts_batched[i_batch,...],option,device)
        # ConvR = ConvR.repeat(pts_batched[i_batch,...].shape[0], 1,1)[None,...]
        ConvR = ConvR[None,...]
        if i_batch == 0:
            ConvR_batched = ConvR
        else:
            ConvR_batched = torch.cat((ConvR_batched,ConvR),0)
    return ConvR_batched
            


def calculateConvPose(pts,option,device):
    """Calculate roto-translation matrix from global reference frame to *convenient* reference frame.
    Voxel-array dimensions are calculated in this new refence frame. This rotation is important whenever the US scans sihouette is remarkably
    oblique to some axis of the global reference frame. In this case, the voxel-array dimensions (calculated by the smallest parallelepipedon 
    wrapping all the realigned scans), calculated in the global refrence frame, would not be optimal, i.e. larger than necessary.
    
    .. image:: diag_scan_direction.png
        :scale: 30 %          
        
    Parameters
    ----------
    convR : mixed
        Roto-translation matrix.
        If str, it specifies the method for automatically calculate the matrix.
        If 'auto_PCA', PCA is performed on all US image corners. The x, y and z of the new convenient reference frame are represented by the eigenvectors out of the PCA.
        If 'first_last_frames_centroid', the convenent reference frame is expressed as:
        
        - x from first image centroid to last image centroid
        - z orthogonal to x and the axis and the vector joining the top-left corner to the top-right corner of the first image
        - y orthogonal to z and x
        
        If np.ndarray, it must be manually specified as a 4 x 4 affine matrix.
        
    """
    # pts = torch.reshape(pts,(pts.shape[0],-1,3))
    # pts = torch.permute(pts, (2, 0, 1))
    
    # Calculating best pose automatically, if necessary
    # ivx = np.array(self.voxFrames)
    if option == 'auto_PCA':
        # Perform PCA on image corners
        # print ('Performing PCA on images corners...')
        with torch.no_grad():
            pts1 = pts.permute(0,2,1).reshape([-1,3])#.cpu().numpy()
            U, s = pca(torch.transpose(pts1, 0, 1)) 
            # Build convenience affine matrix
            convR = torch.vstack((torch.hstack((U,torch.zeros((3,1)).to(device))),torch.tensor([0,0,0,1]).to(device)))#.T
            # convR = torch.from_numpy(convR).to(torch.float32).to(device)
        # print ('PCA perfomed')
    elif option == 'first_last_frames_centroid':
        # Search connection from first image centroid to last image centroid (X)
        # print ('Performing convenient reference frame calculation based on first and last image centroids...')
        C0 = torch.mean(pts[0,:,:], 1)  # 3
        C1 = torch.mean(pts[-1,:,:], 1)  # 3
        X = C1 - C0
        # Define Y and Z axis
        Ytemp = pts[0,:,0] - pts[0,:,1]   # from top-left corner to top-right corner
        
        Z = torch.cross(X, Ytemp)
        Y = torch.cross(Z, X)
        # Normalize axis length
        X = X / torch.linalg.norm(X)
        Y = Y / torch.linalg.norm(Y)
        Z = Z / torch.linalg.norm(Z)
        # Create rotation matrix
        # M = np.array([X, Y, Z]).T
        M = torch.transpose(torch.stack((X,Y,Z),0),0,1)
        # Build convenience affine matrix
        # convR = np.vstack((np.hstack((M,np.zeros((3,1)))),[0,0,0,1])).T
        convR = torch.transpose(torch.vstack((torch.hstack((M,torch.zeros((3,1)).to(device))),torch.tensor([0,0,0,1]).to(device))),0,1)
        # print ('Convenient reference frame calculated')

    return convR

def ConvPose(labels,ori_pts,pre, option_method,device):
    convR_batched = calculateConvPose_batched(labels,option = option_method,device=device)    
    
    for i_batch in range(convR_batched.shape[0]):
        
        labels_i = torch.matmul(ori_pts[i_batch,...],convR_batched[i_batch,...])[None,...]
        minx = torch.min(labels_i[...,0])
        miny = torch.min(labels_i[...,1])
        minz = torch.min(labels_i[...,2])
        labels_i[...,0]-=minx
        labels_i[...,1]-=miny
        labels_i[...,2]-=minz
        pred_pts_i = torch.matmul(pre[i_batch,...],convR_batched[i_batch,...])[None,...]

        # return for future use
        minxyz=torch.from_numpy(np.array([minx.item(),miny.item(),minz.item()]))
        
        pred_pts_i[...,0]-=minx
        pred_pts_i[...,1]-=miny
        pred_pts_i[...,2]-=minz

        if i_batch == 0:
            labels_opt = labels_i
            pred_pts_opt = pred_pts_i
            minxyz_all = minxyz
        else:
            labels_opt = torch.cat((labels_opt,labels_i),0)
            pred_pts_opt = torch.cat((pred_pts_opt,pred_pts_i),0)
            minxyz_all = torch.cat((minxyz_all,minxyz),0)

    labels_opt = labels_opt[:,:,:,0:3].permute(0,1,3,2)
    pred_pts_opt = pred_pts_opt[:,:,:,0:3].permute(0,1,3,2)
    return labels_opt,pred_pts_opt, convR_batched,minxyz_all

def pca(D):
    """Run Principal Component Analysis on data matrix. It performs SVD
    decomposition on data covariance matrix.
    
    Parameters
    ----------
    D : np.ndarray
        Nv x No matrix, where Nv is the number of variables 
        and No the number of observations.
    
    Returns
    -------
    list
        U, s as out of SVD (``see np.linalg.svd``)

    """
    cov = torch.cov(D)
    U, s, V = torch.linalg.svd(cov)
    return U, s


class Train_Rec_Reg_Model():

    def __init__(
        self, 
        opt,
        non_improve_maxmum, 
        reg_loss_weight,
        val_loss_min,
        val_dist_min,
        val_loss_min_reg,
        dset_train,
        dset_val,
        dset_train_reg,
        dset_val_reg,
        device,
        writer,
        option
        
        ):

        self.non_improve_maxmum = non_improve_maxmum
        self.val_loss_min = val_loss_min
        self.val_dist_min = val_dist_min
        self.val_loss_min_reg = val_loss_min_reg

        
        self.val_dist_min_T = val_loss_min
        self.val_dist_min_R = val_loss_min

        self.device = device
        self.writer = writer
        self.option = option
        self.opt = opt
        self.dset_train = dset_train
        self.dset_val = dset_val
       
        self.data_pairs = data_pairs_adjacent(opt.NUM_SAMPLES)

        
        



        self.train_loader_rec = torch.utils.data.DataLoader(
            self.dset_train,
            batch_size=self.opt.MINIBATCH_SIZE_rec,
            shuffle=True,
            num_workers=0
            )
        
            

        self.val_loader_rec = torch.utils.data.DataLoader(
            self.dset_val,
            batch_size=1, 
            shuffle=True,
            num_workers=0
            )
        



        ## loss
        self.tform_calib_scale, self.tform_calib_R_T,  self.tform_calib = read_calib_matrices(filename_calib=self.opt.FILENAME_CALIB, resample_factor=self.opt.RESAMPLE_FACTOR, device=self.device)


        self.image_points = reference_image_points((self.dset_train[0][0].shape[1],self.dset_train[0][0].shape[2]),(self.dset_train[0][0].shape[1],self.dset_train[0][0].shape[2])).to(self.device)
        self.pred_dim = compute_dimention(self.opt.PRED_TYPE, self.image_points.shape[1],self.opt.NUM_SAMPLES,'pred')
        self.label_dim = compute_dimention(self.opt.LABEL_TYPE, self.image_points.shape[1],self.opt.NUM_SAMPLES,'label')



        self.transform_label = LabelTransform(
            self.opt.LABEL_TYPE,
            pairs= self.data_pairs,  #
            image_points= self.image_points,
            in_image_coords=True,
            tform_image_to_tool= self.tform_calib,
            tform_image_mm_to_tool= self.tform_calib_R_T
            )
       
        self.transform_prediction = PredictionTransform(
             self.opt.PRED_TYPE,
            "transform",
            num_pairs= self.data_pairs.shape[0]-1,
            image_points= self.image_points,
            in_image_coords=True,
            tform_image_to_tool= self.tform_calib,
            tform_image_mm_to_tool= self.tform_calib_R_T
            )

        # loss
        self.criterion = torch.nn.MSELoss()
        self.img_loss = MSELoss()
        self.regularization = BendingEnergyLoss()


        ## network
        self.model = build_model(
            self.opt,
            in_frames =  self.opt.NUM_SAMPLES,
            pred_dim =  self.pred_dim,
            label_dim =  self.label_dim,
            image_points =  self.image_points,
            tform_calib =  self.tform_calib,
            tform_calib_R_T =  self.tform_calib_R_T
            ).to( self.device)


        self.VoxelMorph_net = VoxelMorph(in_channels = self.opt.in_ch_reg,ddf_dirc=self.opt.ddf_dirc).to(self.device)

        
        self.current_epoch = 0
        self.reg_loss_weight = reg_loss_weight

        # set optimiser
        all_paras = list(self.model.parameters())+list(self.VoxelMorph_net.parameters())
        self.optimiser_rec_reg = torch.optim.Adam(all_paras, lr=self.opt.LEARNING_RATE_rec)



    def train_rec_model(self):
        # train reconstruction network

        self.model.train(True)
        self.VoxelMorph_net.train(True)
        self.switch_off_batch_norm()
        
        

        for epoch in range(int(self.opt.retain_epoch), int(self.opt.retain_epoch)+self.opt.NUM_EPOCHS):

            train_epoch_loss = 0
            train_epoch_dist,train_epoch_wrap_dist = 0,0
            train_epoch_loss_reg = 0
            train_epoch_loss_rec = 0
            for step, (frames, tforms, tforms_inv) in enumerate(self.train_loader_rec):
                frames, tforms, tforms_inv = frames.to(self.device), tforms.to(self.device), tforms_inv.to(self.device)

                # cannot use the ground truth coordinates based on the camera coordinates system, 
                # which will depend on the posotion of camera
                # the transformation between each frame and frame 0
                tforms_each_frame2frame0 = self.transform_label(tforms, tforms_inv)
                # obtain the coordinates of each frame, set frame 0 as the reference frame

                frames = frames/255 # normalise image into range (0,1)
                self.optimiser_rec_reg.zero_grad()
                outputs = self.model(frames)
                # 6 parameter to 4*4 transformation
                pred_transfs = self.transform_prediction(outputs)
                # make the predicted transformations are based on frame 0
                # predict only opt.NUM_FRAES-1 transformatons,and let the first frame equals to identify matrix
                predframe0 = torch.eye(4,4)[None,...].repeat(pred_transfs.shape[0],1, 1,1).to(self.device)
                pred_transfs = torch.cat((predframe0,pred_transfs),1)

                # transformtion to points
                if self.opt.img_pro_coord == 'img_coord':
                    labels = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(tforms_each_frame2frame0,torch.matmul(self.tform_calib,self.image_points)))[:,:,0:3,...]
                    pred_pts = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.image_points)))[:,:,0:3,...]
                else:
                    labels = torch.matmul(tforms_each_frame2frame0,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]
                    pred_pts = torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]
                
                loss1 = self.criterion(pred_pts, labels)
                dist = ((pred_pts-labels)**2).sum(dim=2).sqrt().mean()

                if self.opt.Conv_Coords == 'optimised_coord' and self.opt.img_pro_coord == 'pro_coord':
                    # change the corrdinates system of the origial data (label), such that the occupied volume is smallest
                    # note: only calculate the nre coordinates syatem of the groundtruth, and predicted points will use the same transformation as groundtruth 
                    ori_pts = torch.matmul(tforms_each_frame2frame0,torch.matmul(self.tform_calib,self.image_points)).permute(0,1,3,2)
                    pre = torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.image_points)).permute(0,1,3,2)

                    labels, pred_pts, convR_batched,minxyz_all = ConvPose(labels, ori_pts, pre, 'auto_PCA',self.device)
                    
                elif self.opt.Conv_Coords == 'optimised_coord' and self.opt.img_pro_coord != 'pro_coord':
                    raise('optimised_coord must be used when pro_coord')


                
                if self.opt.Loss_type == "MSE_points":
                    loss = loss1
                    # gt_volume, pred_volume = self.scatter_pts_intepolation(labels,pred_pts,frames,step)
                    loss2=loss#self.criterion(pred_volume, gt_volume)
                elif self.opt.Loss_type == "Plane_norm":
                    
                    normal_gt = compute_plane_normal(labels)
                    normal_np = compute_plane_normal(pred_pts)
                    cos_value = angle_between_planes(normal_gt,normal_np)
                    loss = loss1-sum(sum(cos_value))
                elif self.opt.Loss_type == "reg" or self.opt.Loss_type == "rec_reg" or self.opt.Loss_type == "wraped":
                    # scatter points to grid points
                    gt_volume,pred_volume, warped, ddf = self.scatter_pts_registration(labels,pred_pts,frames,step)
                    # obtain metric
                    if self.opt.ddf_dirc == 'Move' and self.opt.Conv_Coords == 'optimised_coord':
                        # the function below is suit for points at optimised coodinates system
                        # wrap_dist is the MSE between wrapped prediction and ground truth
                        wrap_mseloss, wrap_dist,pred_pts_warped = wrapped_pred_dist(ddf,pred_pts,labels,self.option,frames.shape[2],frames.shape[3],convR_batched,minxyz_all,self.device)

                    
                    if self.opt.ddf_dirc == 'Fix':
                        loss2 = self.img_loss(torch.squeeze(warped,1),gt_volume) + self.regularization(ddf)
                    elif self.opt.ddf_dirc == 'Move':
                        loss2 = self.img_loss(torch.squeeze(warped,1),pred_volume) + self.regularization(ddf)
                    if self.opt.Loss_type == "reg":
                        # test if only use registartion can backward
                        loss = loss2
                    elif self.opt.Loss_type == "rec_reg":
                        loss = loss1+self.reg_loss_weight*loss2
                    elif self.opt.Loss_type == "wraped" and self.opt.ddf_dirc == 'Move':
                        loss = wrap_mseloss + self.regularization(ddf)

                elif self.opt.Loss_type == "rec_volume":
                    gt_volume, pred_volume = self.scatter_pts_intepolation(labels,pred_pts,frames,step)
                    loss2 = self.criterion(pred_volume, gt_volume)
                    loss = loss1 + loss2
                elif self.opt.Loss_type == "rec_volume10000":
                    gt_volume, pred_volume = self.scatter_pts_intepolation(labels,pred_pts,frames,step)
                    loss2 = self.criterion(pred_volume, gt_volume)
                    loss = loss1 + self.reg_loss_weight*loss2
                elif self.opt.Loss_type == "volume_only":
                    gt_volume, pred_volume = self.scatter_pts_intepolation(labels,pred_pts,frames,step)
                    loss = self.criterion(pred_volume, gt_volume)
                    # for plotting use
                    
                    loss2=loss

                train_epoch_loss += loss.item()
                train_epoch_dist += dist.item()
                train_epoch_wrap_dist += wrap_dist.item()
                train_epoch_loss_rec = train_epoch_loss_rec + loss1.item()
                train_epoch_loss_reg += loss2.item()

                if epoch !=0:
                    loss.backward()
                    self.optimiser_rec_reg.step()


            train_epoch_loss /= (step + 1)
            train_epoch_dist /= (step + 1)
            train_epoch_wrap_dist /= (step + 1)
            train_epoch_loss_reg /= (step + 1)
            train_epoch_loss_rec /= (step + 1)

            if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.FREQ_INFO):
                print('[Rec - Epoch %d] train-loss-rec=%.3f, train-dist=%.3f' % (epoch, train_epoch_loss_rec, train_epoch_dist))

            
            # validation    
            if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.val_fre):
                self.model.train(False)
                self.VoxelMorph_net.train(False)
                self.switch_off_batch_norm()

                epoch_loss_val = 0
                epoch_dist_val = 0
                epoch_loss_val_reg = 0
                epoch_loss_val_rec = 0
                epoch_wrap_dist_val = 0
                for step, (fr_val, tf_val, tf_val_inv) in enumerate(self.val_loader_rec):

                    fr_val, tf_val, tf_val_inv = fr_val.to(self.device), tf_val.to(self.device), tf_val_inv.to(self.device)
                    tforms_each_frame2frame0_val = self.transform_label(tf_val, tf_val_inv)  
                    fr_val = fr_val/255
                    out_val = self.model(fr_val)

                    pr_transfs_val = self.transform_prediction(out_val)
                    predframe0_val = torch.eye(4,4)[None,...].repeat(pr_transfs_val.shape[0],1, 1,1).to(self.device)
                    pr_transfs_val = torch.cat((predframe0_val,pr_transfs_val),1)

                    if self.opt.img_pro_coord == 'img_coord':
                        labels_val = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(tforms_each_frame2frame0_val,torch.matmul(self.tform_calib,self.image_points)))[:,:,0:3,...]
                        pred_pts_val = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(pr_transfs_val,torch.matmul(self.tform_calib,self.image_points)))[:,:,0:3,...]
                    else:
                        pred_pts_val = torch.matmul(pr_transfs_val,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]
                        labels_val = torch.matmul(tforms_each_frame2frame0_val,torch.matmul(self.tform_calib,self.image_points))[:,:,0:3,...]

                    loss1_val = self.criterion(pred_pts_val, labels_val)
                    dist_val = ((pred_pts_val-labels_val)**2).sum(dim=2).sqrt().mean()

                    if self.opt.Conv_Coords == 'optimised_coord' and self.opt.img_pro_coord == 'pro_coord':
                        # change labels to a convenient coordinates system
        
                        ori_pts_val = torch.matmul(tforms_each_frame2frame0_val,torch.matmul(self.tform_calib,self.image_points)).permute(0,1,3,2)
                        pre_val = torch.matmul(pr_transfs_val,torch.matmul(self.tform_calib,self.image_points)).permute(0,1,3,2)

                        labels_val, pred_pts_val, convR_batched_val,minxyz_all_val = ConvPose(labels_val, ori_pts_val, pre_val, 'auto_PCA',self.device)
                    
                    elif self.opt.Conv_Coords == 'optimised_coord' and self.opt.img_pro_coord != 'pro_coord':
                        raise('optimised_coord must be used when pro_coord')

                    if self.opt.Loss_type == "MSE_points":
                        
                        # gt_volume_val, pred_volume_val = self.scatter_pts_intepolation(labels_val,pred_pts_val,fr_val,step)
                        loss_val = loss1_val
                        loss2_val = loss_val#self.criterion(gt_volume_val, pred_volume_val)
                    elif self.opt.Loss_type == "Plane_norm":
                        
                        normal_gt_val = compute_plane_normal(labels_val)
                        normal_np_val = compute_plane_normal(pred_pts_val)
                        cos_value_val = angle_between_planes(normal_gt_val,normal_np_val)
                        loss_val = loss1_val-sum(sum(cos_value_val))
                    elif self.opt.Loss_type == "reg" or self.opt.Loss_type == "rec_reg" or self.opt.Loss_type == "wraped":
                        # scatter points to grid points
                        
                        gt_volume_val,pred_volume_val, warped_val, ddf_val = self.scatter_pts_registration(labels_val,pred_pts_val,fr_val,step)
                        # obtain metric
                        if self.opt.ddf_dirc == 'Move' and self.opt.Conv_Coords == 'optimised_coord':
                            # the function below is suit for points at optimised coodinates system
                            # wrap_dist is the MSE between wrapped prediction and ground truth
                            # with torch.no_grad():
                            wrap_mseloss_val, wrap_dist_val,pred_pts_warped_val = wrapped_pred_dist(ddf_val,pred_pts_val,labels_val,self.option,fr_val.shape[2],fr_val.shape[3],convR_batched_val,minxyz_all_val,self.device)

                        if self.opt.ddf_dirc == 'Fix':
                            loss2_val = self.img_loss(torch.squeeze(warped_val,1),gt_volume_val) + self.regularization(ddf_val)
                        elif self.opt.ddf_dirc == 'Move':
                            loss2_val = self.img_loss(torch.squeeze(warped_val,1),pred_volume_val) + self.regularization(ddf_val)
                        

                        if self.opt.Loss_type == "reg":
                            # test if only use registartion can backward      
                            loss_val = loss2_val
                        elif self.opt.Loss_type == "rec_reg":     
                            loss_val = loss1_val+self.reg_loss_weight*loss2_val

                        elif self.opt.Loss_type == "wraped" and self.opt.ddf_dirc == 'Move':
                            loss_val = wrap_mseloss_val + self.regularization(ddf_val)

                    elif self.opt.Loss_type == "rec_volume":
                        gt_volume_val, pred_volume_val = self.scatter_pts_intepolation(labels_val,pred_pts_val,fr_val,step)
                        
                        loss2_val = self.criterion(gt_volume_val, pred_volume_val)
                        loss_val = loss1_val + loss2_val
                    elif self.opt.Loss_type == "rec_volume10000":
                        gt_volume_val, pred_volume_val = self.scatter_pts_intepolation(labels_val,pred_pts_val,fr_val,step)
                        
                        loss2_val = self.criterion(gt_volume_val, pred_volume_val)
                        loss_val = loss1_val + self.reg_loss_weight*loss2_val
                    elif self.opt.Loss_type == "volume_only":
                        gt_volume_val, pred_volume_val = self.scatter_pts_intepolation(labels_val,pred_pts_val,fr_val,step)
                        loss_val = self.criterion(gt_volume_val, pred_volume_val)
                        
                        loss2_val = loss_val                
                    
                    epoch_loss_val += loss_val.item()
                    epoch_dist_val += dist_val.item()
                    epoch_wrap_dist_val += wrap_dist_val.item()
                    epoch_loss_val_reg += loss2_val.item()
                    epoch_loss_val_rec = epoch_loss_val_rec + loss1_val.item()

                epoch_loss_val /= (step+1)
                epoch_dist_val /= (step+1)
                epoch_wrap_dist_val /= (step+1)
                epoch_loss_val_reg /= (step+1)
                epoch_loss_val_rec /= (step+1)

                # save model
                self.save_rec_model(epoch)
                self.save_reg_model(epoch)

                self.save_best_models_val_T(epoch,epoch_dist_val)
                self.save_best_models_val_R(epoch,epoch_wrap_dist_val)

                
                if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.FREQ_INFO):
                    print('[Rec - Epoch %d] val-loss-rec=%.3f, val-dist=%.3f' % (epoch, epoch_loss_val_rec, epoch_dist_val))
                    # print('[Rec - Epoch %d] count_non_improved_loss=%d' % (epoch,count_non_improved_loss))
                # add to tensorboard
                loss_dists = {'train_epoch_loss_all': train_epoch_loss, 
                            'train_epoch_dist': train_epoch_dist,
                            'train_epoch_loss_reg':train_epoch_loss_reg,
                            'train_epoch_loss_rec':train_epoch_loss_rec,

                            'epoch_loss_val_all':epoch_loss_val,
                            'epoch_dist_val':epoch_dist_val,
                            'epoch_loss_val_reg':epoch_loss_val_reg,
                            'epoch_loss_val_rec':epoch_loss_val_rec}
                add_scalars_rec_volume(self.writer, epoch, loss_dists)

                dist_wrap = {'train_wrap_dist': train_epoch_wrap_dist,
                            'val_wrap_dist':epoch_wrap_dist_val,
                            }
                add_scalars_wrap_dist(self.writer, epoch, dist_wrap,'rec_reg')
                

                self.model.train(True)
                
                self.VoxelMorph_net.train(True)
                self.switch_off_batch_norm()
        
        


    def scatter_pts_registration(self,labels,pred_pts,frames,step):
        # intepelote scatter points and thenregistartion

        gt_volume, pred_volume = self.scatter_pts_intepolation(labels,pred_pts,frames,step)

        warped, ddf = self.VoxelMorph_net(moving = torch.unsqueeze(pred_volume, 1), 
                    fixed = torch.unsqueeze(gt_volume, 1))
        
        return gt_volume,pred_volume, warped, ddf
    
    def scatter_pts_intepolation(self,labels,pred_pts,frames,step):
        # intepelote scatter points
        if self.option == 'common_volume':
            common_volume = compute_common_volume(labels,pred_pts,self.device)

        gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(scatter_pts = labels,
                                                            frames = frames,
                                                            time_log=None,
                                                            saved_folder_test = None,
                                                            scan_name='gt_step'+str(step),
                                                            device = self.device,
                                                            option = self.opt.intepoletion_method,
                                                            volume_size = self.opt.intepoletion_volume,
                                                            volume_position = common_volume
                                                            )
                    
        
        pred_volume,pred_volume_position = interpolation_3D_pytorch_batched(scatter_pts = pred_pts,
                                                frames = frames,
                                                time_log=None,
                                                saved_folder_test = None,
                                                scan_name='pred_step'+str(step),
                                                device = self.device,
                                                option = self.opt.intepoletion_method,
                                                volume_size = self.opt.intepoletion_volume,
                                                volume_position = common_volume
                                                )
        
        return gt_volume, pred_volume



    def save_rec_model(self,epoch):
        if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.FREQ_SAVE):
            if self.opt.multi_gpu:
                torch.save(self.model.module.state_dict(), os.path.join(self.opt.SAVE_PATH,'saved_model','model_epoch%08d' % epoch))

            else:
                torch.save(self.model.state_dict(), os.path.join(self.opt.SAVE_PATH,'saved_model','model_epoch%08d' % epoch))

            print('Model parameters saved.')
            # list_dir = os.listdir(os.path.join(self.opt.SAVE_PATH, 'saved_model'))
            # saved_models = [i for i in list_dir if i.startswith('model_epoch')]
            # if len(saved_models)>4:
            #     print(saved_models)
            #     os.remove(os.path.join(self.opt.SAVE_PATH,'saved_model',sorted(saved_models)[0]))

            
    def save_reg_model(self,epoch):
        if epoch in range(0, self.opt.NUM_EPOCHS, self.opt.FREQ_SAVE):
            if self.opt.multi_gpu:
                torch.save(self.VoxelMorph_net.module.state_dict(), os.path.join(self.opt.SAVE_PATH,'saved_model','model_reg_epoch%08d' % epoch))

            else:
                torch.save(self.VoxelMorph_net.state_dict(), os.path.join(self.opt.SAVE_PATH,'saved_model','model_reg_epoch%08d' % epoch))

            print('Model parameters saved.')
            # list_dir = os.listdir(os.path.join(self.opt.SAVE_PATH, 'saved_model'))
            
            # saved_models_reg = [i for i in list_dir if i.startswith('model_reg_epoch')]
            # if len(saved_models_reg)>4:
            #     print(saved_models_reg)
            #     os.remove(os.path.join(self.opt.SAVE_PATH,'saved_model',sorted(saved_models_reg)[0]))


    def load_best_rec_model(self):
        # load the best transformation model for training again or for generating volume data for registartion model use
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH,'saved_model', 'best_validation_loss_model'),map_location=torch.device(self.device)))
        except:
            print('No best rec model saved at the moment...')

    def load_best_reg_model(self):
        # load the best registation model registartion
        try:
            self.VoxelMorph_net.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH,'saved_model', 'best_validation_loss_model_reg'),map_location=torch.device(self.device)))
        except:
            print('No best reg model saved at the moment...')

    def load_recon_model_initial(self):
        # load the best registation model registartion
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH,'saved_model', 'ete_initial_recon'),map_location=torch.device(self.device)))
        except:
            raise('No best model saved at the moment...')
        
        
    def load_def_model_initial(self):
        # load the best registation model registartion
        try:
            self.VoxelMorph_net.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH,'saved_model', 'ete_initial_def'),map_location=torch.device(self.device)))
        except:
            raise('No best model saved at the moment...')

        
    def switch_off_batch_norm(self):
        # # turn off batchnorm
        if self.opt.BatchNorm=='BNoff':
            # self.features = torch.nn.Sequential(*list(self.model.children()))
            # l3=[i for i in self.model.modules() if list(i.children())==[]] list all the single layers
            # l1 = self.model.modules() 
            for m in self.model.modules() :
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_parameters()
                    m.eval()
                    with torch.no_grad():
                        m.weight.fill_(1.0)
                        m.bias.zero_()
                        m.momentum = 1

            for m in self.VoxelMorph_net.modules() :
                if isinstance(m, nn.BatchNorm2d):
                    m.reset_parameters()
                    m.eval()
                    with torch.no_grad():
                        m.weight.fill_(1.0)
                        m.bias.zero_()
                        m.momentum = 1
                # if isinstance(m, nn.BatchNorm2d):
                #     m.eval()
                #     m.weight.requires_grad = False
                #     m.bias.requires_grad = False
            # self.model = self.features
                        
    def multi_model(self):
        if self.opt.multi_gpu:
            self.model= nn.DataParallel(self.model)
            self.VoxelMorph_net = nn.DataParallel(self.VoxelMorph_net)
            print('multi-gpu')
            print(os.environ["CUDA_VISIBLE_DEVICES"])

    def save_best_models_train_T(self, epoch,running_dist):
        # save best T model with minimum train dist
        
        if running_dist < self.train_dist_min_T:
            self.train_dist_min_T = running_dist
            file_name = os.path.join(self.opt.SAVE_PATH, 'config.txt')
            with open(file_name, 'a') as opt_file:
                opt_file.write('------------ best train dist T - epoch %s: dist = %f -------------\n' % (str(epoch),running_dist))
            if self.opt.multi_gpu:
                torch.save(self.model.module.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_train_dist_T_T'))
                torch.save(self.VoxelMorph_net.module.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_train_dist_T_R'))

            else:
                torch.save(self.model.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_train_dist_T_T' ))
                torch.save(self.VoxelMorph_net.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_train_dist_T_R' ))

            print('Best train dist T parameters saved.')

    def save_best_models_train_R(self, epoch,running_dist):
    # save best T model with minimum train dist
        
        if running_dist < self.train_dist_min_R:
            self.train_dist_min_R = running_dist
            file_name = os.path.join(self.opt.SAVE_PATH, 'config.txt')
            with open(file_name, 'a') as opt_file:
                opt_file.write('------------ best train dist R - epoch %s: dist = %f -------------\n' % (str(epoch),running_dist))
            if self.opt.multi_gpu:
                torch.save(self.model.module.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_train_dist_R_T'))
                torch.save(self.VoxelMorph_net.module.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_train_dist_R_R'))

            else:
                torch.save(self.model.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_train_dist_R_T' ))
                torch.save(self.VoxelMorph_net.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_train_dist_R_R' ))

            print('Best train dist R parameters saved.')


    def save_best_models_val_T(self, epoch,running_dist):
        # save best T model with minimum train dist
        
        if running_dist < self.val_dist_min_T:
            self.val_dist_min_T = running_dist
            file_name = os.path.join(self.opt.SAVE_PATH, 'config.txt')
            with open(file_name, 'a') as opt_file:
                opt_file.write('------------ best val dist T - epoch %s: dist = %f -------------\n' % (str(epoch),running_dist))
            if self.opt.multi_gpu:
                torch.save(self.model.module.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_val_dist_T_T'))
                torch.save(self.VoxelMorph_net.module.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_val_dist_T_R'))

            else:
                torch.save(self.model.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_val_dist_T_T' ))
                torch.save(self.VoxelMorph_net.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_val_dist_T_R' ))

            print('Best val dist T parameters saved.')

    def save_best_models_val_R(self, epoch,running_dist):
    # save best T model with minimum train dist
        
        if running_dist < self.val_dist_min_R:
            self.val_dist_min_R = running_dist
            file_name = os.path.join(self.opt.SAVE_PATH, 'config.txt')
            with open(file_name, 'a') as opt_file:
                opt_file.write('------------ best val dist R - epoch %s: dist = %f -------------\n' % (str(epoch),running_dist))
            if self.opt.multi_gpu:
                torch.save(self.model.module.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_val_dist_R_T'))
                torch.save(self.VoxelMorph_net.module.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_val_dist_R_R'))

            else:
                torch.save(self.model.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_val_dist_R_T' ))
                torch.save(self.VoxelMorph_net.state_dict(), os.path.join(self.opt.SAVE_PATH, 'saved_model', 'best_val_dist_R_R' ))

            print('Best val dist R parameters saved.')
        

