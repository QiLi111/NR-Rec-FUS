import os
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from utils.network import build_model
from data.calib import read_calib_matrices
from utils.transform import LabelTransform, PredictionTransform
from utils.utils import reference_image_points

from utils.utils_grid_data import *
from utils.utils_meta import *
import sys
sys.path.append(os.getcwd()+'/utils')
# from monai.networks.nets.voxelmorph import VoxelMorphUNet, VoxelMorph
from monai.networks.nets import VoxelMorph


class Visualizer_plot_volume():  # plot scan

    def __init__(self, opt,device, dset,model_name,data_pairs,option,batchsize = 1):
        self.opt = opt
        self.opt_test = self.opt
        self.device = device
        self.dset = dset
        self.opt.MINIBATCH_SIZE = batchsize

        self.FILENAME_WEIGHTS = model_name
        self.model_name = model_name
        self.option = option

        self.data_pairs = data_pairs
        self.tform_calib_scale, self.tform_calib_R_T, self.tform_calib = read_calib_matrices(
            filename_calib=self.opt.FILENAME_CALIB,
            resample_factor=self.opt.RESAMPLE_FACTOR,
            device=self.device#'cpu'#self.device''
        )
        # using prediction: transform frame_points from current image to starting (reference) image 0
        # four corner points
        self.four_points = reference_image_points((self.dset[0][0].shape[1],self.dset[0][0].shape[2]), 2)#.to(self.device)
        

        # using GT: transform pixel_points from current image to starting (reference) image 0
        # all points in a frame
        self.all_points = reference_image_points((self.dset[0][0].shape[1],self.dset[0][0].shape[2]),(self.dset[0][0].shape[1],self.dset[0][0].shape[2])).to(self.device)
        # four corner points in a frame
        self.four_pts = reference_image_points((self.dset[0][0].shape[1],self.dset[0][0].shape[2]),2).to(self.device)

       
        self.transform_label = LabelTransform(
            label_type=self.opt.LABEL_TYPE,
            pairs=self.data_pairs,  #
            image_points=self.all_points ,
            in_image_coords=True,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T
            )
        

        self.transform_prediction = PredictionTransform(
            self.opt.PRED_TYPE,
            "transform",
            num_pairs=self.data_pairs.shape[0]-1,
            image_points=self.all_points,
            in_image_coords=True,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T
            )
        
        self.pred_dim = compute_dimention(self.opt.PRED_TYPE, self.all_points.shape[1],self.opt.NUM_SAMPLES,type_option='pred')
        self.label_dim = compute_dimention(self.opt.LABEL_TYPE, self.all_points.shape[1],self.opt.NUM_SAMPLES,'label')

        ## load the model
        self.model = build_model(
            self.opt,
            in_frames =  self.opt.NUM_SAMPLES,
            pred_dim =  self.pred_dim,
            label_dim =  self.label_dim,
            image_points =  self.all_points,
            tform_calib =  self.tform_calib,
            tform_calib_R_T =  self.tform_calib_R_T
            ).to( self.device)
        
        self.VoxelMorph_net = VoxelMorph(in_channels = self.opt.in_ch_reg,ddf_dirc=self.opt.ddf_dirc).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH, self.opt_test.MODEL_FN,self.model_name[0]), map_location=torch.device(self.device)))
        
        try:
            # if self.opt.in_ch_reg == 1 and opt.ddf_dirc == 'Move':
            # if opt.in_ch_reg=2, the wrapped prediction cannot be used as the final output as it contains ground truth
            self.VoxelMorph_net.load_state_dict(torch.load(os.path.join(self.opt.SAVE_PATH,self.opt_test.MODEL_FN, self.model_name[1]),map_location=torch.device(self.device)))
        except:
            print('No R model loaded')

        
        self.model.train(False)
        self.VoxelMorph_net.train(False)

        # evaluation metrics initialistion
        self.T_Global_AllPts_Dist = []
        self.T_R_Warp_Global_AllPts_Dist = []
        
        self.T_Global_FourPts_Dist = []
        self.T_R_Warp_Global_FourPts_Dist = []
        
        self.T_Local_FourPts_Dist = []
        self.T_Local_AllPts_Dist = []
       

    def generate_volume_data(self, scan_index,saved_folder,based_volume = 'common_volume'):
        
        # four evaluation metric
        # all pts using global transfromations
        # all pts using local transfromations
        # 4 corner pts using global transfromations
        # 4 corner pts using global transfromations


        frames, tforms, tforms_inv = self.dset[scan_index]
        #  the first dimention is batchsize
        frames, tforms, tforms_inv = (torch.tensor(t)[None,...].to(self.device) for t in [frames, tforms, tforms_inv])
        frames = frames/255
        saved_folder = saved_folder
        saved_name = str(list(self.dset.indices_in_use[scan_index]))+'__'+self.dset.name_scan[self.dset.indices_in_use[scan_index][0], self.dset.indices_in_use[scan_index][1]].decode("utf-8")

        idx = 0
        
        while True:
            
            # start from the second sequence, for comparision with other methods
            
            # in order to let all frames are all based on frames, 
            # two sub-sequences should at least have one frame overlap 
            # for example, 0-99, 99-198, 198-297
            
            

            if (idx + self.opt.NUM_SAMPLES) > frames.shape[1]:
                break

            frames_sub = frames[:,idx:idx + self.opt.NUM_SAMPLES, ...]
            tforms_sub = tforms[:,idx:idx + self.opt.NUM_SAMPLES, ...]
            tforms_inv_sub = tforms_inv[:,idx:idx + self.opt.NUM_SAMPLES, ...]

            # obtain the transformation from current frame to frame 0
            tforms_each_frame2frame0_gt_sub = self.transform_label(tforms_sub, tforms_inv_sub)
            
            # calculate local tarsformation, the previous frame is ground truth
            transf_0 = tforms_each_frame2frame0_gt_sub[:,0:-1,...]
            transf_1 = tforms_each_frame2frame0_gt_sub[:,1:,...]
            tforms_each_frame2frame0_gt_sub_local = torch.matmul(torch.linalg.inv(transf_0),transf_1)
            tforms_each_frame2frame0_gt_sub_local = torch.matmul(transf_0,tforms_each_frame2frame0_gt_sub_local)
            tforms_each_frame2frame0_gt_sub_local = torch.cat((tforms_each_frame2frame0_gt_sub[:,0,...][None,...],tforms_each_frame2frame0_gt_sub_local),1)
            
            with torch.no_grad():
                outputs = self.model(frames_sub)
                # 6 parameter to 4*4 transformation
                pred_transfs = self.transform_prediction(outputs)

                # make the predicted transformations are based on frame 0
                predframe0 = torch.eye(4,4)[None,...].repeat(pred_transfs.shape[0],1, 1,1).to(self.device)
                pred_transfs = torch.cat((predframe0,pred_transfs),1)

                # calculate local transformation
                pred_transfs_0 = pred_transfs[:,0:-1,...]
                pred_transfs_1 = pred_transfs[:,1:,...]
                pred_transfs_local = torch.matmul(torch.linalg.inv(pred_transfs_0),pred_transfs_1)
                pred_transfs_local = torch.matmul(transf_0,pred_transfs_local)
                pred_transfs_local = torch.cat((predframe0,pred_transfs_local),1)
            


                if idx !=0:
                    # if not the first sub-sequence, should be transformed into frame 0
                    tforms_each_frame2frame0_gt_sub = torch.matmul(tform_last_frame[None,...],tforms_each_frame2frame0_gt_sub)
                    pred_transfs = torch.matmul(tform_last_frame_pred[None,...],pred_transfs) 

                    tforms_each_frame2frame0_gt_sub_local = torch.matmul(tform_last_frame_local[None,...],tforms_each_frame2frame0_gt_sub_local)
                    pred_transfs_local = torch.matmul(tform_last_frame_pred_local[None,...],pred_transfs_local) 

                
                tform_last_frame = tforms_each_frame2frame0_gt_sub[:,-1,...]
                tform_last_frame_pred = pred_transfs[:,-1,...]

                tform_last_frame_local = tforms_each_frame2frame0_gt_sub_local[:,-1,...]
                tform_last_frame_pred_local = pred_transfs_local[:,-1,...]

                # obtain the coordinates of each frame, using frame 0 as the reference frame
                if self.opt.img_pro_coord == 'img_coord':
                    labels_gt_sub = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(tforms_each_frame2frame0_gt_sub,torch.matmul(self.tform_calib,self.all_points)))[:,:,0:3,...]
                    # transformtion to points
                    pred_pts_sub = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.all_points)))[:,:,0:3,...]

                    labels_gt_sub_local = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(tforms_each_frame2frame0_gt_sub_local,torch.matmul(self.tform_calib,self.all_points)))[:,:,0:3,...]
                    pred_pts_sub_local = torch.matmul(torch.linalg.inv(self.tform_calib_R_T),torch.matmul(pred_transfs_local,torch.matmul(self.tform_calib,self.all_points)))[:,:,0:3,...]

    
                else:
                    labels_gt_sub = torch.matmul(tforms_each_frame2frame0_gt_sub,torch.matmul(self.tform_calib,self.all_points))[:,:,0:3,...]
                    # transformtion to points
                    pred_pts_sub = torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.all_points))[:,:,0:3,...]
                    
                    labels_gt_sub_local = torch.matmul(tforms_each_frame2frame0_gt_sub_local,torch.matmul(self.tform_calib,self.all_points))[:,:,0:3,...]
                    # transformtion to points
                    pred_pts_sub_local = torch.matmul(pred_transfs_local,torch.matmul(self.tform_calib,self.all_points))[:,:,0:3,...]


                    # for coordinates system change use
                    ori_pts_sub = torch.matmul(tforms_each_frame2frame0_gt_sub,torch.matmul(self.tform_calib,self.all_points)).permute(0,1,3,2)
                    pre_sub = torch.matmul(pred_transfs,torch.matmul(self.tform_calib,self.all_points)).permute(0,1,3,2)
                    # obtain points in optimised coodinates system, and the coorsponding transformation matrix
                    labels_gt_sub_opt, pred_pts_sub_opt,convR_batched,minxyz_all = self.ConvPose(labels_gt_sub, ori_pts_sub, pre_sub, 'auto_PCA',self.device)

                    ori_pts_sub_local = torch.matmul(tforms_each_frame2frame0_gt_sub_local,torch.matmul(self.tform_calib,self.all_points)).permute(0,1,3,2)
                    pre_sub_local = torch.matmul(pred_transfs_local,torch.matmul(self.tform_calib,self.all_points)).permute(0,1,3,2)
                    


                # this is the warpped prediction, represented in optimised coordinates system
                # as there will be common frame in two US sequence, when produce DDF, only one DDF should be generated
                if idx == 0:
                    labels_gt_sub_non_overlap_opt = labels_gt_sub_opt
                    pred_pts_sub_non_overlap_opt = pred_pts_sub_opt
                    frames_sub_non_overlap = frames_sub
                else:
                    labels_gt_sub_non_overlap_opt = labels_gt_sub_opt[:,1:,...]
                    pred_pts_sub_non_overlap_opt = pred_pts_sub_opt[:,1:,...]
                    frames_sub_non_overlap = frames_sub[:,1:,...]

                pred_pts_warped_sub_opt,common_volume = self.intepolation_and_registration_for_each_patch(labels_gt_sub_non_overlap_opt,pred_pts_sub_non_overlap_opt,frames_sub_non_overlap,option = based_volume)
                # convert it into origial coordinates system
                pred_pts_warped_sub_pro = self.convert_from_optimised_to_origin(pred_pts_warped_sub_opt,minxyz_all,convR_batched,labels_gt_sub_non_overlap_opt,common_volume,option = based_volume)
                pred_pts_warped_sub = pred_pts_warped_sub_pro.permute(0,1,3,2)[:,:,0:3,:]
                

            if idx ==0:
                # points in original coordinates system
                labels_gt = labels_gt_sub
                pred_pts = pred_pts_sub
                pred_pts_warped = pred_pts_warped_sub

                labels_gt_local = labels_gt_sub_local 
                pred_pts_local = pred_pts_sub_local 

                ori_pts = ori_pts_sub
                pre = pre_sub
                pred_warped = pred_pts_warped_sub_pro

                ori_pts_local = ori_pts_sub_local
                pre_local = pre_sub_local

                
            else:
                labels_gt = torch.cat((labels_gt,labels_gt_sub[:,1:,...]),1)
                pred_pts = torch.cat((pred_pts,pred_pts_sub[:,1:,...]),1)
                pred_pts_warped = torch.cat((pred_pts_warped,pred_pts_warped_sub),1)

                labels_gt_local = torch.cat((labels_gt_local,labels_gt_sub_local[:,1:,...]),1)
                pred_pts_local = torch.cat((pred_pts_local,pred_pts_sub_local[:,1:,...]),1)

                ori_pts = torch.cat((ori_pts,ori_pts_sub[:,1:,...]),1)
                pre = torch.cat((pre,pre_sub[:,1:,...]),1)

                ori_pts_local = torch.cat((ori_pts_local,ori_pts_sub_local[:,1:,...]),1)
                pre_local = torch.cat((pre_local,pre_sub_local[:,1:,...]),1)

                pred_warped = torch.cat((pred_warped,pred_pts_warped_sub_pro),1)

            if self.option == 'generate_reg_volume_data':
                idx += 1
            elif self.option == "reconstruction_vlume":
                idx += (self.opt.NUM_SAMPLES-1)

        

        T_global_all_dist = ((pred_pts-labels_gt)**2).sum(dim=2).sqrt().mean().item()
        T_R_wrap_global_all_dist = ((pred_pts_warped-labels_gt)**2).sum(dim=2).sqrt().mean().item()
        T_local_all_dist = ((pred_pts_local-labels_gt_local)**2).sum(dim=2).sqrt().mean().item()

        # global transoformation on four corner points
        
        pred_pts_four = pred_pts[...,[0,self.dset[0][0].shape[2]-1,(self.dset[0][0].shape[1]-1)*self.dset[0][0].shape[2],-1]]
        labels_gt_four = labels_gt[...,[0,self.dset[0][0].shape[2]-1,(self.dset[0][0].shape[1]-1)*self.dset[0][0].shape[2],-1]]
        pred_pts_warped_four = pred_pts_warped[...,[0,self.dset[0][0].shape[2]-1,(self.dset[0][0].shape[1]-1)*self.dset[0][0].shape[2],-1]]
        pred_pts_four_local = pred_pts_local[...,[0,self.dset[0][0].shape[2]-1,(self.dset[0][0].shape[1]-1)*self.dset[0][0].shape[2],-1]]
        labels_gt_four_local = labels_gt_local[...,[0,self.dset[0][0].shape[2]-1,(self.dset[0][0].shape[1]-1)*self.dset[0][0].shape[2],-1]]
        
        
        T_global_four_dist = ((pred_pts_four-labels_gt_four)**2).sum(dim=2).sqrt().mean().item()
        T_R_wrap_global_four_dist = ((pred_pts_warped_four-labels_gt_four)**2).sum(dim=2).sqrt().mean().item()
        T_local_four_dist = ((pred_pts_four_local-labels_gt_four_local)**2).sum(dim=2).sqrt().mean().item()
        # global difference on all points
        self.T_Global_AllPts_Dist.append(T_global_all_dist)
        self.T_R_Warp_Global_AllPts_Dist.append(T_R_wrap_global_all_dist)
        self.T_Local_AllPts_Dist.append(T_local_all_dist)

        # global difference on four corner points
        self.T_Global_FourPts_Dist.append(T_global_four_dist)
        self.T_R_Warp_Global_FourPts_Dist.append(T_R_wrap_global_four_dist)
        self.T_Local_FourPts_Dist.append(T_local_four_dist)


        # # plot trajactory based on four corner points
        self.plot_scan(labels_gt_four,pred_pts_four,frames[:,:labels_gt_four.shape[1],...],saved_folder+'/'+saved_name+'_T_global')
        self.plot_scan(labels_gt_four,pred_pts_warped_four,frames[:,:labels_gt_four.shape[1],...],saved_folder+'/'+saved_name+'_TR_global')
        self.plot_scan(labels_gt_four_local,pred_pts_four_local,frames[:,:labels_gt_four.shape[1],...],saved_folder+'/'+saved_name+'_T_local')

        
 
        # for visulise use
        generate_mha = False
        if generate_mha:
            # change labels to a convenient coordinates system
            if self.opt.Conv_Coords == 'optimised_coord' and self.opt.img_pro_coord == 'pro_coord':
                # change the corrdinates system of the origial data (label), such that the occupied volume is smallest
                # note: only calculate the nre coordinates syatem of the groundtruth, and predicted points will use the same transformation as groundtruth 
                
                labels_gt_opt, pred_pts_opt,convR_batched,minxyz_all = self.ConvPose(labels_gt, ori_pts, pre, 'auto_PCA',self.device)
                labels_gt_opt1, pred_pts_warped_opt,convR_batched,minxyz_all = self.ConvPose(labels_gt, ori_pts, pred_warped, 'auto_PCA',self.device)
                
                # check the correctness of the code
                if not torch.all(labels_gt_opt==labels_gt_opt1):
                    raise('transformation from original coordinates system to optimised coordinates system is not correct')
            elif self.opt.Conv_Coords == 'optimised_coord' and self.opt.img_pro_coord != 'pro_coord':
                raise('optimised_coord must be used when pro_coord')

            # generate 

            # intepolete
            # compute the common volume for ground truth and prediction to intepolete
            
            min_x = torch.min(torch.min(torch.min(labels_gt_opt[0,:,0,:]),torch.min(pred_pts_opt[0,:,0,:])),torch.min(pred_pts_warped_opt[0,:,0,:]))
            max_x = torch.max(torch.max(torch.max(labels_gt_opt[0,:,0,:]),torch.max(pred_pts_opt[0,:,0,:])),torch.max(pred_pts_warped_opt[0,:,0,:]))

            min_y = torch.min(torch.min(torch.min(labels_gt_opt[0,:,1,:]),torch.min(pred_pts_opt[0,:,1,:])),torch.min(pred_pts_warped_opt[0,:,1,:]))
            max_y = torch.max(torch.max(torch.max(labels_gt_opt[0,:,1,:]),torch.max(pred_pts_opt[0,:,1,:])),torch.max(pred_pts_warped_opt[0,:,1,:]))

            min_z = torch.min(torch.min(torch.min(labels_gt_opt[0,:,2,:]),torch.min(pred_pts_opt[0,:,2,:])),torch.min(pred_pts_warped_opt[0,:,2,:]))
            max_z = torch.max(torch.max(torch.max(labels_gt_opt[0,:,2,:]),torch.max(pred_pts_opt[0,:,2,:])),torch.max(pred_pts_warped_opt[0,:,2,:]))


            x = torch.linspace((min_x.item()), (max_x.item()), int(((max_x.item())-(min_x.item()))))
            y = torch.linspace((min_y.item()), (max_y.item()), int(((max_y.item())-(min_y.item()))))
            z = torch.linspace((min_z.item()), (max_z.item()), int(((max_z.item())-(min_z.item()))))
            X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
            X, Y, Z =X.to(self.device), Y.to(self.device), Z.to(self.device) 
            common_volume = [X,Y,Z]
            
            gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(scatter_pts = labels_gt_opt,
                                                frames = frames[0,0:labels_gt_opt.shape[1],...],
                                                time_log=None,
                                                saved_folder_test = saved_folder,
                                                scan_name=saved_name+'_gt',
                                                device = self.device,
                                                option = self.opt.intepoletion_method,
                                                volume_position = common_volume,
                                                volume_size = self.opt.intepoletion_volume,

                                                )
            
            pred_volume,pred_volume_position = interpolation_3D_pytorch_batched(scatter_pts = pred_pts_opt,
                                                    frames = frames[0,0:pred_pts_opt.shape[1],...],
                                                    time_log=None,
                                                    saved_folder_test = saved_folder,
                                                    scan_name=saved_name+'_pred',
                                                    device = self.device,
                                                    option = self.opt.intepoletion_method,
                                                    volume_position = common_volume,
                                                    volume_size = self.opt.intepoletion_volume,
                                                    )
            
            pred_volume_warp,pred_volume_position_warp = interpolation_3D_pytorch_batched(scatter_pts = pred_pts_warped_opt,
                                                    frames = frames[0,0:pred_pts_opt.shape[1],...],
                                                    time_log=None,
                                                    saved_folder_test = saved_folder,
                                                    scan_name=saved_name+'_pred',
                                                    device = self.device,
                                                    option = self.opt.intepoletion_method,
                                                    volume_position = common_volume,
                                                    volume_size = self.opt.intepoletion_volume,
                                                    )
            
            print('done')
            # warped, ddf = self.VoxelMorph_net(moving = torch.unsqueeze(pred_volume, 1), 
            #             fixed = torch.unsqueeze(gt_volume, 1))

            
            save2mha(gt_volume[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
                save_folder=saved_folder+'/'+ saved_name + '_gt-test.mha'
                )

            save2mha(pred_volume[0,...].detach().cpu().numpy(),sx = 1,sy=1,sz=1,
                save_folder=saved_folder+'/'+ saved_name + '_pred-test.mha'
                )
            
            save2mha(pred_volume_warp[0,...].detach().cpu().numpy(),sx = 1,sy=1,sz=1,
                save_folder=saved_folder+'/'+ saved_name + '_wrapped-test.mha'
                )


            # generate local transform-based volume

            # change labels to a convenient coordinates system
            if self.opt.Conv_Coords == 'optimised_coord' and self.opt.img_pro_coord == 'pro_coord':
                # change the corrdinates system of the origial data (label), such that the occupied volume is smallest
                # note: only calculate the nre coordinates syatem of the groundtruth, and predicted points will use the same transformation as groundtruth 
                
                labels_gt_opt_local, pred_pts_opt_local,convR_batched_local,minxyz_all_local = self.ConvPose(labels_gt_local, ori_pts_local, pre_local, 'auto_PCA',self.device)
                
            elif self.opt.Conv_Coords == 'optimised_coord' and self.opt.img_pro_coord != 'pro_coord':
                raise('optimised_coord must be used when pro_coord')

            # generate 

            
            min_x = torch.min(torch.min(labels_gt_opt_local[0,:,0,:]),torch.min(pred_pts_opt_local[0,:,0,:]))
            max_x = torch.max(torch.max(labels_gt_opt_local[0,:,0,:]),torch.max(pred_pts_opt_local[0,:,0,:]))

            min_y = torch.min(torch.min(labels_gt_opt_local[0,:,1,:]),torch.min(pred_pts_opt_local[0,:,1,:]))
            max_y = torch.max(torch.max(labels_gt_opt_local[0,:,1,:]),torch.max(pred_pts_opt_local[0,:,1,:]))

            min_z = torch.min(torch.min(labels_gt_opt_local[0,:,2,:]),torch.min(pred_pts_opt_local[0,:,2,:]))
            max_z = torch.max(torch.max(labels_gt_opt_local[0,:,2,:]),torch.max(pred_pts_opt_local[0,:,2,:]))

            x = torch.linspace((min_x.item()), (max_x.item()), int(((max_x.item())-(min_x.item()))))
            y = torch.linspace((min_y.item()), (max_y.item()), int(((max_y.item())-(min_y.item()))))
            z = torch.linspace((min_z.item()), (max_z.item()), int(((max_z.item())-(min_z.item()))))
            X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
            X, Y, Z =X.to(self.device), Y.to(self.device), Z.to(self.device) 
            common_volume = [X,Y,Z]
            
            gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(scatter_pts = labels_gt_opt_local,
                                                frames = frames[0,0:labels_gt_opt_local.shape[1],...],
                                                time_log=None,
                                                saved_folder_test = saved_folder,
                                                scan_name=saved_name+'_gt',
                                                device = self.device,
                                                option = self.opt.intepoletion_method,
                                                volume_position = common_volume,
                                                volume_size = self.opt.intepoletion_volume,

                                                )
            
            pred_volume,pred_volume_position = interpolation_3D_pytorch_batched(scatter_pts = pred_pts_opt_local,
                                                    frames = frames[0,0:pred_pts_opt_local.shape[1],...],
                                                    time_log=None,
                                                    saved_folder_test = saved_folder,
                                                    scan_name=saved_name+'_pred',
                                                    device = self.device,
                                                    option = self.opt.intepoletion_method,
                                                    volume_position = common_volume,
                                                    volume_size = self.opt.intepoletion_volume,
                                                    )
            
            
            
            print('done')
            # warped, ddf = self.VoxelMorph_net(moving = torch.unsqueeze(pred_volume, 1), 
            #             fixed = torch.unsqueeze(gt_volume, 1))

            
            save2mha(gt_volume[0,...].cpu().numpy(),sx = 1,sy=1,sz=1,
                save_folder=saved_folder+'/'+ saved_name + '_gt-test-local.mha'
                )

            save2mha(pred_volume[0,...].detach().cpu().numpy(),sx = 1,sy=1,sz=1,
                save_folder=saved_folder+'/'+ saved_name + '_pred-test-local.mha'
                )
            
           


        


        # print('done')



    def intepolation_and_registration_for_each_patch(self,labels_gt,pred_pts,frames,option):
        # use common volme to generate volume from scatter points
        if option == 'common_volume':
            # compute the common volume for ground truth and prediction to intepolete
            min_x,max_x = torch.min(torch.min(labels_gt[0,:,0,:],pred_pts[0,:,0,:])),torch.max(torch.max(labels_gt[0,:,0,:],pred_pts[0,:,0,:]))
            min_y,max_y = torch.min(torch.min(labels_gt[0,:,1,:],pred_pts[0,:,1,:])),torch.max(torch.max(labels_gt[0,:,1,:],pred_pts[0,:,1,:]))
            min_z,max_z = torch.min(torch.min(labels_gt[0,:,2,:],pred_pts[0,:,2,:])),torch.max(torch.max(labels_gt[0,:,2,:],pred_pts[0,:,2,:]))


            x = torch.linspace((min_x.item()), (max_x.item()), int(((max_x.item())-(min_x.item()))))
            y = torch.linspace((min_y.item()), (max_y.item()), int(((max_y.item())-(min_y.item()))))
            z = torch.linspace((min_z.item()), (max_z.item()), int(((max_z.item())-(min_z.item()))))
            X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
            X, Y, Z =X.to(self.device), Y.to(self.device), Z.to(self.device) 
            common_volume = [X,Y,Z]
            
            gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(scatter_pts = labels_gt,
                                                frames = frames,#[0,0:labels_gt.shape[1],...],
                                                time_log=None,
                                                saved_folder_test = None,
                                                scan_name=None,
                                                device = self.device,
                                                option = self.opt.intepoletion_method,
                                                volume_position = common_volume,
                                                volume_size = self.opt.intepoletion_volume,

                                                )
            
            pred_volume,pred_volume_position = interpolation_3D_pytorch_batched(scatter_pts = pred_pts,
                                                    frames = frames,#[0,0:pred_pts.shape[1],...],
                                                    time_log=None,
                                                    saved_folder_test = None,
                                                    scan_name=None,
                                                    device = self.device,
                                                    option = self.opt.intepoletion_method,
                                                    volume_position = common_volume,
                                                    volume_size = self.opt.intepoletion_volume,
                                                    )
        else:
            raise('TBC')

        if self.opt.ddf_dirc == 'Move':
            # generate wrapped fixed
            warped, ddf = self.VoxelMorph_net(moving = torch.unsqueeze(pred_volume, 1), 
                            fixed = torch.unsqueeze(gt_volume, 1))
            
            pred_pts_warped = self.generate_wraped_prediction(warped, ddf,pred_volume,pred_pts,labels_gt,common_volume,option = option)
        
        else:
            pred_pts_warped = pred_pts
            
        
        return pred_pts_warped,common_volume


    


    def generate_wraped_prediction(self,warped, ddf,pred_volume,pred_pts,gt_pts,common_volume,option):
        # generate warped moving (prediction) from DDF (DDF is the displacement on moving image)
        
        # intopolate into scatter DDF from grid DDF
        # ddf_scatter = torch.zeros(ddf.shape[0],ddf.shape[1],pred_pts.shape)
        # reshapa and permute to satisfy grid_sample requirment 
        pred_pts = torch.reshape(pred_pts, (pred_pts.shape[0],pred_pts.shape[1],pred_pts.shape[2],self.dset[0][0].shape[1],self.dset[0][0].shape[2])).permute(0,1,3,4,2)
        
        #  normalized into [0,N], which has no unite, float type
        # as DDF is in range [0,N], which is defined by voxelMorph, such that prediction points locations should be 
        # normalised into the same space, and then ADD operation can be added
        pred_pts_norm = torch.zeros_like(pred_pts)
        pred_pts_0_1 = torch.zeros_like(pred_pts)
        
        min_x,max_x = torch.min(common_volume[0]),torch.max(common_volume[0])
        min_y,max_y = torch.min(common_volume[1]),torch.max(common_volume[1])
        min_z,max_z = torch.min(common_volume[2]),torch.max(common_volume[2])
        minxyz = torch.from_numpy(np.array([min_x.item(),min_y.item(),min_z.item()]))

        
        for i in range(pred_pts.shape[-1]):
            # gt_pts is in convinient system
            # the operatipon of normalise of prediction points should be exactly the same as the operation for intepolation for
            # ground truth points and prediction points
            if option == 'gt_based_volume':
                pred_pts_norm[...,i] = (pred_pts[...,i]-torch.min(gt_pts[:,:,i,:]))/1 # 1 is the spaceing, which should be consistent with the value in intepolation function
            elif option == 'common_volume':
                
                pred_pts_norm[...,i] = (pred_pts[...,i]-minxyz[i])/1 # 1 is the spaceing, which should be consistent with the value in intepolation function

            else:
                raise('TBC')
        # normalise pred_pts into [-1,1] to satisfy the grid_sample requirment
        for i, dim in enumerate(ddf.shape[2:]):#[-1:-4:-1]
            pred_pts_0_1[..., 2-i] = pred_pts_norm[..., i] * 2 / (dim - 1) - 1

        

        ddf_scatter = torch.nn.functional.grid_sample(
                                                input = ddf, 
                                                grid = pred_pts_0_1, 
                                                mode='bilinear', 
                                                padding_mode='zeros', align_corners=False)
        

        # generate warpped moving/prediction image
        ddf_scatter = ddf_scatter.permute(0,2,3,4,1)
        # ddf_scatter = ddf_scatter.permute(0,4,2,3,1)
        pred_pts_warped = pred_pts_norm+ddf_scatter
        pred_pts_warped = pred_pts_warped.permute(0,1,4,2,3)
        pred_pts_warped = torch.reshape(pred_pts_warped,(pred_pts_warped.shape[0],pred_pts_warped.shape[1],pred_pts_warped.shape[2],-1))


        return pred_pts_warped


    def calculateConvPose_batched(self,pts_batched,option,device):
        for i_batch in range(pts_batched.shape[0]):
        
            ConvR = self.calculateConvPose(pts_batched[i_batch,...],option,device)
            # ConvR = ConvR.repeat(pts_batched[i_batch,...].shape[0], 1,1)[None,...]
            ConvR = ConvR[None,...]
            if i_batch == 0:
                ConvR_batched = ConvR
            else:
                ConvR_batched = torch.cat((ConvR_batched,ConvR),0)
            return ConvR_batched
            

    def calculateConvPose_batched1(self,pts_batched,option,device):
        for i_batch in range(pts_batched.shape[0]):
            
            ConvR = self.calculateConvPose(pts_batched[i_batch,...],option,device)
            # ConvR = ConvR.repeat(pts_batched[i_batch,...].shape[0], 1,1)[None,...]
            ConvR = ConvR[None,...]
            if i_batch == 0:
                ConvR_batched = ConvR
            else:
                ConvR_batched = torch.cat((ConvR_batched,ConvR),0)
        return ConvR_batched


    def calculateConvPose(self,pts,option,device):
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
                U, s = self.pca(torch.transpose(pts1, 0, 1)) 
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

    def ConvPose(self,labels,ori_pts,pre, option_method,device):
        convR_batched = self.calculateConvPose_batched(labels,option = option_method,device=device)    
        
        for i_batch in range(convR_batched.shape[0]):
            
            labels_i = torch.matmul(ori_pts[i_batch,...],convR_batched[i_batch,...])[None,...]
            minx = torch.min(labels_i[...,0])
            miny = torch.min(labels_i[...,1])
            minz = torch.min(labels_i[...,2])
            labels_i[...,0]-=minx
            labels_i[...,1]-=miny
            labels_i[...,2]-=minz

            pred_pts_i = torch.matmul(pre[i_batch,...],convR_batched[i_batch,...])[None,...]
            
            pred_pts_i[...,0]-=minx
            pred_pts_i[...,1]-=miny
            pred_pts_i[...,2]-=minz

            # return for future use
            minxyz=torch.from_numpy(np.array([minx.item(),miny.item(),minz.item()]))

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
        
                
        
        return labels_opt,pred_pts_opt,convR_batched,minxyz_all

    def pca(self,D):
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

    def convert_from_optimised_to_origin(self,pred_pts_warped_sub,minxyz_all,convR_batched,labels_gt_sub_opt,common_volume,option = 'common_volume'):
        
        

        if option == 'gt_volume':
        
            for i in range(pred_pts_warped_sub.shape[2]):
                pred_pts_warped_sub[:,:,i,:] = pred_pts_warped_sub[:,:,i,:]*1+torch.min(labels_gt_sub_opt[:,:,i,:]) # 1 is the spaceing, which should be consistent with the value in intepolation function
        elif option == 'common_volume':
            min_x,max_x = torch.min(common_volume[0]),torch.max(common_volume[0])
            min_y,max_y = torch.min(common_volume[1]),torch.max(common_volume[1])
            min_z,max_z = torch.min(common_volume[2]),torch.max(common_volume[2])
            minxyz = torch.from_numpy(np.array([min_x.item(),min_y.item(),min_z.item()]))



            for i in range(pred_pts_warped_sub.shape[2]):
                pred_pts_warped_sub[:,:,i,:] = pred_pts_warped_sub[:,:,i,:]*1+minxyz[i] # 1 is the spaceing, which should be consistent with the value in intepolation function
        
        else:
            raise('TBC')

        
        if pred_pts_warped_sub.shape[0]==1:
            pred_pts_warped_sub[:,:,0,:]+=minxyz_all[0]
            pred_pts_warped_sub[:,:,1,:]+=minxyz_all[1]
            pred_pts_warped_sub[:,:,2,:]+=minxyz_all[2]

            pred_pts_warped_sub = pred_pts_warped_sub.permute(0,1,3,2)
            pred_pts_warped_sub_pad = F.pad(input=pred_pts_warped_sub, pad=(0, 1, 0, 0), mode='constant', value=1)
            pred_pts_warped_sub_ori = torch.matmul(pred_pts_warped_sub_pad,torch.linalg.inv(convR_batched))

        elif pred_pts_warped_sub.shape[0]>1:
            raise('batched not implemented')

        return pred_pts_warped_sub_ori
        

    def plot_scan(self,labels_gt_four,pred_pts_four,frames,saved_name):

        # save numpy file, for uture plot use
        all_frames_fd = '/'+'/'.join(saved_name.split('/')[1:7])+'/frames_in_testset'
        if not os.path.exists(all_frames_fd):
            os.makedirs(all_frames_fd)

        all_gt_fd = '/'+'/'.join(saved_name.split('/')[1:7])+'/gts_in_testset'
        if not os.path.exists(all_gt_fd):
            os.makedirs(all_gt_fd)
    

        with open(all_gt_fd+'/'+saved_name.split('/')[-1]+'_gt.npy', 'wb') as f:
            np.save(f, labels_gt_four.cpu().numpy())
        with open(saved_name+'_pred.npy', 'wb') as f:
            np.save(f, pred_pts_four.cpu().numpy())
        with open(all_frames_fd+'/'+saved_name.split('/')[-1]+'_frame.npy', 'wb') as f:
            np.save(f, frames.cpu().numpy())

        ax = plt.figure().add_subplot(projection='3d')
        
        if labels_gt_four.shape[0]==1:
            fx, fy, fz = [labels_gt_four[:,0,...].cpu().numpy()[:,ii,].reshape(2, 2) for ii in range(3)]
            pix_intensities = (frames[0,0, ..., None].float() / 255).expand(-1, -1, 3).cpu().numpy()
        else:
            raise('the first dimention must be 1')

        ax.plot_surface(fx, fy, fz, facecolors=pix_intensities, linewidth=0, antialiased=True)
        ax.plot_surface(fx, fy, fz, edgecolor='g', linewidth=1, alpha=0.2, antialiased=True)

        gx_all, gy_all, gz_all = [labels_gt_four[:, :, ii, :].cpu().numpy() for ii in range(3)]
        prex_all, prey_all, prez_all = [pred_pts_four[:, :, ii, :].cpu().numpy() for ii in range(3)]
        ax.scatter(gx_all, gy_all, gz_all, c='g', alpha=0.2, s=2)
        ax.scatter(prex_all, prey_all, prez_all, c='r', alpha=0.2, s=2)
        
        # plot the last image
        gx, gy, gz = [labels_gt_four[:, -1, ii, :].cpu().numpy() for ii in range(3)]
        prex, prey, prez = [pred_pts_four[:, -1, ii, :].cpu().numpy() for ii in range(3)]


        gx, gy, gz = gx.reshape(2, 2), gy.reshape(2, 2), gz.reshape(2, 2)
        ax.plot_surface(gx, gy, gz, edgecolor='g', linewidth=1, alpha=0.2, antialiased=True, label='gt')#
        prex, prey, prez = prex.reshape(2, 2), prey.reshape(2, 2), prez.reshape(2, 2)
        ax.plot_surface(prex, prey, prez, edgecolor='r', linewidth=1, alpha=0.2, antialiased=True, label='pred')
        ax.axis('equal')
        ax.legend()


        plt.savefig(saved_name+'.png')
        plt.savefig(saved_name+'.pdf')

        plt.close()
