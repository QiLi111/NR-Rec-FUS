import torch
import numpy as np
import torch.nn.functional as F

def compute_common_volume(labels_gt,pred_pts,device):
    # compute the common volume for ground truth and prediction to intepolete
    min_x,max_x = torch.min(torch.min(labels_gt[0,:,0,:],pred_pts[0,:,0,:])),torch.max(torch.max(labels_gt[0,:,0,:],pred_pts[0,:,0,:]))
    min_y,max_y = torch.min(torch.min(labels_gt[0,:,1,:],pred_pts[0,:,1,:])),torch.max(torch.max(labels_gt[0,:,1,:],pred_pts[0,:,1,:]))
    min_z,max_z = torch.min(torch.min(labels_gt[0,:,2,:],pred_pts[0,:,2,:])),torch.max(torch.max(labels_gt[0,:,2,:],pred_pts[0,:,2,:]))


    x = torch.linspace((min_x.item()), (max_x.item()), int(((max_x.item())-(min_x.item()))))
    y = torch.linspace((min_y.item()), (max_y.item()), int(((max_y.item())-(min_y.item()))))
    z = torch.linspace((min_z.item()), (max_z.item()), int(((max_z.item())-(min_z.item()))))
    X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
    X, Y, Z =X.to(device), Y.to(device), Z.to(device) 
    common_volume = [X,Y,Z]

    return common_volume

def wrapped_pred_dist(ddf,pred_pts,gt_pts,option, H, W,convR_batched,minxyz_all,device):
    # compute the loss distence between wrapped prediction and ground truth,
    # where wrapped prediction is the predicted points + DDF, and DDF is the displacement vector based on prediction volume (moving)
    
    # Note: pred_pts are gt_pts base on the optimised coordinates system
    common_volume = compute_common_volume(gt_pts,pred_pts,device)

    pred_pts = torch.reshape(pred_pts, (pred_pts.shape[0],pred_pts.shape[1],pred_pts.shape[2],H,W)).permute(0,1,3,4,2)
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
        if option == 'common_volume':
            
            pred_pts_norm[...,i] = (pred_pts[...,i]-minxyz[i])/1 # 1 is the spaceing, which should be consistent with the value in intepolation function

        else:
            raise('TBC')
        
    for i, dim in enumerate(ddf.shape[2:]):#[-1:-4:-1]
        pred_pts_0_1[..., 2-i] = pred_pts_norm[..., i] * 2 / (dim - 1) - 1


    ddf_scatter = torch.nn.functional.grid_sample(
                                            input = ddf, 
                                            grid = pred_pts_0_1, 
                                            mode='bilinear', 
                                            padding_mode='zeros', align_corners=False)
    

    # generate warpped moving/prediction image
    ddf_scatter = ddf_scatter.permute(0,2,3,4,1)
    pred_pts_warped = pred_pts_norm+ddf_scatter
    pred_pts_warped = pred_pts_warped.permute(0,1,4,2,3)
    pred_pts_warped = torch.reshape(pred_pts_warped,(pred_pts_warped.shape[0],pred_pts_warped.shape[1],pred_pts_warped.shape[2],-1))

    # denormalise
    pred_pts_warped_ori = convert_from_optimised_to_origin(pred_pts_warped,minxyz_all,convR_batched,common_volume,option)

    # compute distance
    criterion = torch.nn.MSELoss()

    mseloss = criterion(pred_pts_warped_ori, gt_pts)
    dist = ((pred_pts_warped_ori-gt_pts)**2).sum(dim=2).sqrt().mean()


    return mseloss,dist,pred_pts_warped_ori



def convert_from_optimised_to_origin(pred_pts_warped_sub,minxyz_all,convR_batched,common_volume,option = 'common_volume'):

    # 0-N space to float space
    if option == 'common_volume':
        min_x,max_x = torch.min(common_volume[0]),torch.max(common_volume[0])
        min_y,max_y = torch.min(common_volume[1]),torch.max(common_volume[1])
        min_z,max_z = torch.min(common_volume[2]),torch.max(common_volume[2])
        minxyz = torch.from_numpy(np.array([min_x.item(),min_y.item(),min_z.item()]))
        for i in range(pred_pts_warped_sub.shape[2]):
            pred_pts_warped_sub[:,:,i,:] = pred_pts_warped_sub[:,:,i,:]*1+minxyz[i] # 1 is the spaceing, which should be consistent with the value in intepolation function
    
    else:
        raise('TBC')
    
    # # optimised coordinates system to original coordinates system
    # if pred_pts_warped_sub.shape[0]==1:
    #     pred_pts_warped_sub[:,:,0,:]+=minxyz_all[0]
    #     pred_pts_warped_sub[:,:,1,:]+=minxyz_all[1]
    #     pred_pts_warped_sub[:,:,2,:]+=minxyz_all[2]

    #     pred_pts_warped_sub = pred_pts_warped_sub.permute(0,1,3,2)
    #     pred_pts_warped_sub_pad = F.pad(input=pred_pts_warped_sub, pad=(0, 1, 0, 0), mode='constant', value=1)
    #     pred_pts_warped_sub_ori = torch.matmul(pred_pts_warped_sub_pad,torch.linalg.inv(convR_batched))

    # elif pred_pts_warped_sub.shape[0]>1:
    #     # raise('batched not implemented')

    #     for i in range(pred_pts_warped_sub.shape[0]):
    #         pred_pts_warped_sub[i,:,0,:]+=minxyz_all[i][0]
    #         pred_pts_warped_sub[i,:,1,:]+=minxyz_all[i][1]
    #         pred_pts_warped_sub[i,:,2,:]+=minxyz_all[i][2]

    #     pred_pts_warped_sub = pred_pts_warped_sub.permute(0,1,3,2)
    #     pred_pts_warped_sub_pad = F.pad(input=pred_pts_warped_sub, pad=(0, 1, 0, 0), mode='constant', value=1)
        
    #     pred_pts_warped_sub_ori = torch.zeros_like(pred_pts_warped_sub)
    #     for i in range(pred_pts_warped_sub.shape[0]):
    #         pred_pts_warped_sub_ori[i,...] = torch.matmul(pred_pts_warped_sub_pad[i,...],torch.linalg.inv(convR_batched[i,...]))



    return pred_pts_warped_sub






    




