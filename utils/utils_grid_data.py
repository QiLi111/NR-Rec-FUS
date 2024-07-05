""" invdisttree.py: inverse-distance-weighted interpolation using KDTree
    fast, solid, local
"""
from __future__ import division
import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree as KDTree
# from mayavi import mlab
import matplotlib.pyplot as plt
import torch, time

def save2mha(data,sx,sy,sz,save_folder):
    # save 3D volume into volume, and then can use 3D slice to look

    img=sitk.GetImageFromArray(data.transpose([2,1,0])) # ZYX
    img.SetSpacing((sx,sy,sz))
    sitk.WriteImage(img,save_folder)

def save2img(x,y,z,data,save_folder):
    # volume = mlab.pipeline.volume(mlab.pipeline.scalar_field(data), 
    #                             #   vmin=0, 
    #                             #   vmax=0.8
    #                               )

    # mlab.draw()
    # mlab.savefig(save_folder)
    data_ravel = data.ravel()
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(
        x, y, z, 
               c=(data_ravel - np.min(data_ravel)) / (np.max(data_ravel) - np.min(data_ravel)), 
               cmap=plt.get_cmap('Greys')
               )

    # plt.show()
    plt.savefig(save_folder)
    plt.close()





def eight_neighbour_points(x,y,z):
    # find the eight_neighbour_points
    # in this case, the sum of x_low and x_up must be equal to 1

    x_low = torch.floor(x).to(torch.int)
    y_low = torch.floor(y).to(torch.int)
    z_low = torch.floor(z).to(torch.int)

    return x_low,x_low+1,y_low,y_low+1,z_low,z_low+1

def eright_points_in_1d(x_low,x_up,y_low,y_up,z_low,z_up,xl,yl,zl,initial):
    # get the coordinates of the erights neighbours of all scatter points
    # as the coordinates of scatter points are nomalised, without unit, such that the 
    # coordinates of eright grid neighbours points are also the index of these points

    x_up[x_up >= xl] = xl-1
    y_up[y_up >= yl] = yl-1
    z_up[z_up >= zl] = zl-1

    # get the coordinates of erights neighbours points for all scatter points

    p1 = torch.stack((x_low,y_low,z_low),dim=1)
    p2 = torch.stack((x_low,y_low,z_up),dim=1)
    p3 = torch.stack((x_low,y_up,z_low),dim=1)
    p4 = torch.stack((x_low,y_up,z_up),dim=1)

    p5 = torch.stack((x_up,y_low,z_low),dim=1)
    p6 = torch.stack((x_up,y_low,z_up),dim=1)
    p7 = torch.stack((x_up,y_up,z_low),dim=1)
    p8 = torch.stack((x_up,y_up,z_up),dim=1)

    # get the 1-d index of each points

    p1_1d = xyz2idx(p1,xl,yl,zl)
    p2_1d = xyz2idx(p2,xl,yl,zl)
    p3_1d = xyz2idx(p3,xl,yl,zl)
    p4_1d = xyz2idx(p4,xl,yl,zl)

    p5_1d = xyz2idx(p5,xl,yl,zl)
    p6_1d = xyz2idx(p6,xl,yl,zl)
    p7_1d = xyz2idx(p7,xl,yl,zl)
    p8_1d = xyz2idx(p8,xl,yl,zl)

    # update batched index
    p1_1d = p1_1d + initial*xl*yl*zl
    p2_1d = p2_1d + initial*xl*yl*zl
    p3_1d = p3_1d + initial*xl*yl*zl
    p4_1d = p4_1d + initial*xl*yl*zl

    p5_1d = p5_1d + initial*xl*yl*zl
    p6_1d = p6_1d + initial*xl*yl*zl
    p7_1d = p7_1d + initial*xl*yl*zl
    p8_1d = p8_1d + initial*xl*yl*zl

    return torch.cat((p1_1d,p2_1d,p3_1d,p4_1d,p5_1d,p6_1d,p7_1d,p8_1d))#.to(torch.long)



def weight_intensity_in_1d(x_low,x_up,y_low,y_up,z_low,z_up,x_norm,y_norm,z_norm,frames_flatten,options):
    # compute weight for each grid points
    # NOTE the order of the stacked eight neighbour points should be the same as in function `eright_points_in_1d`

    if options == 'bilinear':

        weight4pixel_x_low_y_low_z_low = (x_up-x_norm)*(y_up-y_norm)*(z_up-z_norm)   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
        weight4pixel_x_low_y_low_z_up =(x_up-x_norm)*(y_up-y_norm)*(z_norm-z_low)  #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
        weight4pixel_x_low_y_up_z_low =(x_up-x_norm)*(y_norm-y_low)*(z_up-z_norm)
        weight4pixel_x_low_y_up_z_up  =(x_up-x_norm)*(y_norm-y_low)*(z_norm-z_low)

        weight4pixel_x_up_y_low_z_low  =(x_norm-x_low)*(y_up-y_norm)*(z_up-z_norm)
        weight4pixel_x_up_y_low_z_up  =(x_norm-x_low)*(y_up-y_norm)*(z_norm-z_low)
        weight4pixel_x_up_y_up_z_low  =(x_norm-x_low)*(y_norm-y_low)*(z_up-z_norm)
        weight4pixel_x_up_y_up_z_up  =(x_norm-x_low)*(y_norm-y_low)*(z_norm-z_low)

        intensity4pixel_x_low_y_low_z_low = frames_flatten*(x_up-x_norm)*(y_up-y_norm)*(z_up-z_norm)   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
        intensity4pixel_x_low_y_low_z_up =frames_flatten*(x_up-x_norm)*(y_up-y_norm)*(z_norm-z_low)  #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
        intensity4pixel_x_low_y_up_z_low =frames_flatten*(x_up-x_norm)*(y_norm-y_low)*(z_up-z_norm)
        intensity4pixel_x_low_y_up_z_up  =frames_flatten*(x_up-x_norm)*(y_norm-y_low)*(z_norm-z_low)

        intensity4pixel_x_up_y_low_z_low  =frames_flatten*(x_norm-x_low)*(y_up-y_norm)*(z_up-z_norm)
        intensity4pixel_x_up_y_low_z_up  =frames_flatten*(x_norm-x_low)*(y_up-y_norm)*(z_norm-z_low)
        intensity4pixel_x_up_y_up_z_low  =frames_flatten*(x_norm-x_low)*(y_norm-y_low)*(z_up-z_norm)
        intensity4pixel_x_up_y_up_z_up  =frames_flatten*(x_norm-x_low)*(y_norm-y_low)*(z_norm-z_low)
    
    elif options == 'IDW':

        weight4pixel_x_low_y_low_z_low = 1/torch.sqrt((x_low-x_norm)**2+ (y_low-y_norm)**2 + (z_low-z_norm)**2)   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
        weight4pixel_x_low_y_low_z_up =1/torch.sqrt((x_low-x_norm)**2+ (y_low-y_norm)**2 + (z_up-z_norm)**2)   #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
        weight4pixel_x_low_y_up_z_low =1/torch.sqrt((x_low-x_norm)**2+ (y_up-y_norm)**2 + (z_low-z_norm)**2) 
        weight4pixel_x_low_y_up_z_up  =1/torch.sqrt((x_low-x_norm)**2+ (y_up-y_norm)**2 + (z_up-z_norm)**2) 

        weight4pixel_x_up_y_low_z_low  =1/torch.sqrt((x_up-x_norm)**2+ (y_low-y_norm)**2 + (z_low-z_norm)**2)
        weight4pixel_x_up_y_low_z_up  =1/torch.sqrt((x_up-x_norm)**2+ (y_low-y_norm)**2 + (z_up-z_norm)**2)
        weight4pixel_x_up_y_up_z_low  =1/torch.sqrt((x_up-x_norm)**2+ (y_up-y_norm)**2 + (z_low-z_norm)**2)
        weight4pixel_x_up_y_up_z_up  =1/torch.sqrt((x_up-x_norm)**2+ (y_up-y_norm)**2 + (z_up-z_norm)**2)

        intensity4pixel_x_low_y_low_z_low = frames_flatten*weight4pixel_x_low_y_low_z_low   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
        intensity4pixel_x_low_y_low_z_up =frames_flatten*weight4pixel_x_low_y_low_z_up #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
        intensity4pixel_x_low_y_up_z_low =frames_flatten*weight4pixel_x_low_y_up_z_low
        intensity4pixel_x_low_y_up_z_up  =frames_flatten*weight4pixel_x_low_y_up_z_up

        intensity4pixel_x_up_y_low_z_low  =frames_flatten*weight4pixel_x_up_y_low_z_low
        intensity4pixel_x_up_y_low_z_up  =frames_flatten*weight4pixel_x_up_y_low_z_up
        intensity4pixel_x_up_y_up_z_low  =frames_flatten*weight4pixel_x_up_y_up_z_low
        intensity4pixel_x_up_y_up_z_up  =frames_flatten*weight4pixel_x_up_y_up_z_up
    else:
        raise("Not supported")


    weight4pixel_8neighbour_pts = torch.cat((weight4pixel_x_low_y_low_z_low,weight4pixel_x_low_y_low_z_up,
                              weight4pixel_x_low_y_up_z_low,weight4pixel_x_low_y_up_z_up,
                            weight4pixel_x_up_y_low_z_low,weight4pixel_x_up_y_low_z_up,
                            weight4pixel_x_up_y_up_z_low,weight4pixel_x_up_y_up_z_up
                            ))
    
  
  
    intensity4pixel_8neighbour_pts = torch.cat((intensity4pixel_x_low_y_low_z_low,intensity4pixel_x_low_y_low_z_up,
                                intensity4pixel_x_low_y_up_z_low,intensity4pixel_x_low_y_up_z_up,
                                intensity4pixel_x_up_y_low_z_low,intensity4pixel_x_up_y_low_z_up,
                                intensity4pixel_x_up_y_up_z_low,intensity4pixel_x_up_y_up_z_up
                                ))

    return weight4pixel_8neighbour_pts,intensity4pixel_8neighbour_pts


def combine_values(weight4pixel,intensity4pixel,weight4pixel_8neighbour_pts,intensity4pixel_8neighbour_pts,idx_1d):
    # combine weighs and weighted inetnsity for the same grid points

    weight4pixel.scatter_add_(0, idx_1d.to(torch.long), weight4pixel_8neighbour_pts)
    intensity4pixel.scatter_add_(0, idx_1d.to(torch.long), intensity4pixel_8neighbour_pts)

    return weight4pixel,intensity4pixel


def xyz2idx(xyz,xl,yl,zl):
    # Transform coordinates of a 3D volume of certain sizes into a list of indices of 1D array.
    index_1d = (xyz[:,2]+xyz[:,1]*zl+xyz[:,0]*(yl*zl))
    return index_1d

    

def interpolation_3D_pytorch_batched(scatter_pts,frames,time_log,saved_folder_test,scan_name,device,option,volume_size,volume_position = None):
    # interpolate 
    # given scatter points in 3D, return the grid points

    # inteplote from scatter data to grid data
    # different from typical inteplotion methods, we index all the scattered points,
    # compute the contribution to the nergbours
    '''

    The idea is to loop all scatter points, compute the contribution to the 2**3 neighbour grid points, 
    and then sum those contributions which are into the same grid points from difference scatter points
    
    intensity of grid points = (weight1*intensity1 + weight2*intensity2 + ... + weightn*intensityn)/(weight1+weight2+..._weightn)
    where n denotes the number of scatter points which has contributions into this grid point

    The core iead is to find the index of contribution scatter points for each grid point
    and sum up all the contribution for each grid point



    Two things are import, the first one is to normalise the coordinates of the scatter points to a unitless one,
    the second is to convert the interpolation problem into a math problem - sum up all the contributions into the same grid points, which can be regarded as a math problem - sum up with known index

    Steps:
    1) Normalise the coordinates of the scatter points to a unitless one, such that the coordinates of the grid points (which is the 8 neigibour grid points computed using torch.floor or torch.ceil) is also the index of this point
       Normalise method: (pts - min(pts))/ step
       where
       xsize,ysize,zsize = int(((max_x)-(min_x))),int(((max_y)-(min_y))),int(((max_z)-(min_z))) #modify xsize as you like
       xstep,ystep,zstep = (max_x-min_x)/(xsize-1),(max_y-min_y)/(ysize-1),(max_z-min_z)/(zsize-1)
        
       For exmaple, in 1D, scatter point A = 3.5, B = 5.6, C = 8.2, D = 10.7
       either size or step is specified
       for example, size = 4
       step = (10.7-3.5)/(4-1) = 2.4
       grid = {}
       after normalisition, A = (3.5-3.5)/2.4 = 0, B = (5.6-3.5)/2.4 = 0.875, C = (8.2-3.5)/2.4 = 1.958, D = (10.7-3.5)/2.4 = 3.0
       after torch.floor, A_low = 1, B_low = 2, C_low = 3

    2) Find 8 neighbour grid points of all scatter points. NOTE: don't use torch.floor and torch.ceil simultaneously, as the distance between x_low and X_up should be eauql to 1.
       In terms of integers, torch.floor and torch.ceil will both be the integers
       Use torch.floor and torch.floor + 1 instead.

    3) For each scatter point, compute the contribution for the 8 neighbour grid points
       Can use a simple bilinear method: the weight for the grid data can be the (1-distance) to the grid data, the final weight is multiplied by each sxis
       for example, for grid points x_low, y_low, z_low =  (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
    
    4) convert the 3d array index into 1d index to use TORCH.TENSOR.SCATTER_ADD_ function

       given the volume = torch.zeros((X_shape,Y_shape,Z_shape)),
       the 3D index (x,y,z) can be converted to 1D index: z + y* Z_shape + x * (Z_shape * Y_shape)

    5) sum up contributions into the same grid data uising TORCH.TENSOR.SCATTER_ADD_
    
    '''

    batchsize = scatter_pts.shape[0]
 
    scatter_pts = torch.permute(scatter_pts[:,:,0:3,:], (0,1,3, 2))
    scatter_pts = torch.reshape(scatter_pts,(-1,scatter_pts.shape[-1]))
    frames_flatten = frames.flatten()

    # only intepolete the grid points within the ground truth inteploted volume
    if volume_position != None:
        
        pts_per_seq = int(scatter_pts.shape[0]/batchsize)
       
        min_x,max_x = torch.min(volume_position[0]),torch.max(volume_position[0])
        min_y,max_y = torch.min(volume_position[1]),torch.max(volume_position[1])
        min_z,max_z = torch.min(volume_position[2]),torch.max(volume_position[2])

        # obtain the index that within the groundtruth intepoleted volume
        inside_min_x = torch.where(scatter_pts[:,0] >= min_x, 1, 0)
        inside_max_x = torch.where(scatter_pts[:,0] <= max_x, 1, 0)
        inside_min_y = torch.where(scatter_pts[:,1] >= min_y, 1, 0)
        inside_max_y = torch.where(scatter_pts[:,1] <= max_y, 1, 0)
        inside_min_z = torch.where(scatter_pts[:,2] >= min_z, 1, 0)
        inside_max_z = torch.where(scatter_pts[:,2] <= max_z, 1, 0)

        index_inside = inside_min_x * inside_max_x * inside_min_y * inside_max_y * inside_min_z * inside_max_z
        scatter_pts = scatter_pts[index_inside==1,:]
        frames_flatten = frames_flatten[index_inside==1]

        
        pts_per_batch=[]
        for i_batch in range(0,batchsize):
            pts_per_batch.append(torch.sum(index_inside[pts_per_seq*i_batch:pts_per_seq*(i_batch+1)]==1))
        
        initial = torch.zeros(pts_per_batch[0])
        for i_batch in range(1,batchsize):
            initial = torch.cat((initial,i_batch*torch.ones(pts_per_batch[i_batch])),dim=0) 
        initial = initial.to(device)

    else:  
    
        min_x,max_x = torch.min(scatter_pts[:,0]),torch.max(scatter_pts[:,0])
        min_y,max_y = torch.min(scatter_pts[:,1]),torch.max(scatter_pts[:,1])
        min_z,max_z = torch.min(scatter_pts[:,2]),torch.max(scatter_pts[:,2])

        pts_per_seq = int(scatter_pts.shape[0]/batchsize)
        initial = torch.zeros(pts_per_seq)
        for i_batch in range(1,batchsize):
            initial = torch.cat((initial,i_batch*torch.ones(pts_per_seq)),dim=0) 
        initial = initial.to(device)

    
    if volume_size == 'fixed_interval':

        # x = torch.linspace((min_x.item()), (max_x.item()), int(((max_x.item())-(min_x.item()))))
        # y = torch.linspace((min_y.item()), (max_y.item()), int(((max_y.item())-(min_y.item()))))
        # z = torch.linspace((min_z.item()), (max_z.item()), int(((max_z.item())-(min_z.item()))))
        
        x = torch.linspace((min_x), (max_x), int(((max_x)-(min_x))))
        y = torch.linspace((min_y), (max_y), int(((max_y)-(min_y))))
        z = torch.linspace((min_z), (max_z), int(((max_z)-(min_z))))
        
        X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
        X, Y, Z =X.to(device), Y.to(device), Z.to(device) 

        # number of pixels
        xsize,ysize,zsize = int(((max_x.item())-(min_x.item())))+2,int(((max_y.item())-(min_y.item())))+2,int(((max_z.item())-(min_z.item())))+2 #modify xsize as you like
        # set the spacing as 1mm, can also set a fixed dimention of the volume, then 
        # each reconstructed US volume will have the same size. Such that can batched into the NN
        # in this case, xsize = fixed_x_size, ysize = fixed_y_size, zsize = fixed_z_size,
        # where fixed_x_size,fixed_y_size,fixed_z_size can be the maxmium value that GPU can handle
        # If the input US sequence has the same length, the number of frames can be set as fixed_x_size
        
        # can be set st 2,2,2 to have a smaller size
        xstep,ystep,zstep = 1,1,1#(max_x.item()-min_x.item())/(xsize-1),(max_y.item()-min_y.item())/(ysize-1),(max_z.item()-min_z.item())/(zsize-1)
       
        # the shape of the volume should be divisiable by 16, such that this can be taken by monai.voxelmorph
        X_shape = max(X.shape[0]+(16 - X.shape[0]%16),X.shape[0]+1)
        Y_shape = max(X.shape[1]+(16 - X.shape[1]%16),X.shape[1]+1)
        Z_shape = max(X.shape[2]+(16 - X.shape[2]%16),X.shape[2]+1)
        
    elif volume_size == 'fixed_volume_size':
    # set a fixed volume size as 128*128*128
    
        volume_size = 127

        x = torch.linspace(0, volume_size, volume_size)
        y = torch.linspace(0, volume_size, volume_size)
        z = torch.linspace(0, volume_size, volume_size)
        X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
        X, Y, Z =X.to(device), Y.to(device), Z.to(device) 
        xsize,ysize,zsize = volume_size,volume_size,volume_size #modify xsize as you like
        xstep,ystep,zstep = (max_x.item()-min_x.item())/(xsize-1),(max_y.item()-min_y.item())/(ysize-1),(max_z.item()-min_z.item())/(zsize-1)
        X_shape = X.shape[0]+1
        Y_shape = X.shape[1]+1
        Z_shape = X.shape[2]+1


    # the initialised value should be pay attention to and this should be same as the background value in the US image,
    # pay attention to whether the image is normalised using (*-min)/(max-min) or (*-mean)/std
    # if the first one, as the orignal pixel value of background is 0, so no difference between backdround in the two kinds of normalisition
    # if the second one, the backgorund value would not be 0. In this case, the initialised value 0 is not acceptable
    
    

   

    # if scan_name.startswith('pred_step'):
    #     weight4pixel = Variable(torch.zeros((batchsize,X_shape,Y_shape,Z_shape)).to(device), requires_grad=True).clone()#torch.full(X.shape, torch.nan).to(device) #torch.from_numpy(np.asarray([[[None]*(X.shape[2])]*(X.shape[1])]*(X.shape[0]), dtype=np.float32)).to(device) #([([[None] * X.shape[2]])* X.shape[1]])* X.shape[0]
    #     intensity4pixel = Variable(torch.zeros((batchsize,X_shape,Y_shape,Z_shape)).to(device), requires_grad=True).clone()#torch.full(X.shape, torch.nan).to(device)
    # elif scan_name.startswith('gt_step'):
    weight4pixel = torch.zeros((batchsize,X_shape,Y_shape,Z_shape)).to(device)#torch.full(X.shape, torch.nan).to(device) #torch.from_numpy(np.asarray([[[None]*(X.shape[2])]*(X.shape[1])]*(X.shape[0]), dtype=np.float32)).to(device) #([([[None] * X.shape[2]])* X.shape[1]])* X.shape[0]
    intensity4pixel = torch.zeros((batchsize,X_shape,Y_shape,Z_shape)).to(device)#torch.full(X.shape, torch.nan).to(device)
    

    # this is very important. In this way, the corridinates of each point is normalised.
    # That is, the normalised coordinates is the same as the index of the points 
    x_norm = ((scatter_pts[:,0]-min_x)/xstep).to(device)
    y_norm = ((scatter_pts[:,1]-min_y)/ystep).to(device)
    z_norm = ((scatter_pts[:,2]-min_z)/zstep ).to(device) 
    

    x_low,x_up,y_low,y_up,z_low,z_up = eight_neighbour_points(x_norm,y_norm,z_norm) 
    neighbour_pts_idx = eright_points_in_1d(x_low,x_up,y_low,y_up,z_low,z_up,X_shape,Y_shape,Z_shape,initial)
    weight4pixel_8neighbour_pts,intensity4pixel_8neighbour_pts = weight_intensity_in_1d(x_low,x_up,y_low,y_up,z_low,z_up,x_norm,y_norm,z_norm,frames_flatten,options = option)
    weight4pixel,intensity4pixel = combine_values(weight4pixel.flatten(),intensity4pixel.flatten(),weight4pixel_8neighbour_pts,intensity4pixel_8neighbour_pts,neighbour_pts_idx)

    weight4pixel = torch.reshape(weight4pixel,(batchsize,X_shape,Y_shape,Z_shape))
    intensity4pixel = torch.reshape(intensity4pixel,(batchsize,X_shape,Y_shape,Z_shape))

    # some pixel has 0 weight or 0 intensity, check
    # weight = 0 means no points contribute to this pixel, and the intensity of this pixel should be 0
    # intensity = 0 means either no points contribute to this pixel or the contributed pixels are all have pixel 0
    nan_index = (weight4pixel == 0).nonzero() # 
    
    # convert 0 to 1, to aviod 0/0
    weight4pixel[nan_index[:,0],nan_index[:,1],nan_index[:,2],nan_index[:,3]] = 1

    volume_intensity = intensity4pixel/weight4pixel
    # volume_intensity[nan_index[:,0],nan_index[:,1],nan_index[:,2],nan_index[:,3]] = 0
    if torch.sum(torch.isnan(volume_intensity))!=0:
        raise("Intensity of pixels has %d nan values" %(volume_intensity == 0).nonzero().shape[0])


    # # save time
    # with open(time_log, 'a') as time_file:
    #     print('Pytorch_tensor GPU: %.3f' % (time_e),file=time_file)
    #     print('\n')


    # # # # compute the normalised axis info for ploting - save mha for ploting
    # don't need to normalise 
    # min_x_norm,max_x_norm = torch.min(x_norm),torch.max(x_norm)
    # min_y_norm,max_y_norm = torch.min(y_norm),torch.max(y_norm)
    # min_z_norm,max_z_norm = torch.min(z_norm),torch.max(z_norm)


    # xx_norm = torch.linspace((min_x_norm), (max_x_norm), int(((max_x_norm)-(min_x_norm)).cpu().numpy()))
    # yy_norm = torch.linspace((min_y_norm), (max_y_norm), int(((max_y_norm)-(min_y_norm)).cpu().numpy()))
    # zz_norm = torch.linspace((min_z_norm), (max_z_norm), int(((max_z_norm)-(min_z_norm)).cpu().numpy()))

    # save2mha(volume_intensity.cpu().numpy(),
    #         sx=np.double(x.cpu().numpy()[1]-x.cpu().numpy()[0]),
    #         sy = np.double(y.cpu().numpy()[1]-y.cpu().numpy()[0]),
    #         sz = np.double(z.cpu().numpy()[1]-z.cpu().numpy()[0]),
    #         save_folder=saved_folder_test+'/'+scan_name+'_pytorch_GPU.mha'
    #         )

    return volume_intensity, [X,Y,Z]


