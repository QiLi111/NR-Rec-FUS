

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from utils.loader import SSFrameDataset
from options.train_options import TrainOptions
from utils.utils_grid_data import *
from utils.utils_meta import *
import sys
sys.path.append(os.getcwd()+'/utils')

opt = TrainOptions().parse()
writer = SummaryWriter(os.path.join(opt.SAVE_PATH))
# if not opt.multi_gpu:
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


opt.FILENAME_VAL=opt.FILENAME_VAL+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json'
opt.FILENAME_TEST=opt.FILENAME_TEST+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json'
opt.FILENAME_TRAIN=[opt.FILENAME_TRAIN[i]+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json' for i in range(len(opt.FILENAME_TRAIN))]

dset_val = SSFrameDataset.read_json(Path(os.getcwd()).as_posix()+opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name)
dset_test = SSFrameDataset.read_json(Path(os.getcwd()).as_posix()+opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name)
dset_train_list = [SSFrameDataset.read_json(Path(os.getcwd()).as_posix()+opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name) for i in range(len(opt.FILENAME_TRAIN))]
dset_train = dset_train_list[0]+dset_train_list[1]
dset_val = dset_val+dset_train_list[2]
print('using %s'%opt.h5_file_name)
               

train_rec_reg_model = Train_Rec_Reg_Model(opt = opt,
                        non_improve_maxmum = 1e10, 
                        reg_loss_weight = 1000,
                        val_loss_min = 1e10,
                        val_dist_min = 1e10,
                        val_loss_min_reg = 1e10,
                        dset_train = dset_train,
                        dset_val = dset_val,
                        dset_train_reg = None,
                        dset_val_reg = None,
                        device = device,
                        writer = writer,
                        option = 'common_volume')

# load pre-trained model
# train_rec_reg_model.load_rec_model_initial()
# train_rec_reg_model.load_reg_model_initial()

train_rec_reg_model.multi_model()
train_rec_reg_model.train_rec_model()



