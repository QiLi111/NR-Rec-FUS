import torch,os,csv
from scipy import stats
from utils.visualizer import Visualizer_plot_volume

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from matplotlib import cm
from utils.loader import SSFrameDataset
from options.train_options import TrainOptions
from utils.utils import str2list

from utils.utils_grid_data import *
from utils.utils_meta import *
sys.path.append(os.getcwd()+'/utils')


opt = TrainOptions().parse()
opt_test = opt

writer = SummaryWriter(os.path.join(opt.SAVE_PATH))
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_pairs = data_pairs_adjacent(opt.NUM_SAMPLES)
data_pairs=torch.tensor(data_pairs)

opt.FILENAME_VAL=opt.FILENAME_VAL+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json'
opt.FILENAME_TEST=opt.FILENAME_TEST+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json'
opt.FILENAME_TRAIN=[opt.FILENAME_TRAIN[i]+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json' for i in range(len(opt.FILENAME_TRAIN))]

dset_val = SSFrameDataset.read_json(Path(os.getcwd()).as_posix()+opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name,num_samples = -1)
dset_test = SSFrameDataset.read_json(Path(os.getcwd()).as_posix()+opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name,num_samples = -1)
dset_train_list = [SSFrameDataset.read_json(Path(os.getcwd()).as_posix()+opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name,num_samples = -1) for i in range(len(opt.FILENAME_TRAIN))]

dset_train = dset_train_list[0]+dset_train_list[1]+dset_train_list[2]
# dset_val = dset_val+dset_train_list[2]
print('using %s'%opt.h5_file_name)




viridis = cm.get_cmap('viridis', 10)

options = ['generate_reg_volume_data','reconstruction_vlume']

test_folders = 'models_all' # creat a new "models_all" folder, and move the generated folder after python trainXXX.py, into "models_all" folder
csv_name = 'metrics.csv'
fd_name_save = 'results'

folders = [f for f in os.listdir(os.getcwd()+'/'+test_folders) if f.startswith('seq_len')  and not os.path.isfile(os.path.join(os.getcwd()+'/'+test_folders, f))]
folders = sorted(folders)


# csv file
if not os.path.exists(os.getcwd()+'/'+test_folders+'/'+fd_name_save):
    os.makedirs(os.getcwd()+'/'+test_folders+'/'+fd_name_save)

with open(os.getcwd()+'/'+test_folders+'/'+fd_name_save+'/'+csv_name, 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    # writer.writerow(['the volume of each sequence'])
    writer.writerow(['file_name','model_name','points dist on all points using global T (mean.std) & ','points dist on all points using global T+R (prdiction + DDF)(mean.std) & '\
                     'points dist on 4 points using global T (mean.std) & ','points dist on 4 points using global T+R (prdiction + DDF)(mean.std) & '\
                      'points dist on all points using local T (mean.std) & ',\
                        'points dist on 4 points using local T (mean.std) & '])

for sub_fd in folders:
    # get paramteters from folder name
    print(sub_fd)
    sub_fd_split = sub_fd.split('__')
    fn = sub_fd.split('__')[0]
    opt.NUM_SAMPLES = int(fn[len(fn.rstrip('0123456789')):])
    
    
    ind_inc = None
    for i in range(len(sub_fd_split)):
        if 'inc_reg' in sub_fd_split[i] or 'in_ch_reg' in sub_fd_split[i]:
            ind_inc = i
    try:
        opt.in_ch_reg = int(sub_fd_split[ind_inc].split('_')[-1])
        
    except:
            
        opt.in_ch_reg = 2
    print('opt.in_ch_reg = %d'%opt.in_ch_reg)

    opt.ddf_dirc = None
    if 'Move' in sub_fd:
        opt.ddf_dirc = 'Move'
    
    print('opt.ddf_dirc = %s'%opt.ddf_dirc)


    opt.saved_results = os.getcwd()+'/'+test_folders+'/'+sub_fd
    opt.SAVE_PATH = opt.saved_results

    models_all = [f for f in os.listdir(opt.SAVE_PATH+'/saved_model') if f.startswith('best_val_dist') and os.path.isfile(os.path.join(opt.SAVE_PATH+'/saved_model', f))]
    
    if len(models_all)==4:
        # non-meta training models
        models_name = [['best_val_dist_R_T','best_val_dist_R_R'],['best_val_dist_T_T','best_val_dist_T_R']]
    
    elif len(models_all)==8:
        # meta training models
        models_name = [['best_val_dist_R_R_T','best_val_dist_R_R_R'],\
                       ['best_val_dist_R_T_T','best_val_dist_R_T_R'],\
                        ['best_val_dist_T_R_T','best_val_dist_T_R_R'],\
                       ['best_val_dist_T_T_T','best_val_dist_T_T_R']]                 
    
    elif len(models_all)==0:
        # previous trained model
        models_pre = [f for f in os.listdir(opt.SAVE_PATH+'/saved_model') if f.startswith('best_validation') and os.path.isfile(os.path.join(opt.SAVE_PATH+'/saved_model', f))]
        if len(models_pre) ==4:
            models_name = [['best_validation_loss_model','best_validation_loss_model_reg'],\
                           ['best_validation_dist_model','best_validation_dist_model_reg']]
        elif len(models_pre) ==2:
            models_name = [['best_validation_loss_model'],['best_validation_dist_model']]

    else:
        raise('Not implenment')




    for i_m in range(len(models_name)):
        model_name = models_name[i_m]
        saved_folder_test = os.path.join(os.getcwd()+'/'+test_folders+'/'+fd_name_save+'/'+sub_fd, str(model_name)+'__TestSet')
        if not os.path.exists(saved_folder_test):
            os.makedirs(saved_folder_test)
    


        # visualizer_scan_train = Visualizer_plot_volume(opt,device, dset_train,model_name,data_pairs,options[1])
        # visualizer_scan_val = Visualizer_plot_volume(opt,device, dset_val,model_name,data_pairs,options[1])
        visualizer_scan_test = Visualizer_plot_volume(opt,device, dset_test,model_name,data_pairs,options[1])

        # test - test set
        for scan_index in range(len(dset_test)):
            visualizer_scan_test.generate_volume_data(
                                                        scan_index,
                                                        saved_folder_test,
                                                        based_volume = 'common_volume',# use common volume to reconstruct volume both for registartion use and for final vlulisation use
                                                        
                                                        )
        # save value for future use
        metric1 = np.array(visualizer_scan_test.T_Global_AllPts_Dist)[None,...]
        metric2 = np.array(visualizer_scan_test.T_R_Warp_Global_AllPts_Dist)[None,...]
        metric3 = np.array(visualizer_scan_test.T_Global_FourPts_Dist)[None,...]
        metric4 = np.array(visualizer_scan_test.T_R_Warp_Global_FourPts_Dist)[None,...]
        metric5 = np.array(visualizer_scan_test.T_Local_AllPts_Dist)[None,...]
        metric6 = np.array(visualizer_scan_test.T_Local_FourPts_Dist)[None,...]
        
        metrics = np.concatenate((metric1, metric2, metric3, metric4,metric5, metric6), axis=0)

        with open(os.getcwd()+'/'+test_folders+'/'+fd_name_save+'/'+sub_fd + '__' + str(model_name)+'.npy', 'wb') as f:
            np.save(f, metrics)
        
        
        with open(os.getcwd()+'/'+test_folders+'/'+fd_name_save+'/'+csv_name, 'a', encoding='UTF8') as f:
            f.write(sub_fd)
            f.write(' &  ')
            f.write(str(model_name))
            f.write(' &  ')
            f.write('$%.2f\pm%.2f$ &  ' % (np.array(visualizer_scan_test.T_Global_AllPts_Dist).mean(), np.std(np.array(visualizer_scan_test.T_Global_AllPts_Dist))))
            f.write('$%.2f\pm%.2f$ &  ' % (np.array(visualizer_scan_test.T_R_Warp_Global_AllPts_Dist).mean(), np.std(np.array(visualizer_scan_test.T_R_Warp_Global_AllPts_Dist))))
            
            f.write('$%.2f\pm%.2f$ &  ' % (np.array(visualizer_scan_test.T_Global_FourPts_Dist).mean(), np.std(np.array(visualizer_scan_test.T_Global_FourPts_Dist))))
            f.write('$%.2f\pm%.2f$ &  ' % (np.array(visualizer_scan_test.T_R_Warp_Global_FourPts_Dist).mean(), np.std(np.array(visualizer_scan_test.T_R_Warp_Global_FourPts_Dist))))

            f.write('$%.2f\pm%.2f$ &  ' % (np.array(visualizer_scan_test.T_Local_AllPts_Dist).mean(), np.std(np.array(visualizer_scan_test.T_Local_AllPts_Dist))))
            f.write('$%.2f\pm%.2f$ &  ' % (np.array(visualizer_scan_test.T_Local_FourPts_Dist).mean(), np.std(np.array(visualizer_scan_test.T_Local_FourPts_Dist))))

            f.write('\n')
        


