import argparse
import os
import json


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--FILENAME_CALIB', type=str, default="data/calib_matrix.csv",help='dataroot of calibration matrix')
        self.parser.add_argument('--multi_gpu', type=bool,default=False,help='whether use multi gpus')
        self.parser.add_argument('--gpu_ids',type=str,default='1',help='gpu id: e.g., 0,1,2...')
        self.parser.add_argument('--RESAMPLE_FACTOR', type=int,default=4,help='resize of the original image')
        self.parser.add_argument('--config', type=str,default='config/config_ete.json',help='config file')
        self.parser.add_argument('--SAVE_PATH', type=str, default='results',help='foldername of saving path')
        self.parser.add_argument('--DATA_PATH', type=str, default='/data', help='foldername of saving path')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        self.opt.h5_file_name=None

        args = vars(self.opt)
        # update from config json
        with open(self.opt.config) as f:
            json_dict = json.load(f)

        args.update(json_dict)

        print('----------Option----------')
        for k,v in sorted(args.items()):
            print('%s, %s' %(str(k),str(v)))
            print('\n')
        print('----------Option----------')

        # create saved result path
        if self.opt.inter=='nointer' and self.opt.meta=='nonmeta':
            saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES)\
                +'__Loss_'+str(self.opt.Loss_type)+'__bs_'+str(self.opt.MINIBATCH_SIZE_rec)\
                +'__inc_reg_'+str(self.opt.in_ch_reg)\
                +'__'+str(self.opt.ddf_dirc)+'__ete'#+'__comp_crop_311'#+'__baseline_311__M'#+'__baseline_corp_311'#+'__baseline_311__M'+'__corp_221_rerun'#+'_HalfConverge_cropped_bs1_debug'#+'_HalfBestModel_bs1_uncrop'#+'_BS1'#+'_HalfBestModel_bs4'#+'_batchsize1'
                #+'_one_scan'#+'__weightloss1000'#+'__iteratively'#+'/'+'isbi'.  
        elif self.opt.inter=='iteratively' and self.opt.meta=='meta':
            
            saved_results = 'seq_len' + str(self.opt.NUM_SAMPLES) \
                +'__Loss_'+str(self.opt.Loss_type)\
                +'__bs_'+str(self.opt.MINIBATCH_SIZE_rec)\
                +'__inc_reg_'+str(self.opt.in_ch_reg)+'__'+str(self.opt.ddf_dirc)+'__meta'



        self.opt.SAVE_PATH = os.path.join(os.getcwd(),'models_all/'+saved_results)
        
        if self.opt.train_set == 'loop':
            self.opt.h5_file_name = 'scans_res4.h5'
        elif self.opt.train_set == 'forth':
            self.opt.h5_file_name = 'scans_res4_forth.h5'
        elif self.opt.train_set == 'back':
            self.opt.h5_file_name = 'scans_res4_back.h5'
        elif self.opt.train_set == 'forth_back':
            self.opt.h5_file_name = 'scans_res4_forth_back.h5'
        

        if not os.path.exists(self.opt.SAVE_PATH):
            os.makedirs(self.opt.SAVE_PATH)
        if not os.path.exists(os.path.join(self.opt.SAVE_PATH,'saved_model')):
            os.makedirs(os.path.join(self.opt.SAVE_PATH,'saved_model'))
    

        file_name = os.path.join(self.opt.SAVE_PATH,'config.txt')
        with open(file_name,'a') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k,v in sorted(args.items()):
                opt_file.write('%s,%s'%(str(k),str(v)))
                opt_file.write('\n')
            opt_file.write('------------ Options -------------\n')
        return self.opt

