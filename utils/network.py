
import torch
import os
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.utils import _pair
from utils.transform import  PredictionTransform
# import resnext

def build_model(opt,in_frames, pred_dim, label_dim, image_points, tform_calib, tform_calib_R_T):
    """
    :param model_function: classification model
    """
    if opt.model_name == "efficientnet_b1":
        model = efficientnet_b1(weights=None)
        model.features[0][0] = torch.nn.Conv2d(
            in_channels  = in_frames, 
            out_channels = model.features[0][0].out_channels, 
            kernel_size  = model.features[0][0].kernel_size, 
            stride       = model.features[0][0].stride, 
            padding      = model.features[0][0].padding, 
            bias         = model.features[0][0].bias
        )
        model.classifier[1] = torch.nn.Linear(
            in_features   = model.classifier[1].in_features,
            out_features  = pred_dim
        )
        # print(model.classifier[1].in_features)
        # print(model)



    elif opt.model_name == "efficientnet_b6":
        model = efficientnet_b6(weights=None)
        model.features[0][0] = torch.nn.Conv2d(
            in_channels=in_frames,
            out_channels=model.features[0][0].out_channels,
            kernel_size=model.features[0][0].kernel_size,
            stride=model.features[0][0].stride,
            padding=model.features[0][0].padding,
            bias=model.features[0][0].bias
        )
        model.classifier[1] = torch.nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=pred_dim
        )
    elif opt.model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.features[0][0] = torch.nn.Conv2d(
            in_channels=in_frames,
            out_channels=model.features[0][0].out_channels,
            kernel_size=model.features[0][0].kernel_size,
            stride=model.features[0][0].stride,
            padding=model.features[0][0].padding,
            bias=model.features[0][0].bias
        )
        model.classifier[1] = torch.nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=pred_dim
        )
    elif opt.model_name[:6] == "resnet":
        model = resnet101()
        model.conv1 = torch.nn.Conv2d(
            in_channels  = in_frames, 
            out_channels = model.conv1.out_channels,
            kernel_size  = model.conv1.kernel_size,
            stride       = model.conv1.stride,
            padding      = model.conv1.padding,
            bias         = model.conv1.bias
        )
        model.fc = torch.nn.Linear(
            in_features   = model.fc.in_features,
            out_features  = pred_dim
        )
 
    else:
        raise("Unknown model.")
    
    return model





def save_best_network(SAVE_PATH, model, epoch_label, gpu_ids):
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'saved_model', 'best_validation_model'))

    file_name = os.path.join(SAVE_PATH, 'opt.txt')
    with open(file_name, 'a') as opt_file:
        opt_file.write('------------ best validation result - epoch: -------------\n')
        opt_file.write(str(epoch_label))
        opt_file.write('\n')


