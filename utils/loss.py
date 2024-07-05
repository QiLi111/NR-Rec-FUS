
import torch
from utils.transform import TransformAccumulation
from utils.transform import LabelTransform
import numpy as np
from itertools import combinations
# TBD


class PointDistance:
    def __init__(self,paired=True):
        self.paired = paired
    
    def __call__(self,preds,labels):
        if self.paired:
            return ((preds-labels)**2).sum(dim=2).sqrt().mean(dim=(0,2))
        else:
            return ((preds-labels)**2).sum(dim=2).sqrt().mean()


class MTL_loss:
    def __init__(self, paired=True):
        self.paired = paired

    def __call__(self, preds, labels):
        if self.paired:
            return ((preds - labels) ** 2).mean(dim=(0,2, 3))
        else:
            return ((preds - labels) ** 2).mean()


class PointDistance_2:
    def __init__(self, paired=True):
        self.paired = paired

    def __call__(self, preds, labels):
        return ((preds - labels) ** 2).sum(dim=0).sqrt().mean()

class PointDistance_1:
    def __init__(self, paired=True):
        self.paired = paired

    def __call__(self, preds, labels):
        return ((preds - labels) ** 2).sum(dim=1).sqrt().mean()

class PointDistance_3:
    def __init__(self, paired=True):
        self.paired = paired

    def __call__(self, preds, labels):
        if len(preds):
            return ((preds - labels) ** 2).sum(dim=1).sqrt().mean(axis=1)
        else:
            return []

