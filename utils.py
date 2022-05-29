import os
import random

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


def rle_decode(mask_rle, shape):
    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    ends = starts + lengths
    h, w = shape
    img = np.zeros((h * w,), dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo: hi] = 1
    return img.reshape(shape)


def set_random_seed(seed=2022):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH).astype('float32')
    # img /= np.max(img)
    return img


def load_msk(msk_path):
    msk = cv2.imread(msk_path).astype('float32')
    return msk


class BuildDataset(Dataset):
    def __init__(self, data_folder, list_file, label=True, transforms=None):
        self.case_list = open(list_file).read().strip().split('\n')
        self.label = label
        self.img_paths = [os.path.join(data_folder, 'images', case+'.png') for case in self.case_list]
        self.msk_paths = [os.path.join(data_folder, 'labels', case+'.png') for case in self.case_list]
        self.transforms = transforms

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype('float32')

        if self.label:
            msk_path = self.msk_paths[index]
            msk = cv2.imread(msk_path).astype('float32')
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1, 0))
    return iou
