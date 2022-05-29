import os
import sys
import time
import argparse

import cv2
import torch
from torch.cuda import amp
from torch import nn
import numpy as np
from tqdm import tqdm
import albumentations
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler


from utils import set_random_seed, BuildDataset, dice_coef, iou_coef


set_random_seed(2022)
output_data_root = r'/kaggle/working'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CFG(object):
    backbone = 'efficientnet-b0'
    train_bs = 64
    valid_bs = train_bs*2
    img_size = [256, 256]
    epochs = 10
    lr = 2e-3
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = int(30000/train_bs*epochs)+50
    T_0 = 25
    wd = 1e-6
    n_accumulate = max(1, 32//train_bs)
    n_fold = 5
    folds = [0]
    num_classes = 3


JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss = smp.losses.DiceLoss(mode='multilabel')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)


data_transforms = {
    "train": albumentations.Compose([
        albumentations.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        albumentations.OneOf([
            albumentations.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            albumentations.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        albumentations.CoarseDropout(max_holes=8, max_height=CFG.img_size[0] // 20, max_width=CFG.img_size[1] // 20,
                                     min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
    ], p=1.0),
    "val": albumentations.Compose([
        albumentations.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
    ], p=1.0)
}


def build_model(model_file_path=None):
    net = smp.Unet(
        encoder_name=CFG.backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=CFG.num_classes,
        activation=None,
    )
    net.to(device)
    if model_file_path is not None:
        net.load_state_dict(torch.load(model_file_path))
    net.eval()
    return net


def criterion(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)


def train_one_epoch(model, optimizer, scheduler, data_loader):
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc='Train ')
    epoch_loss = 0
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        with amp.autocast(enabled=False):
            y_pred = model(images)
            loss = criterion(y_pred, masks)
            loss = loss / CFG.n_accumulate

        scaler.scale(loss).backward()

        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}', lr=f'{current_lr:0.5f}')

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, data_loader):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    val_scores = []
    epoch_loss = 0

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc='Val ')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        y_pred = model(images)
        loss = criterion(y_pred, masks)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])

        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}')
    val_scores = np.mean(val_scores, axis=0)

    return epoch_loss, val_scores


def run_training(model, optimizer, scheduler, num_epochs, train_loader, val_loader, fold_idx=0):
    start = time.time()
    best_dice = -np.inf

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_one_epoch(model, optimizer, scheduler, train_loader)

        val_loss, val_scores = valid_one_epoch(model, val_loader)
        val_dice, val_jaccard = val_scores

        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')

        if val_dice >= best_dice:
            print(f"Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice = val_dice
            best_save_path = f"best_epoch-{fold_idx:02d}.bin"
            torch.save(model.state_dict(), best_save_path)
            print(f"Model Saved{best_save_path}")

        last_save_path = f"last_epoch-{fold_idx:02d}.bin"
        torch.save(model.state_dict(), last_save_path)

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))


def fetch_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, eta_min=CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7,
                                              threshold=0.0001, min_lr=CFG.min_lr, )
    elif CFG.scheduler == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    return None


if __name__ == '__main__':

    for fold_idx in CFG.folds:
        print(f'### Fold: {fold_idx}')
        train_dataset = BuildDataset(data_folder=os.path.join(output_data_root, 'mmseg_train'),
                                     list_file=os.path.join(output_data_root, 'mmseg_train', 'splits', f'fold_{fold_idx}.txt'),
                                     transforms=data_transforms['train'])
        val_dataset = BuildDataset(data_folder=os.path.join(output_data_root, 'mmseg_train'),
                                   list_file=os.path.join(output_data_root, 'mmseg_train', 'splits', f'holdout_{fold_idx}.txt'),
                                   transforms=data_transforms['val'])

        train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs,
                                  num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=CFG.valid_bs,
                                num_workers=4, shuffle=False, pin_memory=True)

        train_model = build_model()
        train_optimizer = optim.Adam(train_model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        train_scheduler = fetch_scheduler(train_optimizer)
        run_training(train_model, train_optimizer, train_scheduler,
                     num_epochs=CFG.epochs, train_loader=train_loader, val_loader=val_loader, fold_idx=fold_idx)


