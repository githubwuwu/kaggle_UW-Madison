import os
import glob
import random

import numpy as np
import pandas as pd

import cv2
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold

from utils import rle_decode


def parse_all_data(raw_data_root):
    csv_path = os.path.join(raw_data_root, 'train.csv')
    df_train = pd.read_csv(csv_path)
    df_train = df_train.sort_values(["id", "class"]).reset_index(drop=True)
    df_train["patient"] = df_train.id.apply(lambda x: x.split("_")[0])
    df_train["days"] = df_train.id.apply(lambda x: "_".join(x.split("_")[:2]))

    all_image_files = sorted(glob.glob(os.path.join(raw_data_root, "train/*/*/scans/*.png")),
                             key=lambda x: x.split(os.sep)[-3] + "_" + x.split(os.sep)[-1])
    size_x = [int(os.path.basename(_)[:-4].split("_")[-4]) for _ in all_image_files]
    size_y = [int(os.path.basename(_)[:-4].split("_")[-3]) for _ in all_image_files]
    spacing_x = [float(os.path.basename(_)[:-4].split("_")[-2]) for _ in all_image_files]
    spacing_y = [float(os.path.basename(_)[:-4].split("_")[-1]) for _ in all_image_files]
    df_train["image_files"] = np.repeat(all_image_files, 3)
    df_train["spacing_x"] = np.repeat(spacing_x, 3)
    df_train["spacing_y"] = np.repeat(spacing_y, 3)
    df_train["size_x"] = np.repeat(size_x, 3)
    df_train["size_y"] = np.repeat(size_y, 3)
    df_train["slice"] = np.repeat([int(os.path.basename(_)[:-4].split("_")[-5]) for _ in all_image_files], 3)
    print(df_train.head(), df_train.isna().sum())
    return df_train


def resave_25d_data(df_train, data_save_folder):
    if not os.path.exists(os.path.join(data_save_folder, 'images')):
        os.makedirs(os.path.join(data_save_folder, 'images'))
    if not os.path.exists(os.path.join(data_save_folder, 'labels')):
        os.makedirs(os.path.join(data_save_folder, 'labels'))

    for day, group in tqdm(df_train.groupby("days")):
        imgs = []
        msks = []
        for file_name in group.image_files.unique():
            img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
            segms = group.loc[group.image_files == file_name]
            masks = {}
            for segm, label in zip(segms.segmentation, segms["class"]):
                if not pd.isna(segm):
                    mask = rle_decode(segm, img.shape[:2])
                    masks[label] = mask
                else:
                    masks[label] = np.zeros(img.shape[:2], dtype=np.uint8)
            masks = np.stack([masks[k] for k in sorted(masks)], -1)
            imgs.append(img)
            msks.append(masks)

        imgs = np.stack(imgs, 0)
        msks = np.stack(msks, 0)
        for i in range(msks.shape[0]):
            img = imgs[[max(0, i - 2), i, min(imgs.shape[0] - 1, i + 2)]].transpose(1, 2, 0)  # 2.5d data
            msk = msks[i]
            new_file_name = f"{day}_{i}.png"
            cv2.imwrite(os.path.join(data_save_folder, 'images', new_file_name), img)
            cv2.imwrite(os.path.join(data_save_folder, 'labels', new_file_name), msk)


def split_train_val(data_save_folder):
    if not os.path.exists(os.path.join(data_save_folder, 'splits')):
        os.makedirs(os.path.join(data_save_folder, 'splits'))
    all_image_files = glob.glob(os.path.join(data_save_folder, 'images/*.png'))
    patients = [os.path.basename(_).split("_")[0] for _ in all_image_files]
    split = list(GroupKFold(5).split(patients, groups=patients))

    for fold, (train_idx, valid_idx) in enumerate(split):
        with open(os.path.join(data_save_folder, 'labels', f"fold_{fold}.txt"), "w") as f:
            for idx in train_idx:
                f.write(os.path.basename(all_image_files[idx])[:-4] + "\n")
        with open(os.path.join(data_save_folder, 'labels', f"holdout_{fold}.txt"), "w") as f:
            for idx in valid_idx:
                f.write(os.path.basename(all_image_files[idx])[:-4] + "\n")


def check_data_mask(data_save_folder):
    from matplotlib import pyplot as plt
    all_image_files = glob.glob(os.path.join(data_save_folder, 'images/*.png'))
    select_img_case = random.sample(all_image_files, 10000)
    # print(select_img_case)
    select_label_case = [str(file_path).replace('images', 'labels') for file_path in select_img_case]

    for img_path, label_path in tqdm(zip(select_img_case, select_label_case)):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        label = cv2.imread(label_path)
        if label.max() == 0:
            continue
        plt.subplot(2, 1, 1)
        plt.imshow(img, cmap='bone')
        plt.subplot(2, 1, 2)
        plt.imshow(label * 255, cmap='bone')
        plt.show()


if __name__ == '__main__':
    data_root = r'../input/uw-madison-gi-tract-image-segmentation'
    df = parse_all_data(os.path.join(data_root, 'uw-madison-gi-tract-image-segmentation'))
    # delete some case
    fault1 = 'case7_day0'
    fault2 = 'case81_day30'
    df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(fault2)].reset_index(drop=True)

    resave_25d_data(df, os.path.join(data_root, 'mmseg_train'))
    split_train_val(os.path.join(data_root, 'mmseg_train'))
    check_data_mask(os.path.join(data_root, 'mmseg_train'))

