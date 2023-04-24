from PIL import Image
from glob import glob
import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from dataset.training.carla_labels import carla_color2trainId, carla_color2oodId, palette

# root = "/home/tb5zhh/workspace/2023/SML/SML/data/new-carla/v3"


def make_dataset(root, mode):
    # mode: train/val
    ret_list = []
    for i in sorted(glob(osp.join(root, mode, "**/rgb_v/*.png")), key=lambda i: file_id(i)):
        ret_list.append((i, i.replace("rgb_v", "mask_v")))
    return ret_list


def file_id(filepath: str):
    # example input: /home/tb5zhh/workspace/2023/SML/SML/data/new-carla/v3/train/seq00-1/rgb_v/1.png
    # example output:seq00-001-001
    return filepath.split("/")[-3].split(
        "-"
    )[0] + "-" + f'{int(filepath.split("/")[-3].split("-")[1]):03d}' + "-" + f'{int(filepath.split("/")[-1][:-4]):03d}'

class Carla(data.Dataset):

    num_train_ids = 19
    mean = [0.4731, 0.4955, 0.5078]
    std = [0.2753, 0.2715, 0.2758]
    ignore_in_eval_ids = 255

    def __init__(self,
                 root,
                 mode,
                 joint_transform=None,
                 transform=None,):
        self.mode = mode
        self.joint_transform = joint_transform
        self.transform = transform

        self.images, self.targets = zip(*make_dataset(root, mode))
        if len(self.images) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mean_std = (self.mean, self.std)

    def __getitem__(self, index):

        img_path, mask_path = self.images[index], self.targets[index]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        mask = np.array(mask)
        trainid_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

        for k, v in carla_color2trainId.items():
            trainid_mask[(mask == k).all(-1)] = v

        seg_mask = Image.fromarray(trainid_mask.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        img = transforms.Normalize(*self.mean_std)(img)

        return img, seg_mask

    def __len__(self):
        return len(self.images)
