import os
import torch
from PIL import Image
import numpy as np
from collections import namedtuple
from glob import glob

from dataset.validation.carla_labels import carla_color2oodId

sort_key = lambda x: f"{int(x.split('/')[-3].split('-')[0][3:]):03d}" + f"{int(x.split('/')[-3].split('-')[1]):03d}" + f"{int(os.path.basename(x)[:-4]):05d}"


class CarlaAnomaly(torch.utils.data.Dataset):
    FishyscapesClass = namedtuple('FishyscapesClass', ['name', 'id', 'train_id', 'hasinstances',
                                                       'ignoreineval', 'color'])
    labels = [
        FishyscapesClass('in-distribution', 0, 0, False, False, (144, 238, 144)),
        FishyscapesClass('out-distribution', 1, 1, False, False, (255, 102, 102)),
        FishyscapesClass('unlabeled', 2, 255, False, True, (0, 0, 0)),
    ]

    train_id_in = 0
    train_id_out = 1
    num_eval_classes = 19

    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, root, transform=None, sample=20):
        self.root = root
        self.transform = transform

        self.image_list = glob(os.path.join(root, '**/rgb_v/*.png'), recursive=True)
        self.image_list = sorted(self.image_list, key=sort_key)
        self.mask_list = [i.replace('rgb_v', 'mask_v') for i in self.image_list]
        if sample is not None:
            self.image_list = self.image_list[::sample]
            self.mask_list = self.mask_list[::sample]

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        mask = Image.open(self.mask_list[index]).convert('RGB')
        mask = np.array(mask)
        oodid_mask = np.zeros(mask.shape[:2], dtype='uint8')
        for k, v in carla_color2oodId.items():
            oodid_mask[(mask == k).all(-1)] = v
        
        mask = Image.fromarray(oodid_mask)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

    def __len__(self):
        return len(self.image_list)
    

if __name__ == '__main__':
    dataset = CarlaAnomaly('/data/tb5zhh/workspace/RPL/RPL/data/anomaly_dataset/v5_release/val')
    for image, mask in dataset:
        print(image.size, mask.size)
        from IPython import embed; embed()
        break
