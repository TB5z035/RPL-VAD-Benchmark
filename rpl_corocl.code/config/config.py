import os
import numpy
from easydict import EasyDict

C = EasyDict()
config = C
cfg = C

C.seed = 666

"""Root Directory Config"""
C.repo_name = 'ood_seg'
C.root_dir = os.path.realpath("")

"""Data Dir and Weight Dir"""
C.carla_root_path = '/home/tb5zhh/workspace/2023/SML/SML/data/new-carla/v3'
C.carla_anomaly_root_path = '/home/tb5zhh/workspace/2023/SML/SML/data/anomaly_dataset/v5_release/val'
C.city_root_path = '/data/tb5zhh/workspace/RPL/RPL/data/cityscapes'
C.coco_root_path = '/data/tb5zhh/workspace/RPL/RPL/data/COCO'
C.fishy_root_path = '/data/tb5zhh/workspace/RPL/RPL/data/fishyscapes'
C.segment_me_root_path = 'path/to/your/smiyc'
C.road_anomaly_root_path = 'path/to/your/roadanomaly'

C.rpl_corocl_weight_path = os.path.join(C.root_dir, 'ckpts', 'exp', 'rev3.pth')
C.pretrained_weight_path = os.path.join(C.root_dir, 'ckpts', 'pretrained_ckpts', 'best_checkpoint_ep8.pth')

"""Network Config"""
C.fix_bias = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Image Config"""
C.num_classes = 19
C.outlier_exposure_idx = 254  # NOTE: it starts from 0

C.image_mean = numpy.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = numpy.array([0.229, 0.224, 0.225])

C.city_image_height = 700
C.city_image_width = 700

C.ood_image_height = C.city_image_height
C.ood_image_width = C.city_image_width

# C.city_train_scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2.0]
C.ood_train_scale_array = [.25, .5, .5, .75, .1, .125]

C.num_train_imgs = 76920
C.num_eval_imgs = 100

"""Train Config"""
C.dataset = "carla"
C.lr = 7.5e-5
C.batch_size = 8
C.energy_weight = .05

C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 1e-4

C.nepochs = 40 
C.niters_per_epoch = C.num_train_imgs // C.batch_size

C.num_workers = 8
C.void_number = 5
C.warm_up_epoch = 0

"""Eval Config"""
C.eval_iter = 500
C.measure_way = "energy"
C.eval_stride_rate = 1 / 3
C.eval_scale_array = [1., ]
C.eval_flip = True
C.eval_crop_size = 1024

"""Display Config"""
C.record_info_iter = 20
C.display_iter = 50

"""Wandb Config"""
# Specify you wandb environment KEY; and paste here
C.wandb_key = ""

# Your project [work_space] name
C.proj_name = "OoD_Segmentation"

C.experiment_name = "rpl.code+corocl-carla"

# half pretrained_ckpts-loader upload images; loss upload every iteration
C.upload_image_step = [0, int((C.num_train_imgs / C.batch_size) / 2)]

# False for debug; True for visualize
C.wandb_online = True

"""Save Config"""
C.saved_dir = os.path.join("ckpts/exp", C.experiment_name)

if not os.path.exists(C.saved_dir):
    os.mkdir(C.saved_dir)

