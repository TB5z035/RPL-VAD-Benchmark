#!/usr/bin/python
#
# Cityscapes labels
#

# yapf: disable
from __future__ import absolute_import, division, print_function

from collections import namedtuple

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'carlaId'     ,

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'inCarla'     ,
    'inCityscapes',

    'color'       , # The color of this label
    'carlaColor'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId    carlaId   category            catId     hasInstances   ignoreInEval  inCarla  inCityscapes  color          carlaColor
    Label(  'unlabeled'            ,  0 ,      255 ,       28 , 'void'            , 0       , False        , True         , True  , True        , (  0,  0,  0), (  0,  0,  0)  ),
    Label(  'ego vehicle'          ,  1 ,      255 ,      255 , 'void'            , 0       , False        , True         , False , True        , (  0,  0,  0), (  0,  0,  0)  ),
    Label(  'rectification border' ,  2 ,      255 ,      255 , 'void'            , 0       , False        , True         , False , True        , (  0,  0,  0), (  0,  0,  0)  ),
    Label(  'out of roi'           ,  3 ,      255 ,      255 , 'void'            , 0       , False        , True         , False , True        , (  0,  0,  0), (  0,  0,  0)  ),
    Label(  'static'               ,  4 ,      255 ,       19 , 'void'            , 0       , False        , True         , True  , True        , (  0,  0,  0), (110,190,160)  ),
    Label(  'dynamic'              ,  5 ,      255 ,       20 , 'void'            , 0       , False        , True         , True  , True        , (111, 74,  0), (170,120, 50)  ),
    Label(  'ground'               ,  6 ,      255 ,       24 , 'void'            , 0       , False        , True         , True  , True        , ( 81,  0, 81), ( 81,  0, 81)  ),
    Label(  'road'                 ,  7 ,        0 ,        0 , 'flat'            , 1       , False        , False        , True  , True        , (128, 64,128), (128, 64,128)  ),
    Label(  'sidewalk'             ,  8 ,        1 ,        1 , 'flat'            , 1       , False        , False        , True  , True        , (244, 35,232), (244, 35,232)  ),
    Label(  'parking'              ,  9 ,      255 ,      255 , 'flat'            , 1       , False        , True         , False , True        , (250,170,160), (250,170,160)  ),
    Label(  'rail track'           , 10 ,      255 ,       26 , 'flat'            , 1       , False        , True         , True  , True        , (230,150,140), (230,150,140)  ),
    Label(  'building'             , 11 ,        2 ,        2 , 'construction'    , 2       , False        , False        , True  , True        , ( 70, 70, 70), ( 70, 70, 70)  ),
    Label(  'wall'                 , 12 ,        3 ,        3 , 'construction'    , 2       , False        , False        , True  , True        , (102,102,156), (102,102,156)  ),
    Label(  'fence'                , 13 ,        4 ,        4 , 'construction'    , 2       , False        , False        , True  , True        , (190,153,153), (190,153,153)  ),
    Label(  'guard rail'           , 14 ,      255 ,       27 , 'construction'    , 2       , False        , True         , True  , True        , (180,165,180), (180,165,180)  ),
    Label(  'bridge'               , 15 ,      255 ,       25 , 'construction'    , 2       , False        , True         , True  , True        , (150,100,100), (150,100,100)  ),
    Label(  'tunnel'               , 16 ,      255 ,      255 , 'construction'    , 2       , False        , True         , False , True        , (150,120, 90), (150,120, 90)  ),
    Label(  'pole'                 , 17 ,        5 ,        5 , 'object'          , 3       , False        , False        , True  , True        , (153,153,153), (153,153,153)  ),
    Label(  'polegroup'            , 18 ,      255 ,      255 , 'object'          , 3       , False        , True         , False , True        , (153,153,153), (153,153,153)  ),
    Label(  'traffic light'        , 19 ,        6 ,        6 , 'object'          , 3       , False        , False        , True  , True        , (250,170, 30), (250,170, 30)  ),
    Label(  'traffic sign'         , 20 ,        7 ,        7 , 'object'          , 3       , False        , False        , True  , True        , (220,220,  0), (220,220,  0)  ),
    Label(  'vegetation'           , 21 ,        8 ,        8 , 'nature'          , 4       , False        , False        , True  , True        , (107,142, 35), (107,142, 35)  ),
    Label(  'terrain'              , 22 ,        9 ,        9 , 'nature'          , 4       , False        , False        , True  , True        , (152,251,152), (145,170,100)  ),
    Label(  'sky'                  , 23 ,       10 ,       10 , 'sky'             , 5       , False        , False        , True  , True        , ( 70,130,180), ( 70,130,180)  ),
    Label(  'person'               , 24 ,       11 ,       11 , 'human'           , 6       , True         , False        , True  , True        , (220, 20, 60), (220, 20, 60)  ),
    Label(  'rider'                , 25 ,       12 ,       12 , 'human'           , 6       , True         , False        , True  , True        , (255,  0,  0), (255,  0,  0)  ),
    Label(  'car'                  , 26 ,       13 ,       13 , 'vehicle'         , 7       , True         , False        , True  , True        , (  0,  0,142), (  0,  0,142)  ),
    Label(  'truck'                , 27 ,       14 ,       14 , 'vehicle'         , 7       , True         , False        , True  , True        , (  0,  0, 70), (  0,  0, 70)  ),
    Label(  'bus'                  , 28 ,       15 ,       15 , 'vehicle'         , 7       , True         , False        , True  , True        , (  0, 60,100), (  0, 60,100)  ),
    Label(  'caravan'              , 29 ,      255 ,      255 , 'vehicle'         , 7       , True         , True         , False , True        , (  0,  0, 90), (  0,  0, 90)  ),
    Label(  'trailer'              , 30 ,      255 ,      255 , 'vehicle'         , 7       , True         , True         , False , True        , (  0,  0,110), (  0,  0,110)  ),
    Label(  'train'                , 31 ,       16 ,       16 , 'vehicle'         , 7       , True         , False        , True  , True        , (  0, 80,100), (  0, 80,100)  ),
    Label(  'motorcycle'           , 32 ,       17 ,       17 , 'vehicle'         , 7       , True         , False        , True  , True        , (  0,  0,230), (  0,  0,230)  ),
    Label(  'bicycle'              , 33 ,       18 ,       18 , 'vehicle'         , 7       , True         , False        , True  , True        , (119, 11, 32), (119, 11, 32)  ),
    Label(  'license plate'        , 34 ,      255 ,      255 , 'vehicle'         , 7       , False        , True         , False , True        , (  0,  0,142), (  0,  0,142)  ),
    # Extra categories in Carla
    Label(  'other'                , 37 ,      255 ,       21 , 'carla'           , 8       , False        , True         , True  , False       , ( 55, 90, 80), ( 55, 90, 80)  ),
    Label(  'water'                , 38 ,      255 ,       22 , 'carla'           , 8       , False        , True         , True  , False       , ( 45, 60,150), ( 45, 60,150)  ),
    Label(  'road line'            , 39 ,        0 ,       23 , 'carla'           , 8       , False        , True         , True  , False       , (157,234, 50), (157,234, 50)  ),
]

carla_color2id = {i.carlaColor: i.carlaId for i in labels if i.inCarla}
carla_color2trainId = {i.carlaColor: i.trainId for i in labels}
carla_color2trainId[(  0,  0,142)] = 13
carla_color2trainId[(153,153,153)] = 5
carla_color2trainId[( 55, 90, 80)] = 9

carla_color2oodId = {i.carlaColor: 0 if i.trainId != 255 else 255 for i in labels}
carla_color2oodId[(  0,  0,142)] = 0
carla_color2oodId[(153,153,153)] = 0
carla_color2oodId[( 55, 90, 80)] = 0
carla_color2oodId[(235, 38, 57)] = 1


import numpy as np
palette = {v: k for k, v in carla_color2trainId.items()}
palette[0] = (128, 64, 128)
palette[255] = (0, 0, 0)
