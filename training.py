#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import glob
from au_class import AU_class

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

all_frames_path = '/amax/cyn/cvpr22_competition/affwild2_au_data/cropped_aligned/'
train_label_path = '/amax/cyn/cvpr22_competition/affwild2_au_data/labels/Train_set_npy/'

all_train_id = os.listdir(train_label_path)

all_train_id = [video[:-4] for video in all_train_id]
all_train_id.sort()


train_not_consistent = []
train_png_list = []
for video in all_train_id:
    label = np.load(os.path.join(train_label_path, video+'.npy'))
    img_list = glob.glob(os.path.join(all_frames_path, video, "*.jpg"))
    img_list_png = glob.glob(os.path.join(all_frames_path, video, "*.png"))
    
    if label.shape[0] != len(img_list):
#         print(f'video: {video}, label length:{label.shape[0]}, img list length: {len(img_list)}')
        train_not_consistent.append(video)
        
    if len(img_list_png) != 0:
        train_png_list.append(video)

print(f'not consistent train video length:{len(train_not_consistent)}')

train_class =AU_class('./config.yaml')

all_train_id = [train_id for train_id in all_train_id if train_id not in train_not_consistent]

train_class.train_classfier(all_train_id, train_label_path, all_frames_path)
