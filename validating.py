#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import glob
from au_class import AU_class

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

all_frames_path = '/amax/cyn/cvpr22_competition/affwild2_au_data/cropped_aligned/'
val_label_path = '/amax/cyn/cvpr22_competition/affwild2_au_data/labels/Validation_set_npy/'

test_class =AU_class('./config.yaml')

all_val_id = os.listdir(val_label_path)
all_val_id = [video[:-4] for video in all_val_id]
all_val_id.sort()

val_not_consistent = []
val_png_list = []
for video in all_val_id:
    label = np.load(os.path.join(val_label_path, video+'.npy'))
    img_list = glob.glob(os.path.join(all_frames_path, video, "*.jpg"))
    img_list_png = glob.glob(os.path.join(all_frames_path, video, "*.png"))
    
    if label.shape[0] != len(img_list):
        print(f'video: {video}, label length:{label.shape[0]}, img list length: {len(img_list)}')
        val_not_consistent.append(video)
        
    if len(img_list_png) != 0:
        val_png_list.append(video)


all_val_id = [val_id for val_id in all_val_id if val_id not in val_not_consistent]

for epoch in range(10):
    tmp = test_class.test(all_val_id, val_label_path, all_frames_path, epoch)
    print(f'epoch: {epoch}, result: {tmp}')