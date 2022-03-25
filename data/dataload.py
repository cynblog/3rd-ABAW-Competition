import os
import torch
import glob
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image
import yaml

class ImageFolder(data.Dataset):
    def __init__(self, img_paths, au_labels, augmentation):
        self.img_paths = img_paths
        
        self.au_labels = au_labels
        
        self.augmentation = augmentation
        
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        
        au_label = self.au_labels[index]
        for i in range(12):
            if au_label[i] == -1:
                au_label[i] = 0
        au_label = au_label.astype('float')
        
        image = self.augmentation(image)
        
        return {'image': image, 'au': au_label}
    
    def __len__(self):
        print(np.size(self.img_paths, 0))
        return np.size(self.img_paths, 0)

class Data_Loader(object):
    def __init__(self, subjects, label_path, all_frames_path, config, mode='train'):        
        self.subjects = subjects
        self.label_path = label_path
        self.all_frames_path = all_frames_path
        self.config = config
        self.mode = mode
        
        
    def get_loader(self, num_workers = 4, shuffle = True):
        
        all_images = []
        for subject in self.subjects:
            subject_path = os.path.join(self.all_frames_path, subject)
            img_list = glob.glob(os.path.join(subject_path, "*.jpg"))
            img_list = sorted(img_list)
            all_images.extend(img_list)
        print(f'all images length: {len(all_images)}')
        
        all_au_labels = np.zeros((0, 12))
        for subject in self.subjects:
            au_labels = np.load(os.path.join(self.label_path, subject+'.npy'))
            all_au_labels = np.vstack((all_au_labels, au_labels))
        
        print(f'all au labels shape: {all_au_labels.shape}')                         
        
        augmentation = self.transform_select()
        
        dataset = ImageFolder(np.array(all_images), all_au_labels, augmentation)
    
        data_loader = data.DataLoader(dataset = dataset, batch_size = self.config["batch_size"], shuffle = shuffle, num_workers = num_workers, drop_last = True)
        
        return data_loader
               
    
    def transform_select(self):
        if self.mode == 'test':
            transform = transforms.Compose([
                # transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        elif self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)], p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        else:
            print("No transforms selected!!")
        return transform
        

        
        