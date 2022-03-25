import torch
from models.Backbone import Backbone, AU_predict
import torch.nn.functional as F
import os
import shutil
import sys
import numpy as np
from data.dataload import Data_Loader
import torch.nn as nn
import yaml
from sklearn import metrics
    
torch.manual_seed(0)

class AU_class(object):
    def __init__(self, config):
        self.config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        self.device = self._get_device()   ##
        print(f"batch size: {self.config['batch_size']}")
    
    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running on:', device)
        return device
            
    
    def train_classfier(self, all_train_id, train_label_path, all_frames_path):
        
        train_loader = Data_Loader(all_train_id, train_label_path, all_frames_path, self.config).get_loader()  ## right
            
        model = Backbone(self.config['model']['base_model']).to(self.device) ## right 
        model_dict = model.state_dict()
        
        emotion_model = torch.load('./models/best_model1.pkl')
        state_dict = {k:v for k,v in emotion_model.items() if k in model_dict.keys()}
        
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        
        predictor = AU_predict(2048).to(self.device)  ## right        
        
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr':1e-4},
                                      {'params': predictor.parameters(), 'lr':5e-4}], weight_decay=1e-8) ##

        bce_loss = nn.BCEWithLogitsLoss()
        
        n_iter = 0
            
        for epoch in range(self.config["epochs"]):   ##
            for index, data in enumerate(train_loader):
                optimizer.zero_grad()
                
                image = data['image'].to(self.device)
                au = data['au'].to(self.device)
                
                global_features = model(image)
                
                predict_au = predictor(global_features)

                loss = bce_loss(predict_au, au)

                loss.backward()
                optimizer.step()
                
                n_iter += 1
                if (index + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss : {:.4f}'.format(epoch + 1, self.config['epochs'], index + 1, len(train_loader), loss.item()))
                    
            torch.save(predictor.state_dict(), './models/trained/cls_saved_epoch%d.pt' % (epoch))
            torch.save(model.state_dict(), './models/trained/model_saved_epoch%d.pt' % (epoch))
            torch.cuda.empty_cache()

            
    def test(self, all_val_id, val_label_path, all_frames_path, epoch):
        
        val_loader = Data_Loader(all_val_id, val_label_path, all_frames_path, self.config, 'test').get_loader()  ## right

        model = Backbone(self.config['model']['base_model']).to(self.device) ## right   
        model_path = f"./models/trained/model_saved_epoch{epoch}.pt"
        model.load_state_dict(torch.load(model_path))

        
        predictor = AU_predict(2048).to(self.device)  ## right        
        predictor_path = f"./models/trained/cls_saved_epoch{epoch}.pt"
        predictor.load_state_dict(torch.load(predictor_path))        
           
        
        model.eval()
        predictor.eval()
        
        gt_occ, pred_occ = np.zeros(shape=(0, 12)), np.zeros(shape=(0, 12))
        
        with torch.no_grad():
            for data in val_loader:
                image = data['image'].to(self.device)
                au = data['au'].to(self.device)
                
                global_features = model(image)
                
                predict_au = predictor(global_features)
                
                predict_au = torch.sigmoid(predict_au).cpu().numpy()
                pred_occ = np.vstack((pred_occ, predict_au))
                gt_occ = np.vstack((gt_occ, au.cpu().numpy()))
        
        occ = np.zeros(12)
        
        gate = np.array([0.1] * gt_occ.shape[0])
        
        for i in range(gt_occ.shape[1]):
            occ[i] = metrics.f1_score(gt_occ[:, i], (pred_occ[:, i]>gate).astype(int))
        
        print(f'gt_occ[:, i] shape: {gt_occ[:, i].shape}')
        return occ.mean()      