import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.ops as ops
import torch

class Backbone(nn.Module):
    def __init__(self, base_model):
        super(Backbone, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                      "resnet50": models.resnet50(pretrained=True)}

        resnet = self._get_basemodel(base_model)

        self.firstconv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.features = nn.Sequential(
            self.firstconv,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool
        )    

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")
    

    def forward(self, x):
        
        x = self.features(x)
            
        return x

class AU_predict(nn.Module):
    def __init__(self, in_dim):
        super(AU_predict, self).__init__()
        self.predictor = nn.Sequential(
                nn.Linear(in_dim, in_dim // 4),
                nn.ReLU(),
                nn.Linear(in_dim // 4,  in_dim // 8),
                nn.ReLU(),
                nn.Linear(in_dim // 8, 12)
         )
        
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.predictor(x)
        return x
    