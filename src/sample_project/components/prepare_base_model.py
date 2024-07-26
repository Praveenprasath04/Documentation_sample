import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torch.nn as nn
from pathlib import Path
from sample_project.entity.config_entity import PrepareBaseModelConfig

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(1,64,3,1,1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64,64,3,1,1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Conv2d(64,128,3,1,1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Conv2d(128,128,3,1,1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Conv2d(128,256,3,1,1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.Conv2d(256,256,3,1,1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.Conv2d(256,256,3,1,1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Conv2d(256,512,3,1,1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Conv2d(512,512,3,1,1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Conv2d(512,512,3,1,1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Flatten(),
                                    nn.Linear(2048,4096),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(4096,1028),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(1028,10),

                        
                                
        )
    def forward(self,x):
        return self.layers(x)
    
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    
    def get_base_model(self):
        self.model = VGG16()
        dim = self.config.params_image_dim
        print(self.model)


        self.save_model(path=self.config.base_model_path, model=self.model)

    
    @staticmethod
    def save_model(path: Path, model: nn.modules):
        torch.save(model,path)

