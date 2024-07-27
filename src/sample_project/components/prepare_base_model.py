import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torch.nn as nn
from pathlib import Path
from sample_project.entity.config_entity import PrepareBaseModelConfig
from sample_project.models.vgg16 import VGG16


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    
    def get_base_model(self):
        self.model = VGG16()
        dim = self.config.params_image_dim



        self.save_model(path=self.config.base_model_path, model=self.model)

    
    @staticmethod
    def save_model(path: Path, model: nn.modules):
        torch.save(model,path)

