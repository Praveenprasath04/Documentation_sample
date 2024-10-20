import os
import urllib.request as request
import zipfile
from sample_project import logger
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.utils.data as data
import numpy as np
import torch
from sample_project.entity.config_entity import DataPreprocessConfig
from pathlib import Path


class DataPreprocess:
    def __init__(self, config: DataPreprocessConfig):
        self.config = config
    
    def transform_data(self):
        image_dim  =self.config.params_image_dim
        data_dir = self.config.data_dir

        train_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((image_dim,image_dim)),
                                      transforms.Normalize((0.1307,), (0.3081,))])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((image_dim,image_dim)),
                                     transforms.Normalize((0.1307,), (0.3081,))])
        

        self.train_data = datasets.MNIST(root = data_dir,
                            train = True,                         
                            transform = train_transform,
                            
                            )
        self.test_data = datasets.MNIST( root = data_dir, 
                            train = False, 
                            transform = test_transform,
                            )
    def split_valid_set(self):
    
        valid_size = self.config.params_valid_size

        train_data = self.train_data
        test_data  = self.test_data

        valid_set_length = int(valid_size*len(train_data))
        train_set_length = len(train_data) - valid_set_length

        train_data,valid_data=data.random_split(train_data,[train_set_length,valid_set_length])

        self.valid_data = valid_data


        logger.info(f"Train_data length: {len(train_data)}, Valid_data length: {len(valid_data)},Test_data length: {len(test_data)}")
    
    def data_loaders(self):

        batch_size = self.config.params_batch_size

        train_iterator= data.DataLoader(self.train_data,
                                        shuffle=True,
                                        batch_size=batch_size)

        valid_iterator= data.DataLoader(self.valid_data,
                                        batch_size=batch_size)

        test_iterator= data.DataLoader(self.test_data,
                                        batch_size=batch_size)
        
        self.save_loader(self.config.train_loader_dir,train_iterator)
        self.save_loader(self.config.valid_loader_dir,valid_iterator)
        self.save_loader(self.config.test_loader_dir,test_iterator)
        
    @staticmethod
    def save_loader(path:Path,loader: data.DataLoader):
        torch.save(loader,path)
        logger.info(f"Saved at , {path}")
                


