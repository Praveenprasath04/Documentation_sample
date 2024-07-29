import urllib.request as request
from sample_project import logger
import numpy as np
import torch
from sample_project.utils.common import accuracy
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sample_project.entity.config_entity import TestingConfig
from sklearn.exceptions import UndefinedMetricWarning
import warnings

class Testing:
    def __init__(self, config: TestingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    def loading_iterators(self):
        self.test_iterator = torch.load(self.config.test_loader_dir)

    def initializing_model(self):
        self.model = torch.load(self.config.trained_model_path)
        self.model = self.model.to(self.device)  # Move model to the specified device
        self.loss_fn = nn.CrossEntropyLoss()

    def test(self):
        self.test_loop(self.model,self.test_iterator,self.loss_fn)



    @staticmethod    
    def test_loop(model,test_iterator,Loss_function):
        device = next(model.parameters()).device 
        model.eval()
        loss_vals = []
        acc_vals = []
        prec_vals = []
        recall_vals = []
        f1_vals= []
        with torch.no_grad():
            for iteration, idata in enumerate(test_iterator,1):
                
                inputs,targets = idata
                inputs,targets = inputs.to(device),targets.to(device)
                
                outputs = model(inputs)
                
                current_loss = Loss_function(outputs,targets)
                current_accuracy = accuracy(outputs,targets)
                
                outputs_np = outputs.cpu().detach().numpy()
                targets_np = targets.cpu().detach().numpy()

                warnings.filterwarnings('ignore', category=UndefinedMetricWarning) 
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets_np, np.argmax(outputs_np, axis=1), average='weighted')
                
                loss_vals.append(float(current_loss))
                acc_vals.append(float(current_accuracy))
                prec_vals.append(precision)
                recall_vals.append(recall)
                f1_vals.append(f1)

                print("\rEvaluating the model: {}/{} ({:.1f}%) Loss: {:.5f} accuracy: {:.5f}".format(
                        iteration, len(test_iterator),
                        iteration * 100 / len(test_iterator),current_loss,current_accuracy),
                    end=" " * 10)
                
                
        avg_loss = np.mean(loss_vals)
        avg_acc = np.mean(acc_vals)
        avg_precision = np.mean(prec_vals)
        avg_recall = np.mean(recall_vals)
        avg_f1 = np.mean(f1_vals)
        
        logger.info("\rTest Results: Loss: {:.6f}, Accuracy: {:.4f}%, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
            avg_loss, avg_acc * 100, avg_precision, avg_recall, avg_f1))
