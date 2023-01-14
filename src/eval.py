import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall, BinaryPrecision

class WakeWordEval: 
    def __init__(self, device):
        self.accuracy = BinaryAccuracy(threshold=0.5).to(device=device)
        self.precision = BinaryPrecision(threshold=0.5).to(device)
        self.recall = BinaryRecall(threshold=0.5).to(device)
        self.f1score = BinaryF1Score(threshold=0.5).to(device) 

    def eval(self, prediction, true_label): 
        
        accuracy = self.accuracy(prediction, true_label)
        precision = self.precision(prediction, true_label)
        recall = self.recall(prediction, true_label)
        f1score = self.f1score(prediction, true_label) 
        return accuracy, precision, recall, f1score
