import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
from tqdm import tqdm 
import argparse
from data.data_loader import torch_dataloader 
from models.model import WakeWordModel
import sys
import os
from pathlib import Path 
from data.data_to_features import Data2Features
from utils.util import *
from eval import WakeWordEval 
from visualization.visualisation import VisualizeData
import mlflow.pytorch
import mlflow
FILE = Path(__file__).resolve() 
ROOT = FILE.parents[1]


def train(X_train, y_train, X_test, y_test, EPOCHS, BATCH_SIZE, DEVICE, seed, OPTIMIZER, exp_name, visualization_class):
    torch.manual_seed(seed) 

    #Loading model 
    model = WakeWordModel().to(device=DEVICE) 

    
    # SETUP optimizer
    if OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters())
    elif OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.8)
    
    # Load data from Dataloader
    train_dataloader, test_dataloader = torch_dataloader(X_train, y_train, X_test, y_test) 
    
    #Load Evaluation object
    evaluate = WakeWordEval(DEVICE)
    all_train_loss = [] 
    all_val_loss = []
    all_train_metrics = {'train_accuracy':[], 
                         'train_precision': [],
                         'train_recall': [],
                         'train_f1score': []}
    all_val_metrics ={'val_accuracy':[], 
                         'val_precision': [],
                         'val_recall': [],
                         'val_f1score': []}
    for epoch in tqdm(range(EPOCHS)): 
        print(f"\nEpoch: {epoch}\n")

        # Training 
        train_loss, train_accuracy, train_precision, train_recall, train_f1score = 0, 0, 0, 0, 0 
        # Add a loop to loop training through batches 
        for batch, (X, y) in enumerate(train_dataloader):
            model.train() 
            
            X = torch.unsqueeze(X, dim=1)
            X = X.permute(0,1,3,2)
            X, y = X.to(DEVICE), y.to(DEVICE)
            # 1. Forward pass 
            y_pred = model(X.float()) 
            loss_fn = nn.BCELoss()
            # 2. Calculate loss (per batch) 

            loss = loss_fn(y_pred.squeeze(dim=1), y.float()) 
            train_loss += loss 

            accuracy, precision, recall, f1score =  evaluate.eval(y_pred.squeeze(dim=1), y)
            train_accuracy += accuracy 
            train_precision += precision 
            train_recall += recall 
            train_f1score += f1score 
            
            # 3. Optimizer zero grad 
            optimizer.zero_grad()

            # 4. loss backward 
            loss.backward() 

            # 5. Optimizer step 
            optimizer.step() 

        # Divide total train loss by length of train loader (average loss per batch per epoch) 

        train_loss /= len(train_dataloader)
        train_accuracy /= len(train_dataloader)
        train_precision /= len(train_dataloader)
        train_recall /= len(train_dataloader)
        train_f1score /= len(train_dataloader) 
        
        log_scalar('train_loss', train_loss, epoch)
        log_scalar('train_accuracy', train_accuracy, epoch)
        log_scalar('train_precision', train_precision, epoch)
        log_scalar('train_recall', train_recall, epoch)
        log_scalar('train_f1score', train_f1score, epoch)

        ### Testing 
        # Setup variable for accumulatively adding up loss and accuracy 

        test_loss, test_accuracy, test_precision, test_recall, test_f1score = 0, 0, 0, 0, 0 
        model.eval() 

        with torch.inference_mode(): 
            for X,y in test_dataloader:
                X = torch.unsqueeze(X, dim=1)
                X = X.permute(0,1,3,2)
                X, y = X.to(DEVICE), y.to(DEVICE)
                # 1. Forward pass 
                test_pred = model(X.float()) 

                # 2. Calculate loss 
                test_loss += loss_fn(test_pred.squeeze(dim=1), y.float()) 

                #3. Test Accuracy
                accuracy, precision, recall, f1score =  evaluate.eval(test_pred.squeeze(dim=1), y) 
                test_accuracy += accuracy 
                test_precision += precision 
                test_recall += recall 
                test_f1score += f1score 

            test_loss /= len(test_dataloader)
            test_accuracy /= len(test_dataloader)
            test_precision /= len(test_dataloader)
            test_recall /= len(test_dataloader)
            test_f1score /= len(test_dataloader) 
            
        #Mlflow Record
        log_scalar('test_loss', test_loss, epoch)
        log_scalar('test_accuracy', test_accuracy, epoch)
        log_scalar('test_precision', test_precision, epoch)
        log_scalar('test_recall', test_recall, epoch)
        log_scalar('test_f1score', test_f1score, epoch)
        
        
        #Record loss and metrics 
        print(train_accuracy, type(train_accuracy))
        all_train_loss.append(train_loss.cpu().detach().numpy().flat[0])
        all_val_loss.append(test_loss.cpu().detach().numpy().flat[0]) 
        all_train_metrics['train_accuracy'].append(train_accuracy.cpu().detach().numpy().flat[0])
        all_val_metrics['val_accuracy'].append(test_accuracy.cpu().detach().numpy().flat[0])
        all_train_metrics['train_precision'].append(train_precision.cpu().detach().numpy().flat[0])
        all_val_metrics['val_precision'].append(test_precision.cpu().detach().numpy().flat[0])
        all_train_metrics['train_recall'].append(train_recall.cpu().detach().numpy().flat[0])
        all_val_metrics['val_recall'].append(test_recall.cpu().detach().numpy().flat[0])
        all_train_metrics['train_f1score'].append(train_f1score.cpu().detach().numpy().flat[0])
        all_val_metrics['val_f1score'].append(test_f1score.cpu().detach().numpy().flat[0])
        
        

        print(f"\n Train Loss: {train_loss:.5f} | Test loss: {test_loss:.5f} Test acc: {test_accuracy:.5f}, Precision: {test_precision:.5f}, Recall:{test_recall:.5f}, f1score:{test_f1score:.5f}\n")
    visualization_class.visualize_training_metrics(all_train_loss, all_val_loss, all_train_metrics, all_val_metrics)
    model_path = ROOT / "models" / exp_name
    if not(os.path.exists(model_path)):
        os.mkdir(model_path)
    state_dict = model.state_dict()
    torch.save(state_dict, model_path / "WakeWordDetection.pth")
    mlflow.pytorch.log_state_dict(state_dict, artifact_path="model")
    mlflow.pytorch.log_model(model, 'model')


def arg_parse():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset", type=str, default= ROOT / 'dataset/', help='Dataset path')
    parser.add_argument('--feature_file', type=str, default=ROOT / 'dataset/processed/wakeword_features.npz')
    parser.add_argument("--epochs", type=int, default= 25, help = 'total training epochs') 
    parser.add_argument("--batch_size", type= int, default= 32, help= 'Batch size of data') 
    parser.add_argument("--device", default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam', help='optimizer')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed') 
    parser.add_argument('--target_class', type=str, default='stop', help='Target class for wake word')

    return parser.parse_args()

def log_scalar(name, value, step): 
    mlflow.log_metric(name, value, step=step)

def main(arg_out):
    DATASET_PATH = arg_out.dataset
    TARGET_CLASS_NAME = arg_out.target_class

    if not(Path(arg_out.feature_file).is_file()):
        #File doesn't exist 
        data2features = Data2Features(target_class=TARGET_CLASS_NAME, 
                                    data_path=DATASET_PATH)
        X_train, X_test, y_train, y_test = data2features.create_features()
        np.savez(ROOT / "dataset/processed/wakeword_features", X_train = X_train, 
                            X_test = X_test,
                            y_train = y_train,
                            y_test = y_test)

    X_train, y_train, X_test, y_test = load_features(file_name= ROOT / "dataset/processed/wakeword_features.npz") 
    print(f"No of Non-Wakeword class data points:{(y_train==0).sum()}")
    print(f"No of Wakeword class data points: {(y_train==1).sum()}")
    
    visualization_class = VisualizeData(exp_name=ROOT / "reports" / arg_out.name, epochs=arg_out.epochs)
    visualization_class.visualize_train_data(X_train = X_train, y_train = y_train)
    
    # Train the model
    with mlflow.start_run() as run:
        
        for key, value in vars(arg_out).items():
            mlflow.log_param(key, value)
        
        train(X_train, y_train, X_test, y_test, 
                            EPOCHS=arg_out.epochs,
                            BATCH_SIZE= arg_out.batch_size, 
                            DEVICE= arg_out.device,
                            seed = arg_out.seed,
                            OPTIMIZER = arg_out.optimizer,
                            exp_name=arg_out.name,
                            visualization_class = visualization_class)


if __name__ == "__main__": 
    arg_out = arg_parse()
    main(arg_out)