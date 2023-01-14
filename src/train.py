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
FILE = Path(__file__).resolve() 
ROOT = FILE.parents[1]
print(ROOT)


def train(X_train, y_train, X_test, y_test, EPOCHS, BATCH_SIZE, DEVICE, seed, OPTIMIZER, exp_name):
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
    
    for epoch in tqdm(range(EPOCHS)): 
        print(f"\nEpoch: {epoch}\n")

        # Training 
        train_loss = 0
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

            # 3. Optimizer zero grad 
            optimizer.zero_grad()

            # 4. loss backward 
            loss.backward() 

            # 5. Optimizer step 
            optimizer.step() 

            # Print out how many samples have been seen 
            if batch % 50 == 0:
                print(f"Looked at {batch * len(X)/ len(train_dataloader.dataset)} samples")

        # Divide total train loss by length of train loader (average loss per batch per epoch) 

        train_loss /= len(train_dataloader)

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

        print(f"\n Train Loss: {train_loss:.5f} | Test loss: {test_loss:.5f} Test acc: {test_accuracy:.5f}, Precision: {test_precision:.5f}, Recall:{test_recall:.5f}, f1score:{test_f1score:.5f}\n")

    model_path = ROOT / "models" / exp_name
    if not(os.path.exists(model_path)):
        os.mkdir(model_path)
    torch.save(model.state_dict(), model_path / "WakeWordDetection.pth")


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

def main(arg_out):
    DATASET_PATH = arg_out.dataset
    TARGET_CLASS_NAME = arg_out.target_class
    print(DATASET_PATH)
    print(arg_out.feature_file)
    if not(Path(arg_out.feature_file).is_file()):
        #File doesn't exist 
        data2features = Data2Features(target_class=TARGET_CLASS_NAME, 
                                    data_path=DATASET_PATH)
        X_train, X_test, y_train, y_test = data2features.create_features()
        np.savez(ROOT / "dataset/processed/wakeword_features", X_train = X_train, 
                            X_test = X_test,
                            y_train = y_train,
                            y_test = y_test)

    X_train, y_train, X_test, y_test = load_features() 
    print(f"No of Non-Wakeword class data points:{(y_train==0).sum()}")
    print(f"No of Wakeword class data points: {(y_train==1).sum()}")

    train(X_train, y_train, X_test, y_test, 
                            EPOCHS=arg_out.epochs,
                            BATCH_SIZE= arg_out.batch_size, 
                            DEVICE= arg_out.device,
                            seed = arg_out.seed,
                            OPTIMIZER = arg_out.optimizer,
                            exp_name=arg_out.name)


if __name__ == "__main__": 
    #pass
    arg_out = arg_parse()
    main(arg_out)