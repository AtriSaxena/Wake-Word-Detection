import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.model import WakeWordModel
from utils.util import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test(model_path= "WakeWordDetection.pth", wave_file = "test.wav"): 

    # Create feature
    features = create_lfbe_feature(wave_file)
    features = torch.from_numpy(np.array(features, dtype=np.float32)) 

    # Load Model 
    model = WakeWordModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, DEVICE))
    model.eval()
    print("Model loaded successfully.")

    features = torch.unsqueeze(features, dim=0)
    features = torch.unsqueeze(features, dim=0)
    print(features.shape)
    features = features.permute(0,1,3,2)
    features = features.to(DEVICE) 
    prediction = model(features.float())
    return prediction

print(test(wave_file = "B:\Datasets\speech_commands_v0.01\\four\\00f0204f_nohash_0.wav"))