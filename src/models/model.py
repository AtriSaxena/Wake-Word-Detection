import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class WakeWordModel(nn.Module):
    """WakeWordModel - as described in
            https://assets.amazon.science/db/e3/571c26744b2d9c94f77969b1277e/accurate-detection-of-wake-word-start-and-end-using-a-cnn.pdf
    """
    def __init__(self):
        super(WakeWordModel, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels = 1, 
                                out_channels = 96,
                                kernel_size = (9,5)
                                ),
                       nn.MaxPool2d(kernel_size=(2,3),
                       ))
        self.relu = nn.ReLU()
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=96,
                                        out_channels=192,
                                        kernel_size=(7,3),
                                        stride=(3,1)),
                                    nn.MaxPool2d(kernel_size=(1,2)))
        self.conv_3 = nn.Conv2d(in_channels=192, out_channels=192,
                                    kernel_size=(4,3),
                                    stride=1)
        self.conv_4 = nn.Conv2d(192,192,3)
        self.conv_5 = nn.Conv2d(192,192,kernel_size=3)
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(in_features=960, out_features=500)
        self.dense_2 = nn.Linear(in_features=500, out_features=500)
        self.output = nn.Linear(in_features=500, out_features=1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv_1(x))

        x = self.relu(self.conv_2(x)) 

        x = self.relu(self.conv_3(x))

        x = self.relu(self.conv_4(x))

        x = self.relu(self.conv_5(x))

        x = self.flatten(x) 
        x = self.relu(self.dense_1(x)) 
        x = self.relu(self.dense_2(x)) 
        x = self.sigmoid(self.output(x))
        return x

model = WakeWordModel()
print(model)