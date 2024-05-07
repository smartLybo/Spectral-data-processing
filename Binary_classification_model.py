import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error

# #  The binary classification models mainly include CNN, BLS, and Bi-LSTM

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 3, kernel_size=10, padding=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=5)
        self.conv2 = nn.Conv1d(3, 3, kernel_size=15, padding=10)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=10)
        self.fc1 = nn.Linear(90, 45)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(45, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1500, hidden_size=64, num_layers=3, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64 *2, 2)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x)
        return x

def show_accuracy(predictLabel, Label):
    if isinstance(predictLabel, torch.Tensor):
        predictLabel= predictLabel.detach().numpy()
        predictLabel=np.argmax(predictLabel, axis=1)
    Label = np.argmax(Label, axis=1)
    Label = np.ravel(Label).tolist()
    predictLabel = np.ravel(predictLabel).tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count / len(Label), 5))

