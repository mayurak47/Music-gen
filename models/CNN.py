import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, input_size, target_size):
        super(CNN, self).__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv1d(1, 64, 7)
        self.conv2 = nn.Conv1d(64, 128, 7)
        self.conv3 = nn.Conv1d(128, 128, 6)
        self.conv4 = nn.Conv1d(128, 256, 6)

        self.maxpool1 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(256, 256, 4)
        self.conv6 = nn.Conv1d(256, 512, 4)
        self.conv7 = nn.Conv1d(512, 512, 4)
        self.conv8 = nn.Conv1d(512, 1024, 4)
        self.conv9 = nn.Conv1d(1024, 1024, 4)

        self.maxpool2 = nn.MaxPool1d(2)

        self.lin1 = nn.Linear(1024*12, target_size)
        self.lin2 = nn.Linear(target_size, target_size)

    def forward(self, seq):
        seq = seq.view(-1, 1, self.input_size)

        seq = F.relu(self.conv1(seq))
        seq = F.relu(self.conv2(seq))
        seq = F.relu(self.conv3(seq))
        seq = F.relu(self.conv4(seq))

        seq = self.maxpool1(seq)

        seq = F.relu(self.conv5(seq))
        seq = F.relu(self.conv6(seq))
        seq = F.relu(self.conv7(seq))
        seq = F.relu(self.conv8(seq))
        seq = F.relu(self.conv9(seq))
        
        seq = self.maxpool2(seq)
        
        seq = F.relu(self.lin1(seq.view(-1, 1024*12)))
        seq = self.lin2(seq)
        return seq