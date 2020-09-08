import torch
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, n_in, n_out):
        self.n_in = torch.FloatTensor(n_in)
        self.n_out = torch.tensor(n_out)

    def __len__(self):
        return len(self.n_in)

    def __getitem__(self, idx):
        return self.n_in[idx], self.n_out[idx]
