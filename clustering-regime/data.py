import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class TensorData(Dataset):

    def __init__(self, vdata):
        self.vdata = torch.FloatTensor(vdata) 
        self.len = self.vdata.shape[0] 

    def __getitem__(self, index):
        return self.vdata[index]

    def __len__(self):
        return self.len 
        
def datasets(data_path, batch_size, shuffle=False):
    train_data = TensorData(np.load(data_path)) 
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle) 
    return dataloader
