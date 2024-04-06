import torch
from torch.utils.data import Dataset, DataLoader
class CR_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # 数据传入GPU
        feature = torch.tensor(sample['feature'], dtype=torch.float32)
        # 由于特征混入了距离因素，会导致数据中有超大值，后续训练导致爆梯度，故在这里正则化一次
        mean_f=torch.mean(feature)
        std_f=torch.std(feature)
        feature=(feature-mean_f)/(std_f+1e-4)
        label = torch.tensor(sample['label'], dtype=torch.long)
        return feature, label