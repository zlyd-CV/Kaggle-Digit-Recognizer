import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from model.config import Config


# 定义GetData类 用于读取"../dataset"里的数据集,自定义transform参数
class GetData(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 假设数据集的第一列是标签，其他列是特征
        features = self.data_frame.iloc[idx, 1:].values.astype(np.float32)
        label = self.data_frame.iloc[idx, 0].astype(np.int64)
        sample = features.reshape(28, 28)  # 假设每个样本是28x28的图像
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# 定义GetData类 用于读取"../dataset"里的数据集,自定义transform参数


class GetTestData(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 测试数据没有标签，所有列都是特征
        features = self.data_frame.iloc[idx, :].values.astype(np.float32)
        sample = features.reshape(28, 28)  # 假设每个样本是28x28的图像
        if self.transform:
            sample = self.transform(sample)
        return sample, idx  # 返回样本和索引


def get_dataloaders(train_csv, batch_size, transform=None):
    train_dataset = GetData(train_csv, transform=transform)
    # 创建DataLoader实例，打包成可迭代数据
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_test_dataloaders(test_csv, batch_size, transform=None):
    test_dataset = GetTestData(test_csv, transform=transform)
    # 创建DataLoader实例，打包成可迭代数据
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def split_dataset_train_and_validation(train_dataset, validation_rate=0.4):
    assert 0 < validation_rate < 1
    seed = random.randint(1, 32767)
    torch.manual_seed(seed)
    total_size = len(train_dataset)
    validation_size = int(total_size * validation_rate)
    train_size = total_size - validation_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset,
                                                                      [train_size, validation_size])
    train_loader = DataLoader(train_dataset,batch_size=Config.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset,batch_size=Config.BATCH_SIZE, shuffle=False)
    return train_loader, validation_loader


if __name__ == "__main__":
    train_transform = transforms.Compose([transforms.ToTensor()])
    dataset = GetData("../dataset/train.csv", transform=train_transform)
    print(f"Dataset size: {len(dataset)}")
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}, Label: {label}")
