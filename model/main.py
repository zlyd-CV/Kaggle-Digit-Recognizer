import torch.optim.lr_scheduler as lr_scheduler
from read_dataset import *
from networks import *
from config import Config
from train import *

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # mean 和 std 都是单值的元组
    ])
    train_dataset = GetData(Config.TRAIN_DATA_PATH, transform=transform)
    model = ResNet34(block=BasicBlock, num_blocks=[3, 4, 6, 3], num_classes=10).to(Config.DEVICE)
    # model = VGG16(num_classes=10).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer_Adam = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=0.9,
                                    weight_decay=Config.WEIGHT_DECAY)
    # 设置学习率更新器: 每 3 个 epoch 将学习率乘以 0.1（即除以 10）
    scheduler_SGD = lr_scheduler.StepLR(optimizer_SGD, step_size=5, gamma=0.1)
    scheduler_Adam = lr_scheduler.ExponentialLR(optimizer_Adam, gamma=0.9)

    train_model(model, train_dataset, criterion, optimizer_Adam, Config.DEVICE, Config.EPOCHS, scheduler=scheduler_Adam)
