import torch


class Config:
    TRAIN_DATA_PATH = "../dataset/train.csv"
    TEST_DATA_PATH = "../dataset/test.csv"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    EPOCHS = 100
    WEIGHT_DECAY = 1e-4
    MODEL_SAVE_PATH = "../save/best_ResNet340.002.pth"