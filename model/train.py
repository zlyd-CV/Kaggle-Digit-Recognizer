import torch
from tqdm import tqdm
from config import Config
import os
from read_dataset import *

def train_model(model, train_dataset, criterion, optimizer, device, epoches, scheduler=None):
    model.to(device)
    best_loss = float('inf')

    for epoch in range(epoches):
        print('=' * 50)
        # 每个epoch随机采样法划分训练集和测试集
        train_loader,validation_loader=split_dataset_train_and_validation(train_dataset,validation_rate=0.4)

        if os.path.exists(Config.MODEL_SAVE_PATH):
            model_parameters = torch.load(Config.MODEL_SAVE_PATH)
            model.load_state_dict(model_parameters['model'])
            optimizer.load_state_dict(model_parameters['optimizer'])
            if scheduler and 'scheduler' in model_parameters:
                scheduler.load_state_dict(model_parameters['scheduler'])
            best_loss = model_parameters['best_loss']
            tqdm.write(f"加载历史最优模型成功,最优损失{best_loss:.4f}")
        else:
            tqdm.write(f"没有找到保存的最优模型，加载失败")

        model.train()
        train_total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        train_pbar = tqdm(
            train_loader, desc=f"训练进度{epoch+1}/{epoches}", unit="batch", leave=False)
        for index, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            train_pbar.set_postfix({"loss": loss.item()})

        if scheduler:
            scheduler.step()
        epoch_loss = train_total_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        tqdm.write(f"\n{epoch+1}/{epoches}:Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f},当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 对验证集进行验证
        validation_total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        model.eval()
        with torch.no_grad():
            validation_pbar = tqdm(validation_loader, desc=f"验证进度{epoch + 1}/{Config.EPOCHS}", unit="batches",
                                         leave=False)
            for index, (images, labels) in enumerate(validation_pbar):
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_total_loss+= loss.item() * images.size(0)
                _,predicted = torch.max(outputs, 1)
                batch_correct = predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
                correct_predictions += batch_correct
                validation_pbar.set_postfix(loss=loss.item())

            epoch_loss = validation_total_loss / total_samples
            epoch_accuracy = correct_predictions / total_samples
            tqdm.write(f"{epoch + 1}/{epoches}:Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}")

        if epoch_loss <= best_loss or True: # 使用的是验证集的损失比较
            best_loss = epoch_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_loss': best_loss
            }, f"{Config.MODEL_SAVE_PATH}")
            print(f"最优模型已保存")
        else:
            print(f"当前epoch训练损失升高，即将加载历史最优模型...")
