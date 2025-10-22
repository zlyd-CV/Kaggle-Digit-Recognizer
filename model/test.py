import torch
import pandas as pd
from torchvision.transforms import transforms
from tqdm import tqdm
from config import Config
from model.networks import VGG16
from networks import AlexNet, ResNet34, BasicBlock
from read_dataset import get_test_dataloaders

def test():
    """
    对测试集进行预测，并将结果保存为CSV文件
    """
    # 设置设备
    device = Config.DEVICE
    print(f"使用设备: {device}")

    # 定义数据预处理transform (与训练时保持一致)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # mean 和 std 都是单值的元组
    ])

    # 加载测试数据
    print("加载测试数据...")
    test_loader = get_test_dataloaders(
        Config.TEST_DATA_PATH,
        batch_size=Config.BATCH_SIZE,
        transform=test_transform
    )
    print(f"测试数据集大小: {len(test_loader.dataset)}")

    # 创建模型实例 (根据你训练时使用的模型选择)
    # 如果训练时使用的是AlexNet，使用下面这行:
    # model = AlexNet(num_classes=10).to(device)

    # 如果训练时使用的是ResNet34，使用下面这行:
    model = ResNet34(BasicBlock, [3, 4, 6, 3], num_classes=10).to(device)
    # model = VGG16(num_classes=10).to(device)

    # 加载训练好的模型权重
    print(f"加载模型权重: {Config.MODEL_SAVE_PATH}")
    try:
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        # 如果checkpoint是字典，提取model部分
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print("模型权重加载成功！")
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        return

    # 设置模型为评估模式
    model.eval()

    # 存储预测结果
    predictions = []
    image_ids = []

    # 进行预测
    print("开始预测...")
    with torch.no_grad():
        for images, indices in tqdm(test_loader, desc="预测进度"):
            images = images.to(device)

            # 前向传播
            outputs = model(images)

            # 获取预测类别
            _, predicted = torch.max(outputs, 1)

            # 保存预测结果
            predictions.extend(predicted.cpu().numpy().tolist())
            image_ids.extend((indices.numpy() + 1).tolist())  # ImageId从1开始

    # 创建DataFrame
    submission = pd.DataFrame({
        'ImageId': image_ids,
        'Label': predictions
    })

    # 按ImageId排序
    submission = submission.sort_values('ImageId')

    # 保存为CSV文件
    output_path = "test.csv"
    submission.to_csv(output_path, index=False)
    print(f"\n预测完成！结果已保存到: {output_path}")
    print(f"预测样本总数: {len(submission)}")
    print("\n前10条预测结果:")
    print(submission.head(10))


if __name__ == "__main__":
    test()
