# Kaggle-Digit-Recognizer

## 一、项目介绍
+ 本项目python版本：3.10
+ 本项目基于Pytorch构建了VGG-16和ResNe-34参加kaggle入门竞赛手写数字识别

## 二、内容介绍
+ 安装正常的pytorch环境，还有tqdm、numpy包即可。
+ save用来存放模型字典权重文件，运行程序需在main.py中实例化模型后运行main.py即可
+ 注意推理（预测）时实例化的模型与保存的模型是否一致（不要保存VGG模型实例化ResNet），如果推理（test.py）出错多半是这个原因

## 三、成绩展示
![成绩展示](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/2bbecabad4a090b10429d84ec45a9e1a2f878df4/Kaggle-Digit-Recognizer/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-10-22%20132954.png)
![成绩展示](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/2bbecabad4a090b10429d84ec45a9e1a2f878df4/Kaggle-Digit-Recognizer/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-10-22%20133100.png)
+ 其中，未训练到完全收敛的ResNet-34取得了99.275的准确率，完全训练的VGG-16取得了99.239的准确率，完全训练的ResNet-34取得了99.503的成绩!
+ 其中两次正确率很低的原因是用了仿射变换和随机擦除，实验证明这样会改变数字特征引起模型拟合错误的数据集去得到完全错误的预测能力，因此不对样本进行任何改动是非专业参赛者的明智行为。

## 四、部分资源下载地址
+ pytorch官网下载带cuda的pytorch：https://pytorch.org
+ Anaconda官网下载地址：https://anaconda.org/anaconda/conda
+ kaggle竞赛地址：https://www.kaggle.com/competitions/digit-recognizer
