# 导入所需的库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# 定义PSPNet类
class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        self.num_classes = num_classes  # 设置类别数
        self.backbone = models.resnet50(pretrained=True)  # 使用预训练的ResNet50作为特征提取器
        # 定义金字塔池化模块，每个模块的pool_size可以不同
        self.layer5a = _PSPModule(2048, 512, pool_size=1)  # 定义金字塔池化模块，池化大小为1
        self.layer5b = _PSPModule(2048, 512, pool_size=2)  # 定义金字塔池化模块，池化大小为2
        self.layer5c = _PSPModule(2048, 512, pool_size=3)  # 定义金字塔池化模块，池化大小为3
        self.layer5e = _PSPModule(2048, 512, pool_size=6)  # 定义金字塔池化模块，池化大小为6
        self.fc = nn.Conv2d(512 * 4, num_classes, kernel_size=1)  # 定义1x1卷积层，输出类别数

    def forward(self, x):
        x = self.backbone(x)  # 使用ResNet50提取特征
        x1 = self.layer5a(x)  # 通过金字塔池化模块
        x2 = self.layer5b(x)  # 通过金字塔池化模块
        x3 = self.layer5c(x)  # 通过金字塔池化模块
        x4 = self.layer5e(x)  # 通过金字塔池化模块
        x = torch.cat((x1, x2, x3, x4), dim=1)  # 沿通道维度拼接特征图
        x = self.fc(x)  # 通过1x1卷积层输出最终预测结果
        return x


# 定义金字塔池化模块
class _PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(_PSPModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)  # 自适应平均池化层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积层
        self.bn = nn.BatchNorm2d(out_channels)  # 批标准化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

    def forward(self, x):
        x = self.pool(x)  # 应用自适应平均池化
        x = self.conv(x)  # 应用1x1卷积
        x = self.bn(x)  # 应用批标准化
        x = self.relu(x)  # 应用ReLU激活函数
        return x











