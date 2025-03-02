import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16

class ViTSegmentationHead(nn.Module):
    """Segmentation Head for Vision Transformer"""
    def __init__(self, in_channels, num_classes):
        super(ViTSegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x


class ViTSegmentation(nn.Module):
    """Vision Transformer for Semantic Segmentation"""
    def __init__(self, num_classes):
        super(ViTSegmentation, self).__init__()
        
        # Load Pretrained Vision Transformer Backbone
        self.backbone = vit_b_16(weights=None)
        self.backbone.heads = nn.Identity()  # Remove classification head
        
        # Feature extraction layers (example: use last stage output)
        self.segmentation_head = ViTSegmentationHead(in_channels=768, num_classes=num_classes)
        
    def forward(self, x):
        features = self.backbone(x)  # Extract features from Vision Transformer
        B, N, C = features.shape
        H = W = int(N ** 0.5)  # Assuming square patches
        features = features.permute(0, 2, 1).reshape(B, C, H, W)  # Reshape to feature map
        out = self.segmentation_head(features)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)  # Upsample to input size
        return out




