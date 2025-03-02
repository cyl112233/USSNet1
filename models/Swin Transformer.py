import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import swin_t

class SwinSegmentationHead(nn.Module):
    """Segmentation Head for Swin Transformer"""
    def __init__(self, in_channels, num_classes):
        super(SwinSegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x


class SwinSegmentation(nn.Module):
    """Swin Transformer for Semantic Segmentation"""
    def __init__(self, num_classes):
        super(SwinSegmentation, self).__init__()
        
        # Load Pretrained Swin Transformer Backbone
        self.backbone = swin_t(weights=None)
        self.backbone.head = nn.Identity()  # Remove classification head
        
        # Feature extraction layers (example: use last stage output)
        self.segmentation_head = SwinSegmentationHead(in_channels=768, num_classes=num_classes)
        
    def forward(self, x):
        features = self.backbone(x)  # Extract features from Swin Transformer
        out = self.segmentation_head(features)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)  # Upsample to input size
        return out


if __name__ == "__main__":
    model = SwinSegmentation(num_classes=19)
    x = torch.randn(1, 3, 512, 512)  # Example input
    y = model(x)
    print(y.shape)  # Expected output shape: (1, num_classes, 512, 512)

