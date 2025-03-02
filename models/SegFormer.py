import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer

class MLPDecoder(nn.Module):
    """Multi-layer Perceptron Decoder for SegFormer"""
    def __init__(self, embed_dims, num_classes):
        super(MLPDecoder, self).__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(embed_dims[i], 256, kernel_size=1) for i in range(len(embed_dims))
        ])
        self.fuse = nn.Conv2d(256 * len(embed_dims), 256, kernel_size=1)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, features):
        """Fuse multi-scale features and predict segmentation mask"""
        features = [F.interpolate(proj(f), size=features[-1].shape[2:], mode='bilinear', align_corners=True)
                    for proj, f in zip(self.proj, features)]
        fused = torch.cat(features, dim=1)
        fused = self.fuse(fused)
        out = self.classifier(fused)
        return out


class SegFormer(nn.Module):
    def __init__(self, num_classes):
        super(SegFormer, self).__init__()
        
        # Mix Transformer Backbone (MiT)
        self.backbone = VisionTransformer(image_size=224, patch_size=16, num_layers=12,
                                          num_heads=6, hidden_dim=768, mlp_dim=3072)
        
        # Extract features from different transformer stages
        self.embed_dims = [64, 128, 320, 512]  # Example dimensions from MiT
        self.decoder = MLPDecoder(self.embed_dims, num_classes)
    
    def forward(self, x):
        """Forward pass of SegFormer"""
        features = self.backbone(x)  # Extract multi-scale features
        out = self.decoder(features)  # Decode segmentation map
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

