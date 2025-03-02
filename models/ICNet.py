import torch
import torch.nn as nn
import torch.nn.functional as F

class PSPModule(nn.Module):
    """Pyramid Pooling Module (PPM)"""
    def __init__(self, in_channels, pool_sizes):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=s),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            for s in pool_sizes
        ])
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        ppm_outs = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        return torch.cat([x] + ppm_outs, dim=1)


class ICNetBranch(nn.Module):
    """Branch of ICNet processing a specific resolution"""
    def __init__(self, in_channels, out_channels):
        super(ICNetBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ICNet(nn.Module):
    def __init__(self, num_classes):
        super(ICNet, self).__init__()
        
        # Encoder Branches
        self.branch1 = ICNetBranch(3, 32)  # Low-resolution
        self.branch2 = ICNetBranch(32, 64) # Mid-resolution
        self.branch3 = ICNetBranch(64, 128) # High-resolution
        
        # Pyramid Pooling Module
        self.ppm = PSPModule(128, pool_sizes=[1, 2, 3, 6])
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        
        # Apply PPM to the highest resolution feature map
        x3 = self.ppm(x3)
        
        # Upsample and concatenate feature maps
        x2_up = F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x1_up = F.interpolate(x1, size=x3.shape[2:], mode='bilinear', align_corners=True)
        
        x_fused = torch.cat([x1_up, x2_up, x3], dim=1)
        
        # Final classification layer
        out = self.classifier(x_fused)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return out




