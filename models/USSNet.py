import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F
class MccBlock(nn.Module):
    def __init__(self, in_c, out_c, dp=0.1):
        super(MccBlock,self).__init__()
        self.SSM = nn.Sequential(
            # nn.LayerNorm(in_c),
            # nn.Dropout(dp)
        )  # 引入状态空间模型
        self.Resblock = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
            nn.Dropout(dp)
        )

    def forward(self, x):
        block = self.Resblock(x)
        x = self.Conv(x)
        x += block

        return x  #return F.relu(x)改为return x
class CCRBlock(nn.Module):
    def __init__(self, in_c, out_c, dp=0.2):
        super(CCRBlock,self).__init__()
        self.Resblock = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
            nn.Dropout(dp)
        )
    def forward(self, x):
        block = self.Resblock(x)
        x = self.Conv(x)
        x += block
        return x  #return F.relu(x)改为return x
class DownSampling(nn.Module):
    '''下采样模块'''
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.Down(x)
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        

        up = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.Up(up)
    
        return torch.cat((x, r),1)
class SSNet(nn.Module):
    def __init__(self,output_channel,input_channel=3,dp=0.1):
        super(SSNet,self).__init__()
        
        self.C1 = MccBlock(input_channel, 64,dp=dp)
        self.D1 = DownSampling(64)
        self.C2 = CCRBlock(64, 128, dp=dp)
        self.D2 = DownSampling(128)
        self.C3 = MccBlock(128, 256, dp=dp)
        self.D3 = DownSampling(256)
        self.C4 = CCRBlock(256, 512, dp=dp)
        self.D4 = DownSampling(512)
        self.C5 = MccBlock(512, 1024, dp=dp)
        # 上采样
        self.U1 = UpSampling(1024)
        self.C6 = MccBlock(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = CCRBlock(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = MccBlock(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = CCRBlock(128, 64)
        self.pred = torch.nn.Conv2d(64, output_channel, 1)


    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))
        
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))
        return F.softmax(self.pred(O4),1)
