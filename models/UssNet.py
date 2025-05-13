import torch
import torch.nn as  nn
# from mamba_ssm import Mamba
import torch.nn.functional as F

class SSM_Block(nn.Module):
    def __init__(self,in_c,out_c,dp=0.1):
        super().__init__()
        # self.SSM = nn.Sequential(
        #     Mamba(in_c,d_state=16,expand=2),
        #     nn.GELU(),
        #     # nn.Dropout(dp)

        self.Conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1,bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            # nn.Dropout(dp),
        )#配合卷积操作
        self.BLOCK =nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU()
        )


    def forward(self,x):
        # block = self.BLOCK(x)
        # x = self.SSM(x)

        # x = x.reshape(B,H,W,C) .transpose(1, 3)
        x = self.Conv(x)
        # x+=block
        return x

class DownSampling(nn.Module):
    '''下采样模块'''
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.GELU()
        )

    def forward(self, x):
        return self.Down(x)
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样

        up = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r),1)

class USS_Net(nn.Module):
    def __init__(self,output_channel,input_channel=3):
        super(USS_Net,self).__init__()
        #下采样
        self.C1 = SSM_Block(input_channel, 64)
        self.D1 = DownSampling(64)
        self.C2 = SSM_Block(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = SSM_Block(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = SSM_Block(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = SSM_Block(512, 1024)
        # 上采样
        self.U1 = UpSampling(1024)
        self.C6 = SSM_Block(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = SSM_Block(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = SSM_Block(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = SSM_Block(128, 64)
        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, output_channel, 1)

        self.res1 = nn.Conv2d(3,64,1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))


        return F.softmax(self.pred(O4),1)
#
