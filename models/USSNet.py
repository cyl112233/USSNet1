

import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F
class MccBlock(nn.Module):
    def __init__(self, in_c, out_c, dp=0.1):
        super(MccBlock,self).__init__()
        self.SSM = nn.Sequential(
            # nn.LayerNorm(in_c),
            Mamba(in_c, d_state=16, expand=2),
            nn.PReLU(),
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
        # 状态空间
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.SSM(x)
        x = x.reshape(B, H, W, C).transpose(1, 3)

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
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.ReLU()
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
class SSNet(nn.Module):
    def __init__(self,output_channel,input_channel=3,dp=0.1):
        super(SSNet,self).__init__()
        #下采样
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

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))

        O4 = self.C9(self.U4(O3, R1))



        return F.softmax(self.pred(O4),1)
# from torchvision import transforms
# Transform  = transforms.Compose([
#     transforms.ToTensor()
#     ]
# )
# import numpy as np
# import PIL.Image as Image
# import tqdm
#
# image = Image.open("").convert('RGB')
# label = Image.open("").convert('L')
# label = torch.tensor(np.array(label)).unsqueeze(0).cuda()
#
# input = torch.unsqueeze(Transform(image),0).cuda()
# moudle = SSNet(2,3,0.2).cuda()
# loss = torch.nn.CrossEntropyLoss()
# OPT = torch.optim.Adam(params=moudle.parameters(), lr=1e-3)
#
# moudle.train()
# for i in tqdm.tqdm(range(150)):
#
#     output = moudle(input)
#     loss_fn = loss(output, label.long())
#
#     OPT.zero_grad()
#     loss_fn.backward()
#     OPT.step()
# moudle.eval()
# with torch.no_grad():
#     output = moudle(input)
#     output = output.cpu()
#     output = np.array(torch.squeeze(output.argmax(dim=1),0),dtype=np.uint8)*100
#     label = label.cpu().squeeze(0)
#     import matplotlib
#     matplotlib.use('qt5agg')
#     import matplotlib.pyplot as plt
#     # plt.imshow(label)
#     plt.imshow(output)
#     plt.show()

