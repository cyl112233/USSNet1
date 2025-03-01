# import torchvision.models.resnet as re
import torch
import torch.nn as nn
# pretraind = re.resnet34(pretrained=False)
# print(list(pretraind.children())[:-4])
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Fcn(nn.Module):
    def __init__(self, num_class):
        super(Fcn,self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256)#卷积替换池化
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512)
        )

        #
        self.scores1 = nn.Conv2d(512, num_class,1)
        self.scores2 = nn.Conv2d(256, num_class,1)
        self.scores3 = nn.Conv2d(128, num_class,1)

        self.up_8x = nn.ConvTranspose2d(num_class,num_class,4,2,1,bias=True)

        self.up_4x = nn.ConvTranspose2d(num_class, num_class, 4, 2, 1, bias=True)
        self.up_2x = nn.ConvTranspose2d(num_class, num_class, 4, 2, 1, bias=True)


    def forward(self,x):
        x = self.stage1(x)
        s1 = x
        x = self.stage2(x)
        s2 = x
        x = self.stage3(x)
        s3 = x

        #1*1卷积 上采样
        s3 = self.scores1(s3)
        s3 = self.up_2x(s3)

        s2 = self.scores2(s2)
        #print("3",s3.shape)
        #print("2",s2.shape)
        s2 = s2 + s3

        s1 = self.scores3(s1)
        
        s2 = self.up_4x(s2)

        s = s1 + s2
       
        s = self.up_8x(s)
        return s
# input = torch.rand((1,3,1024,512))
# input = torch.rand((1, 3, 1061, 506))
# import torch.nn.functional as F
# # 使用双线性插值进行缩放
# scaled_input = F.interpolate(input, size=(1024, 512), mode='bilinear', align_corners=False)
# MOUDLE = Fcn(3)
# output = MOUDLE(scaled_input)
# print(output.shape)
# from PIL import Image
# import torchvision.transforms as trans
# trans_s = trans.Compose([trans.ToTensor(),
#                          trans.Resize(size=(256, 256))])
#
# Moudel = Fcn(3)
# image = r"D:\PythonProject\Vim-main\1723380554555.jpg"
# image = Image.open(image).convert('RGB')
# image = torch.unsqueeze(trans_s(image), 0)
# output = Moudel(image)
#
# import matplotlib.pyplot as plt
# import numpy as np
# output = torch.transpose(output,1, 3).reshape(output.shape[2], output.shape[3],output.shape[1]).detach().numpy()
# print(output.shape)
# plt.imshow(output)
# plt.show()




