import json
from types import NoneType

from PIL import Image
import tqdm
import os
import numpy as np
import cv2


class ImageTransformation():
    def __init__(self,class_name,class_color,data_segmentation,data):
        self.class_name = class_name #类名

        self.class_color_RGB = class_color#颜色rgb
        self.class_color_BGR = []#颜色bgr
        BRG_list = []
        #转换成RGB
        for i in self.class_color_RGB:
            BRG_list.append(i[2])
            BRG_list.append(i[1])
            BRG_list.append(i[0])
            self.class_color_BGR.append(BRG_list)
            BRG_list = []
        self.data_segmentation = data_segmentation
        self.data = data #数据目录
        self.class_num = len(class_name) #类别数
        self.class_num_sum = {} #类别总数
        for i in self.class_name:
            self.class_num_sum[i] = 0

        #索引
        self.cm21b = np.zeros(256 ** 3)
        for i, cm in enumerate(self.class_color_BGR):
            self.cm21b[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        #计数器
        self.register = 0
        self.skipfile = 0

        self.save_png = ".png"

    def open_file(self,file_date):
        '''提取json数据，返回json格式的数据'''
        with open(file_date, "r", encoding="utf-8") as F:
            date = json.load(F)
        return date

    def drawimage(self,date,size):
        black = np.zeros(size, dtype=np.uint8)
        # 图片blackk大小为传入图片尺寸，灰度值全为0，也就是黑色图像
        Background = cv2.cvtColor(black, cv2.COLOR_GRAY2BGR)

        for i in date:
            if i[0] not in self.class_num_sum.keys():
                return np.array([0])

            else:
                self.class_num_sum[i[0]] += 1
                pts = np.array(i[1], np.int32)  # 数据类型必须为 int32
                pts = pts.reshape((-1, 1, 2))  # 将numpy转换为opencv类型
                image = cv2.fillPoly(Background,  # 背景
                                     [pts],  # 点列表
                                     color=self.class_color_BGR[self.class_name.index(i[0])]  # i[0] 是类名json中的类别名称
                                     )
        return image



    def imgtolable(self,img):
        """生成标签矩阵"""
        date = np.array(img, dtype="int32")
        idx = (date[:, :, 0] * 256 + date[:, :, 1]) * 256 + date[:, :, 2]
        return np.array(self.cm21b[idx], dtype="int64")

    def savelabel(self,save_path,image):
            new_imgae = Image.fromarray(image, "L")
            new_imgae.save(save_path)

    def saveimage(self, image_path,save_path):
        image = Image.open(image_path).convert("RGB")
        image.save(save_path)


    def jsontolabel(self):
        image_name = r"/images"
        json_name = r"/json"
        data_len = len(os.listdir(self.data+json_name))

        for i in tqdm.tqdm(os.listdir(self.data+json_name)): #读取数据集
            json_file = os.path.join(self.data+json_name,i) #构建路径
            image_path = os.path.join(self.data + image_name, i[:-5])
            image_file = i[:-5]
            data = self.open_file(json_file) #读取json文件
            dist = [] #保存标注数据
            for i in data["shapes"]:
                x = (i['label'], i["points"])
                dist.append(x)


            image_size = (data["imageHeight"], data["imageWidth"])
            file_suffix = data["imagePath"][-4:]
            #数据分块
            if self.register - self.skipfile < int((data_len - self.skipfile) * self.data_segmentation[0]):
                save_label_name = os.path.join("ImageProcessing/Data/Train/Label", image_file + self.save_png)
                save_image_name = os.path.join("ImageProcessing/Data/Train/Image", image_file + self.save_png)

            if self.register - self.skipfile > int((data_len - self.skipfile) * self.data_segmentation[0]):
                save_label_name = os.path.join("ImageProcessing/Data/Test/Label", image_file + self.save_png)
                save_image_name = os.path.join("ImageProcessing/Data/Test/Image", image_file + self.save_png)
            if self.register - self.skipfile > int((data_len - self.skipfile) * self.data_segmentation[0]) + int((data_len - self.skipfile) * self.data_segmentation[1]):
                save_label_name = os.path.join("ImageProcessing/Data/Val/Label", image_file + self.save_png)
                save_image_name = os.path.join("ImageProcessing/Data/Val/Image", image_file + self.save_png)
            BGR_image = self.drawimage(dist, image_size)  # 转换为rgb图像
            if BGR_image.shape[0] != 1:
                label_image = np.uint8(self.imgtolable(BGR_image))  # 转为label
                self.savelabel(save_label_name,label_image)
                self.saveimage(image_path+file_suffix,save_image_name)
            else:
                # print(F"跳过文件{json_file}")
                txt_file = open("ImageProcessing/Data/Errorjson.txt","a")
                txt_file.write(f"{json_file}\n")
                self.skipfile+=1
            self.register+=1
        print(f"处理文件数{self.register},跳过文件数{self.skipfile},类别标柱数{self.class_num_sum}")







