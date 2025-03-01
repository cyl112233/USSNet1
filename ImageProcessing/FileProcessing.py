import os
import time
import shutil



class DataFileProcessing:
    def __init__(self):
        pass
    def _creat_folder(self,folder:str):
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
                print(f"成功创建文件夹: {folder}")
            except FileExistsError:
                print(f"文件夹已存在: {folder}")
            except Exception as e:
                print(f"创建文件夹失败: {folder}, 错误: {e}")
    def _remove_folder(self,folder: str):
        if not os.path.exists(folder):
            print("无文件")
        else:
            try:
                shutil.rmtree(folder)
                print(f"成功删除文件夹及其内容: {folder}")
            except Exception as e:
                print(f"删除失败: {folder}, 错误: {e}")
    def update(self):
        self._remove_folder("./ImageProcessing/Data/Train")
        self._remove_folder("./ImageProcessing/Data/Test")
        self._remove_folder("./ImageProcessing/Data/Val")
        time.sleep(0.5)
        self._creat_folder("./ImageProcessing/Data/Train/Image")
        self._creat_folder("./ImageProcessing/Data/Train/Label")
        self._creat_folder("./ImageProcessing/Data/Test/Image")
        self._creat_folder("./ImageProcessing/Data/Test/Label")
        self._creat_folder("./ImageProcessing/Data/Val/Image")
        self._creat_folder("./ImageProcessing/Data/Val/Label")




