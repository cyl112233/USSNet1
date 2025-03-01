import cv2

# 图片的路径
image_path = '/home/bift/Y_L/fei/Seg_training/Seg_training/ImageProcessing/Data/Val/Label/1723348505284.jpg'

# 使用cv2.imread()函数读取图片
image = cv2.imread(image_path)

# 检查图片是否正确读取
if image is not None:
    # 使用cv2.imshow()函数显示图片
    cv2.imshow('Image', image)

    # 等待键盘事件，0表示无限期等待
    cv2.waitKey(0)

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()