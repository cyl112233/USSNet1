from ConfigProject import Config
import torch

if __name__ == '__main__':
    # from Moudle.UssNet import USS_Net
    # from Moudle.umamba import UNet
    #from Moudle.Munet2 import SSNet
    from Moudle.FCN1 import Fcn

    main = Config("./config.yaml")
    # Moudle = USS_Net(len(main.ClassName))
    # Moudle = UNet(3, len(main.ClassName))
    # Moudle = UNet(3, 3)
    # Moudle = SSNet(output_channel=len(main.ClassName),
    #                 input_channel=3,
    #                 dp=0.1)
    Moudle = Fcn(3)
    loss_function = torch.nn.CrossEntropyLoss()
    Cuda_num = [i for i in range(torch.cuda.device_count())]  # 获取GPU数量
    Moudle = torch.nn.DataParallel(module=Moudle, device_ids=Cuda_num)  # 多卡并行
    Moudle = Moudle.cuda(device=Cuda_num[0])  # 从第一张卡开始索引

    main.MainRun(Cuda_num=Cuda_num, Moudle=Moudle, loss=loss_function)