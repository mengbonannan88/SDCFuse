import os

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from model.common import RGB2YCrCb

to_tensor = transforms.Compose([transforms.ToTensor()])



class llvip(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  # 获得TNO数据集的子目录
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'Inf':
                self.inf_path = temp_path  # 获得红外路径
            elif sub_dir == 'Vis':
                self.vis_rain = temp_path  # 获得可见光路径
            elif sub_dir == 'gt':
                self.vis_gt = temp_path  # 获得可见光路径

        self.name_list = os.listdir(self.vis_gt)  # 获得子目录下的图片的名称
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L').resize((480,480)) # 获取红外图像
        vis_rain = Image.open(os.path.join(self.vis_rain, name)).resize((480,480))
        vis_gt = Image.open(os.path.join(self.vis_gt, name)).resize((480,480))

        #####处理信息熵选择权重

        inf_image = self.transform(inf_image)
        vis_rain = self.transform(vis_rain)
        vis_gt = self.transform(vis_gt)

        vis_rain_y, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_rain)
        vis_gt_y, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_gt)

        vis_clip = Image.open(os.path.join(self.vis_rain, name)).resize((224,224))
        vis_clip = self.transform(vis_clip)

        return vis_rain_y, vis_gt_y, inf_image, vis_cb_image, vis_cr_image, name, vis_rain, vis_clip

    def __len__(self):
        return len(self.name_list)


