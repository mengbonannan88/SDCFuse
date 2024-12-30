import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# ---- 模型部分：low_light_enhance ----
class derain_net(nn.Module):
    def __init__(self):
        super(derain_net, self).__init__()
        # 卷积层：用于提取特征
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # 提取透射率图的特征
        x = nn.LeakyReLU()(self.conv1(x))
        x = nn.LeakyReLU()(self.conv2(x))
        x = nn.LeakyReLU()(self.conv3(x))

        return x

class derain_image(nn.Module):
    def __init__(self, dim):
        super(derain_image, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=dim, kernel_size=3, stride=1, padding=1)

    def forward(self, image, base_feature):
        return image - self.conv(base_feature)



