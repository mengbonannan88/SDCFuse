import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)


#--------------导入去雾，去雨，低光增强部分--------------#
from model.dehaze import DehazeNet, dehaze_image
from model.low_enhance import Low_enhance_net, low_enhance_image
from model.gate import GatedLayer
class F_Net(nn.Module):
    def __init__(self):
        super(F_Net, self).__init__()
        # ----------参数生成或者特征提取部分-----------------#
        self.dehaze_net = DehazeNet()
        self.low_enhance_net = Low_enhance_net()

        self.conv1_vi = ConvBnLeakyRelu2d(1, 16)
        self.conv2_vi = ConvBnLeakyRelu2d(16, 32)
        self.conv3_vi = ConvBnLeakyRelu2d(32, 64)
        self.conv4_vi = ConvBnLeakyRelu2d(64, 128)

        self.conv1_ir = ConvBnLeakyRelu2d(1, 16)
        self.conv2_ir = ConvBnLeakyRelu2d(16, 32)
        self.conv3_ir = ConvBnLeakyRelu2d(32, 64)
        self.conv4_ir = ConvBnLeakyRelu2d(64, 128)

        self.decode1 = ConvBnLeakyRelu2d(128, 64)
        self.decode2 = ConvBnLeakyRelu2d(64, 32)
        self.decode3 = ConvBnLeakyRelu2d(32, 16)
        self.decode4 = ConvBnTanh2d(16, 1)
        #-----------门控生成部分------------------
        self.gate_layer = GatedLayer()
        #--------------特征融合代码部分---------------#
        self.fusion = fusion_module(dim=128)

    def forward(self, vi, ir, feature):
        # print(y_feature.shape)
        # ----------------预处理的部分，生成参数--------------#
        transmission, A = self.dehaze_net(vi)
        r = self.low_enhance_net(vi)
        # #------------------------------------------------#
        gata1, gata2 = self.gate_layer(feature)


        x = self.conv1_vi(vi)
        #print(gata1, gata2)
        x = dehaze_image(x, transmission, A) + x * (1 + gata1)
        x = low_enhance_image(x, r) + x * (1 + gata2)
        y = self.conv1_ir(ir)

        x = self.conv2_vi(x)
        x = dehaze_image(x, transmission, A) + x * (1 + gata1)
        x = low_enhance_image(x, r) + x * (1 + gata2)
        y = self.conv2_ir(y)

        x = self.conv3_vi(x)
        x = dehaze_image(x, transmission, A) + x * (1 + gata1)
        x = low_enhance_image(x, r) + x * (1 + gata2)
        y = self.conv3_ir(y)

        x = self.conv4_vi(x)
        x = dehaze_image(x, transmission, A) + x * (1 + gata1)
        x = low_enhance_image(x, r) + x * (1 + gata2)
        y = self.conv4_ir(y)

        f = self.fusion(x, y)

        f = self.decode1(f)
        f = self.decode2(f)
        f = self.decode3(f)
        f = self.decode4(f)

        return f



class fusion_module(nn.Module):
    def __init__(self, dim):
        super(fusion_module, self).__init__()
        self.conv = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        return self.conv(torch.cat([x,y],dim=1))





if __name__ == '__main__':
    model = F_Net().cuda()
    img = torch.randn(1,1,64,64).cuda()
    feature = torch.randn(1, 128, 64, 64).cuda()
    for xx in model(img, img, feature):
        print(xx.shape)




