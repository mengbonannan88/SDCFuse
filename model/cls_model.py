import torch
import torch.nn as nn
import clip  # 引入 CLIP 模型库

class Illumination_classifier(nn.Module):
    def __init__(self, input_channels=3, num_classes=9, pretrained=True):
        super(Illumination_classifier, self).__init__()

        # 加载 CLIP 模型
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

        # 如果输入不是RGB的3通道，则需要额外卷积层调整输入
        if input_channels != 3:
            self.input_adjust = nn.Conv2d(input_channels, 3, kernel_size=1)

        # 冻结CLIP的参数，不进行训练
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 将 CLIP 的图像编码器作为特征提取器，输出的特征向量维度为512
        self.fc = nn.Linear(512, 128)  # 映射到128维度

        # 自定义线性层用于分类
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 如果输入不是3通道，则调整通道数
        if hasattr(self, 'input_adjust'):
            x = self.input_adjust(x)

        # 将输入转换为 float32 以兼容 CLIP
        x = x.to(torch.float32)

        # 使用 CLIP 的图像编码器提取特征
        with torch.no_grad():
            x = self.clip_model.encode_image(x)

        # 将 CLIP 提取到的特征转换为 float32，确保与 fc 层参数匹配
        x = x.to(torch.float32)

        # 保存提取到的特征
        feature = x

        # 经过自定义的线性层进行分类
        x = self.fc(x)
        x = self.linear1(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.linear2(x)

        # 最终输出带有ReLU激活函数的分类结果
        x = nn.ReLU()(x)

        return x, feature


