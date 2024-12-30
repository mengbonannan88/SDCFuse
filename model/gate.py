import torch
import torch.nn as nn

class GatedLayer(nn.Module):
    def __init__(self):
        super(GatedLayer, self).__init__()
        # 使用全局平均池化将特征缩小到 1x1
        self.global_avg_pool = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
        )
        # 用于生成每个门控的线性层
        self.fc1 = nn.Linear(128, 1)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        # 全局平均池化，将输入特征缩小为 (b, 128, 1, 1)
        #pooled = self.global_avg_pool(x).view(x.size(0), -1)  # 变为 (b, 128)
        pooled = self.global_avg_pool(x)
        #print(pooled.shape)
        # 通过线性层生成门控权重
        gate1 = (self.fc1(pooled))  # 输出 (b, 1)
        gate2 = (self.fc2(pooled))  # 输出 (b, 1)


        gate1 = gate1.view(-1, 1, 1, 1)  # 输出形状 (b, 1, 1, 1)
        gate2 = gate2.view(-1, 1, 1, 1)  # 输出形状 (b, 1, 1, 1)

        return gate1, gate2


# 测试代码
if __name__ == "__main__":
    # 假设输入 tensor 为 (b, 128, h, w)
    x = torch.randn(4, 128, 32, 32)  # batch size = 4, height = width = 32

    # 实例化模型
    model = GatedLayer()

    # 生成门控
    gate1, gate2, gate3 = model(x)

    # 打印门控的形状和值
    print("Gate 1 shape:", gate1.shape, gate1)
    print("Gate 2 shape:", gate2.shape, gate2)
    print("Gate 3 shape:", gate3.shape, gate3)
