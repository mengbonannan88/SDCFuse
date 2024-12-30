import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import torch.nn.functional as F
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloder.data_loder import llvip
from model.common import clamp, gradient
from model.model import F_Net
from model.cls_model import Illumination_classifier


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

import torch.nn as nn
class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return (self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size).mean()





if __name__ == '__main__':
    init_seeds(2024)
    datasets = ''
    save_path = 'runs/'
    cls_model = Illumination_classifier(input_channels=3).cuda()  # 先将模型移到cuda:0
    # 加载多卡模型权重，移除 'module.' 前缀
    state_dict = torch.load('runs/best_cls.pth')

    # 如果是 DataParallel 模型，权重前缀带有 'module.'，需要移除
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 去除 'module.' 前缀
        name = k.replace('module.', '')
        new_state_dict[name] = v

    # 将修改后的 state_dict 加载到单卡模型
    cls_model.load_state_dict(new_state_dict)
    cls_model.eval()
    batch_size = 2
    num_works = 1
    lr = 0.0001
    Epoch = 30


    tv_loss = L_TV()

    train_dataset = llvip(datasets)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_works, pin_memory=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = torch.nn.DataParallel(F_Net()).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(Epoch):
        train_tqdm = tqdm(train_loader, total=len(train_loader), ascii=True)
        for vis_rain_y, vis_gt_y, inf_image, vis_cb_image, vis_cr_image, name, vis_rain, vis_clip in train_tqdm:
            optimizer.zero_grad()
            vis_rain = vis_rain.cuda()

            vis_rain_y = vis_rain_y.cuda()
            vis_gt_y = vis_gt_y.cuda()
            inf_image = inf_image.cuda()

            vis_clip = vis_clip.cuda()


            # Mixed precision autocast
            with torch.cuda.amp.autocast():
                #print(vis_clip.shape)
                _, feature = cls_model(vis_clip)

                fused = model(vis_rain_y, inf_image, feature)

                # Loss functions
                loss_f = 50 * F.l1_loss(gradient(fused), torch.max(gradient(vis_gt_y), gradient(inf_image))) + 40 * F.l1_loss(fused, torch.max(vis_gt_y, inf_image))
                loss = loss_f

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            # **梯度裁剪**
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            ##### Display loss
            train_tqdm.set_postfix(epoch=epoch,
                                   loss=loss.item(),
                                   loss_f=loss_f.item()
                                   )

        #### Save the trained model
        torch.save(model.state_dict(), f'{save_path}/model_{epoch}.pth')
