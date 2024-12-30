"""测试融合网络"""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloder.data_loder_test import llvip
from model.model import F_Net
from model.common import clamp, YCrCb2RGB
from model.cls_model import Illumination_classifier
import time






if __name__ == '__main__':
    num_works = 1
    test_data = 'test/low_light_llvip'
    print(test_data)
    fusion_result_path = test_data + '/ours'

    if not os.path.exists(fusion_result_path):
        os.makedirs(fusion_result_path)
    #----------------加载基于clip的分类模型-----------------------#
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
    #-------------------------------------------------#


    test_dataset = llvip(test_data)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_works, pin_memory=True)

    if not os.path.exists(fusion_result_path):
        os.makedirs(fusion_result_path)

    #######加载模型
    model = torch.nn.DataParallel(F_Net()).cuda()
    model.load_state_dict(torch.load('runs/model_29.pth'))
    model.eval()



    ##########加载数据
    test_tqdm = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for vis_rain_y, vis_cb_image, vis_cr_image, inf_image, vis_clip_image, name in test_tqdm:
            vis_rain_y=vis_rain_y.cuda()
            cb = vis_cb_image.cuda()
            cr = vis_cr_image.cuda()
            inf_image = inf_image.cuda()
            vis_clip = vis_clip_image.cuda()

            # Mixed precision autocast
            with torch.cuda.amp.autocast():
                # print(vis_clip.shape)
                _, feature = cls_model(vis_clip)

                fused = model(vis_rain_y, inf_image, feature)
            fused = clamp(fused)

            rgb_fused_image = YCrCb2RGB(fused[0], cb[0], cr[0])
            rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
            rgb_fused_image.save(f'{fusion_result_path}/{name[0]}')
