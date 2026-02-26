#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   test.py
@Time    :   2025/03/22 13:44:48
@Author  :   biabuluo
@Version :   1.0
@Desc    :   None
"""

from networks.net_factory import *
import utils.losses as losses

dice_loss = losses.DiceLoss(4)


import torch

# x = torch.rand((6, 4, 256, 256))
# img_l = torch.ones_like(torch.rand((6, 1, 256, 256)))
# mask = torch.rand((6, 1, 256, 256))
# print(
#     dice_loss(x, img_l, mask).shape,
# )
# model = net_factory("res34vnet")
# x = torch.rand((4, 1, 112, 112, 80)).cuda()
# out, g1, g2, g3, g4 = model(x, is_feature=True)
# print(out.shape, g1.shape, g2.shape, g3.shape, g4.shape)

# out = model(x)
# print(x.shape)

model = net_factory("unet")
x = torch.rand((4, 1, 256, 256)).cuda()
out, features = model(x, is_feature=True)
print(out.shape)
for i in features:
    print(i.shape)
