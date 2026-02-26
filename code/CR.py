import torch
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn

class SAC(nn.Module):
    def __init__(self, p=2):
        super(SAC, self).__init__()
        self.p = p

    def forward(self, xs, ys):
        assert len(xs) == len(ys), "Input feature list lengths mismatch."
        losses = [self.at_loss(x, y) for x, y in zip(xs, ys)]
        return sum(losses) / len(losses)

    def at_loss(self, fm_s, fm_t):
        am_s = self.attention_map(fm_s)
        am_t = self.attention_map(fm_t) 
        loss = F.mse_loss(am_s, am_t)

        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)
        return am


class RSC(nn.Module):
    def __init__(self):
        super(RSC, self).__init__()

    def forward(self, fml_s, fml_t):
        assert len(fml_s) == len(fml_t), "Input feature list lengths mismatch."

        sp_loss_all = torch.tensor(0.0, device=fml_s[0].device)
        for fm_s, fm_t in zip(fml_s, fml_t):
            sp_loss_all += self.sp_loss(fm_s, fm_t)
        return sp_loss_all / len(fml_s)

    def sp_loss(self, fm_s, fm_t):
        fm_s = F.normalize(fm_s.view(fm_s.size(0), -1), p=2, dim=1)  
        fm_t = F.normalize(fm_t.view(fm_t.size(0), -1), p=2, dim=1)
        G_s = torch.mm(fm_s, fm_s.t())  
        G_t = torch.mm(fm_t, fm_t.t())
        return F.mse_loss(G_s, G_t)

class EFC(nn.Module):
    def __init__(self):
        super(EFC, self).__init__()

    def forward(self, g_s, g_t):
        assert len(g_s) == len(g_t), "unequal length of feat list"
        s_fsp = self.compute_fsp(g_s)
        t_fsp = self.compute_fsp(g_t)
        loss_group = [self.compute_loss(s, t) for s, t in zip(s_fsp, t_fsp)]
        return sum(loss_group) / len(loss_group)

    @staticmethod
    def compute_loss(s, t):
        return (s - t).pow(2).mean()

    @staticmethod
    def compute_fsp(g):
        fsp_list = []
        for i in range(len(g) - 1):
            bot, top = g[i], g[i + 1]
            b_H, t_H = bot.shape[2], top.shape[2]
            if b_H > t_H:
                bot = F.adaptive_avg_pool2d(bot, (t_H, t_H))
            elif b_H < t_H:
                top = F.adaptive_avg_pool2d(top, (b_H, b_H))
            else:
                pass
            bot = bot.unsqueeze(1)
            top = top.unsqueeze(2)
            bot = bot.view(bot.shape[0], bot.shape[1], bot.shape[2], -1)
            top = top.view(top.shape[0], top.shape[1], top.shape[2], -1)

            fsp = (bot * top).mean(-1)
            fsp_list.append(fsp)
        return fsp_list


class EFC_3d(nn.Module):
    def __init__(self):
        super(EFC_3d, self).__init__()

    def forward(self, g_s, g_t):
        assert len(g_s) == len(g_t), "unequal length of feat list"
        s_fsp = self.compute_fsp(g_s)
        t_fsp = self.compute_fsp(g_t)
        loss_group = [self.compute_loss(s, t) for s, t in zip(s_fsp, t_fsp)]
        return sum(loss_group) / len(loss_group)

    @staticmethod
    def compute_loss(s, t):
        return (s - t).pow(2).mean()

    @staticmethod
    def compute_fsp(g):
        fsp_list = []
        for i in range(len(g) - 1):
            bot, top = g[i], g[i + 1]
            b_H, t_H = bot.shape[2], top.shape[2]
            b_D, t_D = bot.shape[4], top.shape[4]
            # print(bot.shape, top.shape)
            if b_H > t_H: # [4, 1, 112, 112, 80]
                bot = F.adaptive_avg_pool3d(bot, (t_H, t_H, t_D))
            elif b_H < t_H:
                top = F.adaptive_avg_pool3d(top, (b_H, b_H, b_D))
            else:
                pass
            # print(bot.shape, top.shape)
            bot = bot.unsqueeze(1)
            top = top.unsqueeze(2)
            bot = bot.view(bot.shape[0], bot.shape[1], bot.shape[2], -1)
            top = top.view(top.shape[0], top.shape[1], top.shape[2], -1)

            fsp = (bot * top).mean(-1)
            fsp_list.append(fsp)
        return fsp_list


class SAC_3d(nn.Module):
    def __init__(self, p=2):
        super(SAC_3d, self).__init__()
        self.p = p

    def forward(self, xs, ys):
        assert len(xs) == len(ys), "Input feature list lengths mismatch."
        losses = [self.at_loss(x, y) for x, y in zip(xs, ys)]
        return sum(losses) / len(losses)

    def at_loss(self, fm_s, fm_t):
        am_s = self.attention_map(fm_s)
        am_t = self.attention_map(fm_t)  
        loss = F.mse_loss(am_s, am_t)

        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)          # [B, C, H, W/D]
        am = torch.sum(am, dim=1, keepdim=True)        # [B, 1, H, W/D]
        spatial_dims = (2, 3, 4)
        norm = torch.sqrt(torch.sum(am ** 2, dim=spatial_dims, keepdim=True) + eps)
        am = am / norm
        return am
