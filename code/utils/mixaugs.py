import numpy as np
import random
import torch
import scipy.stats as stats


def generate_mask_acdc(img):
    batch_size, channel, img_x, img_y = (
        img.shape[0],
        img.shape[1],
        img.shape[2],
        img.shape[3],
    )
    # loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w : w + patch_x, h : h + patch_y] = 0
    return mask.long()


def generate_mask_la(img):
    batch_size, channel, img_x, img_y, img_z = (
        img.shape[0],
        img.shape[1],
        img.shape[2],
        img.shape[3],
        img.shape[4],
    )
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = (
        int(img_x * 2 / 3),
        int(img_y * 2 / 3),
        int(img_z * 2 / 3),
    )
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w : w + patch_pixel_x, h : h + patch_pixel_y, z : z + patch_pixel_z] = 0
    return mask.long()


def generate_mask_pan(img):
    mask = torch.ones(96, 96, 96).cuda()
    w = np.random.randint(0, 96 - 64)
    h = np.random.randint(0, 96 - 64)
    z = np.random.randint(0, 96 - 64)
    mask[w : w + 64, h : h + 64, z : z + 64] = 0
    return mask.long()


def generate_mask(img, mode="acdc"):
    if mode == "acdc":
        return generate_mask_acdc(img)
    elif mode == "la":
        return generate_mask_la(img)
    else:
        return generate_mask_pan(img)


# # # # # # # # # # # # # # # # # # # # #
# # 1. cutmix for single batch img
# # # # # # # # # # # # # # # # # # # # #


def cut_mix_single(image, mask):
    img_a, img_b = image.chunk(2)
    img_in = img_a * mask + img_b * (1 - mask)
    img_out = img_a * (1 - mask) + img_b * mask
    img_cm = torch.cat([img_in, img_out], dim=0)
    return img_cm


# # # # # # # # # # # # # # # # # # # # #
# # 1. cutmix
# # # # # # # # # # # # # # # # # # # # #


def cut_mix(img_w, target, mode="acdc"):
    img_a, img_b = img_w.chunk(2)
    target_a, target_b = target.chunk(2)
    mask = generate_mask(img_a, mode)
    img_in = img_a * mask + img_b * (1 - mask)
    img_out = img_a * (1 - mask) + img_b * mask
    target_in = target_a * mask + target_b * (1 - mask)
    target_out = target_a * (1 - mask) + target_b * mask
    return torch.cat([img_in, img_out]), torch.cat([target_in, target_out])


def get_cm_stream(img_w, img_s, target, mode="acdc"):
    img_w_a, img_w_b = img_w.chunk(2)
    img_s_a, img_s_b = img_s.chunk(2)
    target_a, target_b = target.chunk(2)
    mask = generate_mask(img_w_a, mode)
    img_in = img_w_a * mask + img_s_b * (1 - mask)
    img_out = img_s_a * (1 - mask) + img_w_b * mask
    target_in = target_a * mask + target_b * (1 - mask)
    target_out = target_a * (1 - mask) + target_b * mask
    if random.random() < 0.5:
        return img_in, target_in, img_out, target_out
    else:
        return img_out, target_out, img_in, target_in


def get_BCP_stream(img_lb, target, img_ulb, pred, mode="acdc"):
    img_a, img_b = img_lb.chunk(2)
    target_a, target_b = target.chunk(2)
    img_a_ul, img_b_ul = img_ulb.chunk(2)
    pred_a, pred_b = pred.chunk(2)
    mask = generate_mask(img_a, mode)
    img_in_1 = img_a * mask + img_a_ul * (1 - mask)
    img_in_2 = img_a * mask + img_b_ul * (1 - mask)

    img_out_1 = img_b * (1 - mask) + img_b_ul * mask
    img_out_2 = img_b * (1 - mask) + img_a_ul * mask

    target_in_1 = target_a * mask + pred_a * (1 - mask)
    target_in_2 = target_a * mask + pred_b * (1 - mask)

    target_out_1 = target_b * (1 - mask) + pred_b * mask
    target_out_2 = target_b * (1 - mask) + pred_a * mask
    img_in = torch.cat([img_in_1, img_in_2])
    img_out = torch.cat([img_out_1, img_out_2])
    target_in = torch.cat([target_in_1, target_in_2])
    target_out = torch.cat([target_out_1, target_out_2])
    if random.random() < 0.5:
        return img_in, target_in, img_out, target_out
    else:
        return img_out, target_out, img_in, target_in


def get_random_stream(img_lb, target, img_ulb_w, img_ulb_s, pred, mode="acdc"):
    img_ulb_a1, target_ulb_a1, img_ulb_b1, target_ulb_b1 = get_BCP_stream(
        img_lb, target, img_ulb_w, pred, mode
    )
    img_ulb_a2, target_ulb_a2, img_ulb_b2, target_ulb_b2 = get_cm_stream(
        img_ulb_w, img_ulb_s, pred, mode
    )
    img_ulb_a = torch.cat([img_ulb_a1, img_ulb_a2], dim=0)
    target_ulb_a = torch.cat([target_ulb_a1, target_ulb_a2], dim=0)
    img_ulb_b = torch.cat([img_ulb_b1, img_ulb_b2], dim=0)
    target_ulb_b = torch.cat([target_ulb_b1, target_ulb_b2], dim=0)

    return img_ulb_a, target_ulb_a, img_ulb_b, target_ulb_b
