from networks.unet import UNet
from networks.ResNet2d import Res34UNet_2d, Res18UNet_2d
from networks.vnet import VNet
from networks.ResVNet import Res34VNet, Res18VNet


def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "res34unet":
        net = Res34UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "res18unet":
        net = Res18UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "res34vnet":
        net = Res34VNet(n_channels=in_chns, n_classes=class_num).cuda()
    elif net_type == "res18vnet":
        net = Res18VNet(n_channels=in_chns, n_classes=class_num).cuda()
    elif net_type == "vnet":
        net = VNet(
            n_channels=in_chns,
            n_classes=class_num,
            normalization="batchnorm",
            has_dropout=True,
        ).cuda()
    else:
        net = None
    return net

