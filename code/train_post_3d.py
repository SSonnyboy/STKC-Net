import argparse
import logging
import os
import os.path as osp
import random
import sys
import yaml

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.dataset_3d import (
    LAHeart,
    Pancreas,
    WeakStrongAugment,
    TwoStreamBatchSampler,
)
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.util import update_values, time_str, AverageMeter
from utils.ema import ModelEMA
from val_3D import var_all_case_LA, var_all_case_Pancrease
from utils.nms import *
from utils.mixaugs import *
from CR import *


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        I. helpers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args["consistency"] * ramps.sigmoid_rampup(epoch, args["consistency_rampup"])


def get_rampup_param(iter, max_iter):
    mu_linear = 2 * iter / max_iter - 1
    b_linear = 1 - iter / max_iter
    return mu_linear, b_linear

def update_ema_variables(model, ema_model, alpha, global_step, args):
    # adjust the momentum param
    if global_step < args["consistency_rampup"]:
        alpha = 0.0
    else:
        alpha = min(1 - 1 / (global_step - args["consistency_rampup"] + 1), alpha)

    # update weights
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    # update buffers
    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.data = buffer_eval.data * alpha + buffer_train.data * (1 - alpha)




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        II. trainer
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def train(args, snapshot_path):
    model_t1, model_t2 = args["model1"], args["model2"]
    base_lr = args["base_lr"]
    batch_size = args["batch_size"]
    max_iterations = args["max_iterations"]
    num_classes = args["num_classes"]
    cur_time = time_str()
    writer = SummaryWriter(snapshot_path + "/log")
    csv_train = os.path.join(
        snapshot_path, "log", "seg_{}_train_iter.csv".format(cur_time)
    )
    csv_test = os.path.join(
        snapshot_path, "log", "seg_{}_validate_ep.csv".format(cur_time)
    )

    def worker_init_fn(worker_id):
        random.seed(args["seed"] + worker_id)

    # + + + + + + + + + + + #
    # 1. create model
    # + + + + + + + + + + + #
    model1 = net_factory(net_type=model_t1, in_chns=1, class_num=num_classes)
    model2 = net_factory(net_type=model_t2, in_chns=1, class_num=num_classes)
    model1.cuda()
    model2.cuda()
    model1.train()
    model2.train()

    # + + + + + + + + + + + #
    # 2. dataset
    # + + + + + + + + + + + #
    fdloader = LAHeart
    flag_pancreas = True if "pancreas" in args["root_path"].lower() else False
    if flag_pancreas:
        fdloader = Pancreas
    db_train = fdloader(
        base_dir=args["root_path"],
        split="train",
        num=None,
        transform=transforms.Compose(
            [WeakStrongAugment(args["patch_size"], flag_rot=not flag_pancreas)]
        ),
    )

    labeled_idxs = list(range(0, args["labeled_num"]))
    unlabeled_idxs = list(range(args["labeled_num"], args["max_samples"]))

    batch_sampler = TwoStreamBatchSampler(
        unlabeled_idxs, labeled_idxs, batch_size, args["labeled_bs"]
    )

    # + + + + + + + + + + + #
    # 3. dataloader
    # + + + + + + + + + + + #
    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=args["workers"],
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    logging.info("{} iterations per epoch".format(len(trainloader)))

    # + + + + + + + + + + + #
    # 4. optim, scheduler
    # + + + + + + + + + + + #
    optimizer1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model1.parameters()),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    )
    optimizer2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model2.parameters()),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    )
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    sac_loss = SAC_3d()
    rsc_loss = RSC()
    efc_loss = EFC_3d()
    # + + + + + + + + + + + #
    # 5. training loop
    # + + + + + + + + + + + #
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance_1 = 0.0
    best_performance_2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        # metric indicators
        meter_sup_losses1 = AverageMeter()
        meter_uns_losses1 = AverageMeter(20)
        meter_train_losses1 = AverageMeter(20)
        meter_sup_losses2 = AverageMeter()
        meter_uns_losses2 = AverageMeter(20)
        meter_train_losses2 = AverageMeter(20)
        meter_learning_rates = AverageMeter()

        for i_batch, sampled_batch in enumerate(trainloader):
            num_lb = args["labeled_bs"]
            num_ulb = batch_size - num_lb

            # 1) get augmented data
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )
            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )
            img_lb_s, img_lb_w, target_lb = (
                strong_batch[num_ulb:],
                weak_batch[num_ulb:],
                label_batch[num_ulb:],
            )
            img_ulb_w, img_ulb_s = weak_batch[:num_ulb], strong_batch[:num_ulb]

            mode = "pan" if flag_pancreas else "la"
            get_masks = get_LA_masks if mode == "la" else get_pan_masks
            cm_flag = random.random() < args["p_cm"] 

            if cm_flag:
                with torch.no_grad():
                    pred1_ulb, pred2_ulb = torch.softmax(
                        model1(img_ulb_w), dim=1
                    ), torch.softmax(model2(img_ulb_w), dim=1)
                    target_ulb = 0.5 * (pred1_ulb + pred2_ulb)
                    target_ulb = get_masks(target_ulb)
                img_ulb_a, target_ulb_a, img_ulb_b, target_ulb_b = get_BCP_stream(
                    img_lb_w, target_lb, img_ulb_w, target_ulb, mode=mode
                )
            if random.random() < 0.5:  # for img_lb_w
                img_lb_w, target_lb = cut_mix(img_lb_w, target_lb, mode=mode)
                img_lb_s = img_lb_w
            # 4) forward
            if not cm_flag:
                img1 = torch.cat((img_lb_w, img_ulb_w))
                img2 = torch.cat((img_lb_s, img_ulb_s))
            else:
                img1 = torch.cat((img_lb_w, img_ulb_a))
                img2 = torch.cat((img_lb_s, img_ulb_b))
            pred1, features_1 = model1(img1, is_feature=True)
            pred_lb1 = pred1[: args["labeled_bs"]]
            pred_ulb1 = pred1[args["labeled_bs"] :]

            pred2, features_2 = model2(img2, is_feature=True)
            pred_lb2 = pred2[: args["labeled_bs"]]
            pred_ulb2 = pred2[args["labeled_bs"] :]

            pred_ulb1_soft = torch.softmax(pred_ulb1, dim=1)
            pred_ulb2_soft = torch.softmax(pred_ulb2, dim=1)

            # 5) supervised loss
            loss_lb1 = (
                ce_loss(pred_lb1, target_lb.long())
                + dice_loss(
                    torch.softmax(pred_lb1, dim=1),
                    target_lb.unsqueeze(1).float(),
                )
            ) / 2.0

            loss_lb2 = (
                ce_loss(pred_lb2, target_lb.long())
                + dice_loss(
                    torch.softmax(pred_lb2, dim=1),
                    target_lb.unsqueeze(1).float(),
                )
            ) / 2.0

            # 6) unsupervised loss
            consistency_weight = get_current_consistency_weight(iter_num // 150, args)
            if cm_flag:
                # print(pred_ulb1_soft.shape, target_ulb.shape)
                # print(pred_ulb1_soft.shape, target_ulb.shape)
                if iter_num < 1000:
                    loss_ulb1, loss_ulb2 = torch.tensor(0.0), torch.tensor(0.0)
                else:
                    loss_ulb1 = dice_loss(
                        pred_ulb1_soft,
                        target_ulb_a.unsqueeze(1).float().detach(),
                    )

                    loss_ulb2 = dice_loss(
                        pred_ulb2_soft,
                        target_ulb_b.unsqueeze(1).float().detach(),
                    )
                loss1 = loss_lb1 + consistency_weight * loss_ulb1
                loss2 = loss_lb2 + consistency_weight * loss_ulb2
                loss = loss1 + loss2
                
            else:
                if iter_num < 1000:
                    loss_ulb1, loss_ulb2 = torch.tensor(0.0), torch.tensor(0.0)
                    loss1 = loss_lb1 + consistency_weight * loss_ulb1 
                    loss2 = loss_lb2 + consistency_weight * loss_ulb2

                    loss = loss1 + loss2
                else:
                    mask1 = pred_ulb1_soft.argmax(dim=1)
                    mask2 = pred_ulb2_soft.argmax(dim=1)
                    # unsup loss
                    loss_ulb1 = dice_loss(
                        pred_ulb1_soft,
                        mask2.unsqueeze(1).float().detach(),
                    )
                    loss_ulb2 = dice_loss(
                        pred_ulb2_soft,
                        mask1.unsqueeze(1).float().detach(),
                    )

                    loss_cr_sac = sac_loss(features_1, features_2)
                    loss_cr_rsc = rsc_loss(features_1, features_2)
                    loss_cr_efc = efc_loss(features_1, features_2)
                    writer.add_scalar("info/loss_sac", loss_cr_sac, iter_num)
                    writer.add_scalar("info/loss_rsc", loss_cr_rsc, iter_num)
                    writer.add_scalar("info/loss_efc", loss_cr_efc, iter_num)

                    loss1 = loss_lb1 + consistency_weight * loss_ulb1 
                    loss2 = loss_lb2 + consistency_weight * loss_ulb2

                    loss = loss1 + loss2 + args["cr_weight"] * (loss_cr_sac + loss_cr_rsc + loss_cr_efc)
                # loss = loss1 + loss2
            # 8) update student model
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            # 10) udpate learing rate
            if args["poly"]:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for optimizer in [optimizer1, optimizer2]:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_
            else:
                lr_ = base_lr

            # 11) record statistics
            iter_num = iter_num + 1
            # --- a) writer
            # writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/loss1", loss1, iter_num)
            writer.add_scalar("info/loss2", loss2, iter_num)
            writer.add_scalar("info/loss_lb1", loss_lb1, iter_num)
            writer.add_scalar("info/loss_ulb1", loss_ulb1, iter_num)
            writer.add_scalar("info/loss_lb2", loss_lb2, iter_num)
            writer.add_scalar("info/loss_ulb2", loss_ulb2, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            # --- b) loggers
            logging.info(
                "iteration:{}  t-loss1/2:{:.4f}/{:.4f}, loss-lb1/2:{:.4f}/{:.4f}, loss-ulb1/2:{:.4f}/{:.4f}, weight:{:.2f}, lr:{:.4f}".format(
                    iter_num,
                    loss1.item(),
                    loss2.item(),
                    loss_lb1.item(),
                    loss_lb2.item(),
                    loss_ulb1.item(),
                    loss_ulb2.item(),
                    consistency_weight,
                    lr_,
                )
            )
            # --- c) avg meters
            meter_sup_losses1.update(loss_lb1.item())
            meter_uns_losses1.update(loss_ulb1.item())
            meter_sup_losses2.update(loss_lb2.item())
            meter_uns_losses2.update(loss_ulb2.item())
            meter_train_losses1.update(loss1.item())
            meter_train_losses2.update(loss2.item())
            meter_learning_rates.update(lr_)

            # --- d) csv
            tmp_results = {
                "loss1": loss1.item(),
                "loss2": loss2.item(),
                "loss_lb1": loss_lb1.item(),
                "loss_lb2": loss_lb2.item(),
                "loss_ulb1": loss_ulb1.item(),
                "loss_ulb2": loss_ulb2.item(),
                "lweight_ub": consistency_weight,
                "lr": lr_,
            }
            data_frame = pd.DataFrame(
                data=tmp_results, index=range(iter_num, iter_num + 1)
            )
            if iter_num > 1 and osp.exists(csv_train):
                data_frame.to_csv(csv_train, mode="a", header=None, index_label="iter")
            else:
                data_frame.to_csv(csv_train, index_label="iter")

            if iter_num >= max_iterations:
                break

        # 12) validating
        if (
            epoch_num % args.get("test_interval_ep", 1) == 0
            or iter_num >= max_iterations
        ):
            model1.eval()
            model2.eval()

            if "pancreas" in args["root_path"].lower():
                performance_1 = var_all_case_Pancrease(
                    model1,
                    args["root_path"],
                    num_classes=num_classes,
                    patch_size=args["patch_size"],
                    stride_xy=16,
                    stride_z=16,
                    flag_nms=True,
                )
                performance_2 = var_all_case_Pancrease(
                    model2,
                    args["root_path"],
                    num_classes=num_classes,
                    patch_size=args["patch_size"],
                    stride_xy=16,
                    stride_z=16,
                    flag_nms=True,
                )
            else:
                performance_1 = var_all_case_LA(
                    model1,
                    args["root_path"],
                    num_classes=num_classes,
                    patch_size=args["patch_size"],
                    stride_xy=18,
                    stride_z=4,
                )
                performance_2 = var_all_case_LA(
                    model2,
                    args["root_path"],
                    num_classes=num_classes,
                    patch_size=args["patch_size"],
                    stride_xy=18,
                    stride_z=4,
                )

            if performance_1 > best_performance_1:
                best_performance_1 = performance_1
                tmp_model1_snapshot_path = os.path.join(snapshot_path, model_t1 + "_1")
                if not os.path.exists(tmp_model1_snapshot_path):
                    os.makedirs(tmp_model1_snapshot_path, exist_ok=True)

                save_best_path_stu = os.path.join(
                    snapshot_path, "best_{}_model1.pth".format(model_t1)
                )
                torch.save(model1.state_dict(), save_best_path_stu)

            if performance_2 > best_performance_2:
                best_performance_2 = performance_2
                tmp_model2_snapshot_path = os.path.join(snapshot_path, model_t2 + "_2")
                if not os.path.exists(tmp_model2_snapshot_path):
                    os.makedirs(tmp_model2_snapshot_path, exist_ok=True)
                save_best_path = os.path.join(
                    snapshot_path, "best_{}_model2.pth".format(model_t2)
                )
                torch.save(model2.state_dict(), save_best_path)

            # writer
            writer.add_scalar("Var_dice/Dice_1", performance_1, epoch_num)
            writer.add_scalar("Var_dice/Best_dice_1", best_performance_1, epoch_num)
            writer.add_scalar("Var_dice/Dice_2", performance_2, epoch_num)
            writer.add_scalar("Var_dice/Best_dice_2", best_performance_2, epoch_num)

            # csv
            tmp_results_ts = {
                "loss_total1": meter_train_losses1.avg,
                "loss_total2": meter_train_losses2.avg,
                "loss_sup1": meter_sup_losses1.avg,
                "loss_unsup1": meter_uns_losses1.avg,
                "loss_sup2": meter_sup_losses2.avg,
                "loss_unsup2": meter_uns_losses2.avg,
                "learning_rate": meter_learning_rates.avg,
                "Dice_1": performance_1,
                "Dice_1_best": best_performance_1,
                "Dice_2": performance_2,
                "Dice_2_best": best_performance_2,
            }
            data_frame = pd.DataFrame(
                data=tmp_results_ts, index=range(epoch_num, epoch_num + 1)
            )
            if epoch_num > 0 and osp.exists(csv_test):
                data_frame.to_csv(csv_test, mode="a", header=None, index_label="epoch")
            else:
                data_frame.to_csv(csv_test, index_label="epoch")

            # logs
            logging.info(
                " <<Test>> - Ep:{}  - Dice-1/2:{:.2f}/{:.2f}, Best-1:{:.2f}, Best-2:{:.2f}".format(
                    epoch_num,
                    performance_1 * 100,
                    performance_2 * 100,
                    best_performance_1 * 100,
                    best_performance_2 * 100,
                )
            )
            logging.info(
                "          - AvgLoss1(lb/ulb/all):{:.4f}/{:.4f}/{:.4f}- AvgLoss2(lb/ulb/all):{:.4f}/{:.4f}/{:.4f}".format(
                    meter_sup_losses1.avg,
                    meter_uns_losses1.avg,
                    meter_train_losses1.avg,
                    meter_sup_losses2.avg,
                    meter_uns_losses2.avg,
                    meter_train_losses2.avg,
                )
            )

            model1.train()
            model2.train()

        if (epoch_num + 1) % args.get("save_interval_epoch", 1000000) == 0:
            save_mode_path = os.path.join(
                snapshot_path, "epoch_" + str(epoch_num) + ".pth"
            )
            torch.save(model1.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if iter_num >= max_iterations:
            iterator.close()
            break
    save_mode_path = os.path.join(snapshot_path, "model1_last.pth")
    torch.save(model1.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))

    save_mode_path = os.path.join(snapshot_path, "model2_last.pth")
    torch.save(model2.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))


    writer.close()
    return "Training Finished!"


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        III. main process
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
if __name__ == "__main__":
    # 1. set up config
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="", help="configuration file")

    # Basics: Data, results, model
    parser.add_argument(
        "--root_path", type=str, default="./data/LA", help="Name of Experiment"
    )
    parser.add_argument(
        "--res_path", type=str, default="./results/LA", help="Path to save resutls"
    )
    parser.add_argument("--exp", type=str, default="LA/POST", help="experiment_name")
    parser.add_argument("--model1", type=str, default="res18vnet")
    parser.add_argument("--model2", type=str, default="res34vnet")
    parser.add_argument(
        "--num_classes", type=int, default=2, help="output channel of network"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="the id of gpu used to train the model"
    )

    # Training Basics
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=15000,
        help="maximum epoch number to train",
    )
    parser.add_argument(
        "--base_lr", type=float, default=0.01, help="segmentation network learning rate"
    )
    # https://blog.csdn.net/qq_43391414/article/details/122992458
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs="+",
        default=[112, 112, 80],
        help="patch size of network input",
    )

    parser.add_argument(
        "--max_samples", type=int, default=80, help="maximum samples to train"
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="whether use deterministic training",
    )
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--workers", type=int, default=4, help="number of workers")
    parser.add_argument("--test_interval_iter", type=int, default=200, help="")
    parser.add_argument("--test_interval_ep", type=int, default=1, help="")
    parser.add_argument("--save_interval_epoch", type=int, default=1000000, help="")
    parser.add_argument(
        "-p",
        "--poly",
        default=False,
        action="store_true",
        help="whether poly scheduler",
    )

    # label and unlabel
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
    parser.add_argument(
        "--labeled_bs", type=int, default=2, help="labeled_batch_size per gpu"
    )
    parser.add_argument("--labeled_num", type=int, default=4, help="labeled data")

    # model related
    parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")

    # unlabeled loss
    parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
    parser.add_argument(
        "--consistency_rampup", type=float, default=40.0, help="consistency_rampup"
    )
    parser.add_argument("--cr_weight", type=float, default=0.01, help="cr_weight")
    parser.add_argument("--p_cm", type=float, default=0.5, help="p_cm")
    # parse args
    args = parser.parse_args()
    args = vars(args)

    # 2. update from the config files
    cfgs_file = args["cfg"]
    cfgs_file = os.path.join("./cfgs", cfgs_file)
    with open(cfgs_file, "r") as handle:
        options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    # convert "1e-x" to float
    for each in options_yaml.keys():
        tmp_var = options_yaml[each]
        if type(tmp_var) == str and "1e-" in tmp_var:
            options_yaml[each] = float(tmp_var)
    # update original parameters of argparse
    update_values(options_yaml, args)
    import pprint

    # 3. setup gpus and randomness
    # if args["gpu_id"] in range(8):
    if args["gpu_id"] in range(10):
        gid = args["gpu_id"]
    else:
        gid = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)

    if not args["deterministic"]:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args["seed"] > 0:
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])

    # 4. outputs and logger
    # 4. outputs and logger
    snapshot_path = "{}/{}_{}_labeled/{}".format(
        args["res_path"],
        args["exp"],
        args["labeled_num"],
        args["model1"] + "_" + args["model2"],
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{}".format(pprint.pformat(args)))

    train(args, snapshot_path)
