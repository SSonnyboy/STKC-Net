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
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.mixaugs import *
from utils.nms import *

from dataloaders.dataset_2d import (
    BaseDataSets,
    TwoStreamBatchSampler,
    WeakStrongAugment,
)
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.util import update_values, time_str, AverageMeter
from utils.ema import ModelEMA
from val_2D import test_single_volume
from CR import *


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        I. helpers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {
            "3": 68,
            "7": 136,
            "14": 256,
            "21": 396,
            "28": 512,
            "35": 664,
            "140": 1312,
        }
    elif "Prostate":
        ref_dict = {
            "2": 27,
            "4": 53,
            "8": 120,
            "12": 179,
            "16": 256,
            "21": 312,
            "42": 623,
        }
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args["consistency"] * ramps.sigmoid_rampup(epoch, args["consistency_rampup"])


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
    num_classes = args["num_classes"]
    batch_size = args["batch_size"]
    max_iterations = args["max_iterations"]
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
    db_train = BaseDataSets(
        base_dir=args["root_path"],
        split="train",
        num=None,
        transform=transforms.Compose([WeakStrongAugment(args["patch_size"])]),
    )
    db_val = BaseDataSets(base_dir=args["root_path"], split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args["root_path"], args["labeled_num"])
    logging.info(
        "Total silices is: {}, labeled slices is: {}".format(
            total_slices, labeled_slice
        )
    )
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(
        unlabeled_idxs, labeled_idxs, batch_size, args["labeled_bs"]
    )

    # + + + + + + + + + + + #
    # 3. dataloader
    # + + + + + + + + + + + #
    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
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
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    sac_loss = SAC()
    rsc_loss = RSC()
    efc_loss = EFC()
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

            # get batched data
            img_lb_s, img_lb_w, target_lb = (
                strong_batch[num_ulb:],
                weak_batch[num_ulb:],
                label_batch[num_ulb:],
            )
            img_ulb_w, img_ulb_s = weak_batch[:num_ulb], strong_batch[:num_ulb]

            cm_flag = random.random() < args["p_cm"]

            if cm_flag:
                with torch.no_grad():
                    pred1_ulb, pred2_ulb = torch.softmax(
                        model1(img_ulb_w), dim=1
                    ), torch.softmax(model2(img_ulb_w), dim=1)
                target_ulb = 0.5 * (pred1_ulb + pred2_ulb)
                target_ulb = get_ACDC_masks(target_ulb)
                img_ulb_a, target_ulb_a, img_ulb_b, target_ulb_b = get_BCP_stream(
                    img_lb_w, target_lb, img_ulb_w, target_ulb, mode="acdc"
                )

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
            writer.add_scalar("info/lr", lr_, iter_num)
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
            epoch_num % args.get("test_interval_ep", 2) == 0
            or iter_num >= max_iterations
        ):
            model1.eval()
            model2.eval()

            metric_list1 = 0.0
            metric_list2 = 0.0

            for _, sampled_batch in enumerate(valloader):
                metric_i1 = test_single_volume(
                    sampled_batch["image"],
                    sampled_batch["label"],
                    # ema_model1,
                    model1,
                    classes=num_classes,
                )
                metric_list1 += np.array(metric_i1)

                metric_i2 = test_single_volume(
                    sampled_batch["image"],
                    sampled_batch["label"],
                    # ema_model2,
                    model2,
                    classes=num_classes,
                )
                metric_list2 += np.array(metric_i2)

            metric_list1 = metric_list1 / len(db_val)
            metric_list2 = metric_list2 / len(db_val)

            for class_i in range(num_classes - 1):
                writer.add_scalar(
                    "info/1val_{}_dice".format(class_i + 1),
                    metric_list1[class_i, 0],
                    epoch_num,
                )
                writer.add_scalar(
                    "info/1val_{}_hd95".format(class_i + 1),
                    metric_list1[class_i, 1],
                    epoch_num,
                )

                writer.add_scalar(
                    "info/2val_{}_dice".format(class_i + 1),
                    metric_list2[class_i, 0],
                    epoch_num,
                )
                writer.add_scalar(
                    "info/2val_{}_hd95".format(class_i + 1),
                    metric_list2[class_i, 1],
                    epoch_num,
                )

            performance_1 = np.mean(metric_list1, axis=0)[0]
            mean_hd95_1 = np.mean(metric_list1, axis=0)[1]
            writer.add_scalar("info/val_mean_dice_1", performance_1, epoch_num)
            writer.add_scalar("info/val_mean_hd95_1", mean_hd95_1, epoch_num)

            performance_2 = np.mean(metric_list2, axis=0)[0]
            mean_hd95_2 = np.mean(metric_list2, axis=0)[1]
            writer.add_scalar("info/val_mean_dice_2", performance_2, epoch_num)
            writer.add_scalar("info/val_mean_hd95_2", mean_hd95_2, epoch_num)

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
                " <<Test>> - Ep:{}  - mean_dice/mean_h95 - M1:{:.2f}/{:.2f}, Best-1:{:.2f}, M2:{:.2f}/{:.2f}, Best-2:{:.2f}".format(
                    epoch_num,
                    performance_1 * 100,
                    mean_hd95_1,
                    best_performance_1 * 100,
                    performance_2 * 100,
                    mean_hd95_2,
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
        "--root_path", type=str, default="./data/ACDC", help="Name of Experiment"
    )
    parser.add_argument(
        "--res_path", type=str, default="./results/ACDC", help="Path to save resutls"
    )
    parser.add_argument(
        "--exp", type=str, default="ACDC/POST-NoT", help="experiment_name"
    )
    parser.add_argument(
        "--num_classes", type=int, default=4, help="output channel of network"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="the id of gpu used to train the model"
    )

    # Training Basics
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=30000,
        help="maximum epoch number to train",
    )
    parser.add_argument(
        "--base_lr", type=float, default=0.01, help="segmentation network learning rate"
    )
    parser.add_argument(
        "--patch_size",
        type=list,
        default=[256, 256],
        help="patch size of network input",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether use deterministic training",
    )
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
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
    parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
    parser.add_argument(
        "--labeled_bs", type=int, default=12, help="labeled_batch_size per gpu"
    )
    parser.add_argument("--labeled_num", type=int, default=136, help="labeled data")

    # unlabeled loss
    parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
    parser.add_argument(
        "--consistency_rampup", type=float, default=150.0, help="consistency_rampup"
    )
    parser.add_argument("--model1", type=str, default="unet")
    parser.add_argument("--model2", type=str, default="resunet")

    parser.add_argument("--cr_weight", type=float, default=0.01, help="cr_weight")
    parser.add_argument("--p_cm", type=float, default=0.3, help="p_cm")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
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
    # print confg information
    import pprint

    # print("{}".format(pprint.pformat(args)))
    # assert 1==0, "break here"

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
    # logging.info(str(args))
    logging.info("{}".format(pprint.pformat(args)))

    train(args, snapshot_path)
