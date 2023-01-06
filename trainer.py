# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import sys
sys.path.insert(0,'/tmp/packages/')
# sys.path.remove('/usr/local/lib/python2.7/site-packages')
import argparse
import datetime
import json
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from dataset import *
import timm
from torch.utils.tensorboard import SummaryWriter

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Dataloader arguments
    parser.add_argument("--data", type=str, default="complete",
                        help="Label data type.")
    parser.add_argument("--img_root", type=str, default="/home/cip/nf2022/iz84abaj/MAE/BraTS2021_TrainingData",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--label_root", type=str, default="/home/cip/nf2022/iz84abaj/MAE/BraTS2021_TrainingData",
                        help="The directory containing the training label dataset")
    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=240, type=int,
                        help='images input size')

    parser.add_argument('--resize', default=True, type=bool,
                        help='images resize')

    parser.add_argument('--resize_in', default=128, type=int,
                        help='images resize input size')

    parser.add_argument('--mask_ratio', default=0.70, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/cip/nf2022/iz84abaj/MAE/BraTS2021_TrainingData/BraTS2021_00000//', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print(torch.cuda.is_available())
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    # transform_train = transforms.Compose([
    #         transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    # print(dataset_train)
    #
    # if True:  # args.distributed:
    #     num_tasks = misc.get_world_size()
    #     global_rank = misc.get_rank()
    #     sampler_train = torch.utils.data.DistributedSampler(
    #         dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )
    #     print("Sampler_train = %s" % str(sampler_train))
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )
    fp_list = [os.path.join(r, f[i]) for r, d, f in os.walk("/tmp/BraTS2021_TrainingData") if
              d == []
              for i in range(len(f))]

    fp_list = sorted(fp_list)
    fp_list_sep = []
    fp_imgs_temp = []
    fp_mask_temp = 0
    for i in range(len(fp_list)):
        try:
            if fp_list[i].split("\\")[-2] == fp_list[i + 1].split("\\")[-2]:
                if fp_list[i].split("_")[-1].split(".")[0] in ["flair", "t1", "t1ce", "t2"]:
                    fp_imgs_temp.append(fp_list[i])
                if fp_list[i].split("_")[-1].split(".")[0] == "seg":
                    fp_mask_temp = fp_list[i]
            else:
                fp_imgs_temp.append(fp_list[i])
                fp_list_sep.append((fp_imgs_temp, fp_mask_temp))
                fp_imgs_temp = []

        except IndexError as e:
            pass
    f = []
    l = []
    for (dirpath, dirnames, filenames) in os.walk("/tmp/MAE/BraTS2021_TrainingData"):
        for x in dirnames:
            l.append(x)
            f.append(os.path.join(dirpath, x))
    print(len(l))
    print(len(f))
    fold = np.zeros(1251)
    dict = {'Brats20ID': l, 'path': f, 'fold': fold}

    df = pd.DataFrame(dict)
    df.to_csv('Brats.csv')
    path_to_csv = '/tmp/MAE/Brats.csv'
    data_loader_train = get_dataloader(path_to_csv = path_to_csv, phase='train', fold = 1, batch_size = args.batch_size, num_workers = 4)
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to("cuda")

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256


    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
