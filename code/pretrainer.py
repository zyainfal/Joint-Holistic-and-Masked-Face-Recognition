# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import math
import sys

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import io
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from einops import rearrange
from pathlib import Path
import torchvision

#from timm.models import create_model
from VIT.mim_pytorch import facet_base, MAE
from optim_factory import create_optimizer
from data.data_pipe import get_train_dataset
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils


def get_args():
    parser = argparse.ArgumentParser('training script', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--eval_freq', default=2000, type=int)
    parser.add_argument('--print_freq', default=10, type=int)

    # Model parameters
    parser.add_argument('--model', default='facet_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--train_type', default='mae', type=str, metavar='pretrain type',
                        help='type of model to train')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=112, type=int,
                        help='images input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.0 for mae, 0.1 for splitmask)')
    
    parser.add_argument('--pool', default='mean', type=str,
                        help='pool type: [mean, adj] ')

    parser.add_argument('--ntype', default='prenorm', type=str,
                        help='pool type: [prenorm, postnorm, deepnorm] ')
                    

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--fp16', action='store_true', dest='fp16',
                        help='use fp16 for training')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data1/yuhao.zhu/data/faces_emore/', type=str,
                        help='dataset path')
    parser.add_argument('--masked_faces', default=0.0, type=float,
                        help='ratio of the masked faces/images')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()



def get_model(args):
    print(f"Creating model: {args.model}")
    if args.model == "facet_base":
        bb = facet_base(drop_path_rate=args.drop_path,classification=True, fp16=args.fp16, ntype=args.ntype)
    else:
        raise NotImplementedError
    
    if args.train_type == "mae":
        model = MAE(bb, masking_ratio = args.mask_ratio, fp16=args.fp16, ntype=args.ntype)
    else:
        raise NotImplementedError
    return model

def de_preprocess(tensor):
    return tensor*0.5 + 0.5

def fill_by_mask(img, value, mask, device):
    batch_range = torch.arange(10, device = device)[:, None]
    img = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 8, p2 = 8)
    img[batch_range, mask] = value
    img = rearrange(img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=14, p1 = 8, p2 = 8)
    return img

def prepare_vis(img, pred, mask, device):
    null = torch.zeros_like(pred)
    crap_img = fill_by_mask(img.clone(), null, mask, device)
    pred_img = fill_by_mask(img.clone(), pred, mask, device)
    return crap_img, pred_img, img

def compare_to_save(a1,b1):
    if a1 < b1:
        return True
    return False


def check_inf_null(loss_value):
    if not math.isfinite(loss_value):
        print("Loss is {}, stopping training".format(loss_value))
        sys.exit(1)
    return

def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # get dataset
    dataset_train, number_classes = get_train_dataset(args)
    args.number_classes = number_classes

    # get model
    model = get_model(args)
    try:
        patch_size = model.backbone.patch_embed.patch_size
        print("Patch size = %s" % str(patch_size))
        args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
        args.patch_size = patch_size
    except:
        patch_size = [args.input_size, args.input_size]
        args.window_size = args.input_size
        args.patch_size = args.input_size

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.lr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler() if args.fp16 else None

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        # train
        start_steps=epoch * num_training_steps_per_epoch
        if log_writer is not None:
            log_writer.set_step(start_steps)
        
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        
        for step, (batch, _, _) in enumerate(metric_logger.log_every(data_loader_train, args.print_freq, header)):
            it = start_steps + step  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            images = batch.to(device, non_blocking=True)

            loss, pred_pixel_values, masked_indices = model(images)
            loss_value = loss.item()
            check_inf_null(loss_value)

            if loss_scaler is None:
                loss.backward()
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                grad_norm = None
                loss_scale_value = 1
            else:
                grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                        parameters=model.parameters(), create_graph=False,
                                        update_grad=True)
                loss_scale_value = loss_scaler.state_dict()["scale"]
            optimizer.zero_grad()
            torch.cuda.synchronize()
            with torch.no_grad():
                # record 10 imgs for visulization
                crap_img, pred_img, img = prepare_vis(images[:10], pred_pixel_values[:10], masked_indices[:10], device)
                vis_map = torch.cat([crap_img, pred_img, img], 0)
            
            metric_logger.update(rec_loss=loss.item())
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(rec_loss=loss.item(), head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                if (it+1) % (num_training_steps_per_epoch//10) == 0:
                    grid = torchvision.utils.make_grid(vis_map.cpu(), nrow=10, normalize=True)
                    log_writer.writer.add_image('vis', grid, it+1)
                log_writer.set_step()
                
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        # log
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        train_stats.clear()
        del metric_logger

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.output_dir and utils.is_main_process():
        save_tag = "last"
        utils.save_model(
            args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=args.epochs, tag =save_tag)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
