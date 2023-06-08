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
from torchvision import transforms as trans
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path
#from timm.models import create_model
from VIT.mim_pytorch import facet_base
from CNN.iresnet import iresnet100
from optim_factory import create_optimizer, LayerDecayValueAssigner
from loss.face_losses import ArcFace, CurricularFace, MagFace, l2_norm
from loss.partial_fc import PartialFCAdamW, PartialFCSGD
from data.data_pipe import get_train_dataset, get_val_data
#from engine import train_one_epoch
from verfication import evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from evaluation.pair_parser import PairsParserFactory
from evaluation.run_verification import run_test

def get_args():
    parser = argparse.ArgumentParser('training script', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--print_freq', default=10, type=int)

    # Model parameters
    parser.add_argument('--model', default='facet_base', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--head', default='arcface', type=str, metavar='MODEL',
                        help='Name of head to train')

    parser.add_argument('--input_size', default=112, type=int,
                        help='images input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    parser.add_argument('--pool', default='mean', type=str,
                        help='pool type: [mean, adj] ')
   
    parser.add_argument('--ntype', default='deepnorm', type=str,
                        help='pool type: [prenorm, postnorm, deepnorm] ')
    
    parser.add_argument('--transform_layer', default=12, type=int,
                        help='number of layers that has prompts')
                    

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

    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--layer_decay', type=float, default=1, metavar='LR',
                        help='lower lr ')

    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data1/yuhao.zhu/data/faces_emore/', type=str,
                        help='dataset path')
    parser.add_argument('--masked_faces', default=0.1, type=float,
                        help='ratio of masked faces/images')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
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
    flag = 0
    if args.model == "facet_base":
        bb = facet_base(drop_path_rate=args.drop_path,classification=True, fp16=args.fp16, ntype=args.ntype, n_prompt_layer=args.transform_layer)
    elif args.model == "ir100":
        bb = iresnet100(fp16=args.fp16)
        flag=1
    return bb, flag

def de_preprocess(tensor):
    return tensor*0.5 + 0.5

hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

def board_val(writer, step, db_name, accuracy, best_threshold, roc_curve_tensor):
    writer.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, step)
    return 

def run_evaluate(model, device, conf, carray, issame, num_features=768, nrof_folds = 5, tta = False):
    #self.model.eval()
    idx = 0
    num_features = 512
    embeddings = np.zeros([len(carray), num_features])
    with torch.no_grad():
        while idx + conf.batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + conf.batch_size])
            ismask = torch.zeros(batch.size(0))
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch.to(device, non_blocking=True), ismask.to(device, non_blocking=True)) + model(fliped.to(device, non_blocking=True), ismask.to(device, non_blocking=True))
                embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch).cpu().numpy()
            else:
                embeddings[idx:idx + conf.batch_size] = l2_norm(model(batch.to(device, non_blocking=True), ismask.to(device, non_blocking=True))).cpu().numpy()
            idx += conf.batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])        
            ismask = torch.zeros(batch.size(0))    
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch.to(device, non_blocking=True), ismask.to(device, non_blocking=True)) + model(fliped.to(device, non_blocking=True), ismask.to(device, non_blocking=True))
                embeddings[idx:] = l2_norm(emb_batch).cpu().numpy()
            else:
                embeddings[idx:] = l2_norm(model(batch.to(device, non_blocking=True), ismask.to(device, non_blocking=True))).cpu().numpy()
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = trans.ToTensor()(roc_curve)
    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
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

    agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame = get_val_data(args)

    # get model
    model, is_cnn = get_model(args)
    # load checkpoint
    if args.resume != '':
        checkpoint = torch.load(args.resume, map_location='cpu')
        #model.load_state_dict(checkpoint['model'], strict=True)
        utils.load_state_dict(model, checkpoint['model'], prefix='encoder.')
    try:
        patch_size = model.patch_embed.patch_size
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
    
    if args.layer_decay < 1.0:
        num_layers = model_without_ddp.get_num_layers()
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))
    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    if args.head == "arcface":
        margin_loss = ArcFace().cuda()
    elif args.head == "curricularface":
        margin_loss = CurricularFace().cuda()
    elif args.head == "magface":
        margin_loss = MagFace().cuda()
        
    if is_cnn:
        module_partial_fc = PartialFCSGD(margin_loss, model_without_ddp.num_features, args.number_classes, 0.1)
    else:
        module_partial_fc = PartialFCAdamW(margin_loss, model_without_ddp.num_classes, args.number_classes, 0.1)
    module_partial_fc.train().cuda()

    optimizer = create_optimizer(
            args, model_without_ddp, module_partial_fc, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler() if args.fp16 else None

    print("Use step level LR & WD scheduler!")
    scheduler = utils.step_scheduler if is_cnn else utils.cosine_scheduler
    lr_schedule_values = scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if log_writer is not None:
        model_without_ddp.eval()
        lfw_pair = PairsParserFactory(os.path.join(args.data_path, "lfw/pair.list"), "LFW").get_parser().parse_pairs()
        lfw_path = os.path.join(args.data_path, "lfw_masked_a/")
        lfw_list = os.listdir(lfw_path)

        cfp_pair = PairsParserFactory(os.path.join(args.data_path, "cfp_fp/pair.list"), "CFPFP").get_parser().parse_pairs()
        cfp_path = os.path.join(args.data_path, "cfp_fp_masked_a/")
        cfp_list = os.listdir(cfp_path)

        agedb_pair = PairsParserFactory(os.path.join(args.data_path, "agedb_30/pair.list"), "AgeDB30").get_parser().parse_pairs()
        agedb_path = os.path.join(args.data_path, "agedb_30_masked_a")
        agedb_list = os.listdir(agedb_path)
        
        rmfvd_pair = PairsParserFactory(os.path.join(args.data_path, "masked_pairs.txt"), "RMFVD").get_parser().parse_pairs()
        rmfvd_path = os.path.join(args.data_path, "masked_whn_a")
        rmfvd_list = []
        people_list = os.listdir(rmfvd_path)
        for people in people_list:
            people_path = os.path.join(rmfvd_path, people)
            imgs = os.listdir(people_path)
            for img in imgs:
                rmfvd_list.append(os.path.join(people, img))
        
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    model.train()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        # train
        start_steps=epoch * num_training_steps_per_epoch
        if log_writer is not None:
            log_writer.set_step(start_steps)
        
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        for step, (batch, ismask, labels) in enumerate(metric_logger.log_every(data_loader_train, args.print_freq, header)):
            it = start_steps + step  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            images = batch.to(device, non_blocking=True)
            ismask = ismask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # inference
            feat = model(images, ismask)
            loss = module_partial_fc(feat, labels, optimizer)
                
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
            

            metric_logger.update(loss=loss_value)
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
            #metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.set_step()
                # eval every eval_freq
                if (it+1) % args.eval_freq == 0 :
                    model_without_ddp.eval()
                    with torch.no_grad():
                        accuracy1, best_threshold, roc_curve_tensor = run_evaluate(model_without_ddp, device, args, lfw, lfw_issame, num_features=model_without_ddp.num_features)
                        board_val(log_writer, it+1, 'lfw', accuracy1, best_threshold, roc_curve_tensor)
                        accuracy2, best_threshold, roc_curve_tensor = run_evaluate(model_without_ddp, device, args, cfp_fp, cfp_fp_issame, num_features=model_without_ddp.num_features)
                        board_val(log_writer, it+1, 'cfp_fp', accuracy2, best_threshold, roc_curve_tensor)
                        accuracy3, best_threshold, roc_curve_tensor = run_evaluate(model_without_ddp, device, args, agedb_30, agedb_30_issame, num_features=model_without_ddp.num_features)
                        board_val(log_writer, it+1, 'agedb_30', accuracy3, best_threshold, roc_curve_tensor)
                    
                        accuracy4 = run_test(model_without_ddp, lfw_path, lfw_list, lfw_pair, mask_type=1, batch_size=256, tta=False)
                        board_val(log_writer, it+1, 'masked_lfw', accuracy4, 0, 0)
                        accuracy5 = run_test(model_without_ddp, cfp_path, cfp_list, cfp_pair, mask_type=1, batch_size=256, tta=False)
                        board_val(log_writer, it+1, 'masked_cfp_fp', accuracy5, 0, 0)
                        accuracy6 = run_test(model_without_ddp, agedb_path, agedb_list, agedb_pair, mask_type=1, batch_size=256, tta=False)
                        board_val(log_writer, it+1, 'masked_agedb_30', accuracy6, 0, 0)

                        accuracy7 = run_test(model_without_ddp, rmfvd_path, rmfvd_list, rmfvd_pair, mask_type=2, batch_size=256, tta=False)
                        board_val(log_writer, it+1, 'rmfvd', accuracy7, 0, 0)

                    model.train()
                    if args.output_dir:
                        it_tag = str(it).zfill(7)
                        save_tag = "{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}".format(it_tag,accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,accuracy7)
                        utils.save_model(
                            args=args,model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=it, tag =save_tag)

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


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
