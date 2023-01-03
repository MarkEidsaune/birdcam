'''
Pytorch Bird Detection Training

For multi-gpu training use:

    torchrun --nproc_per_node=$n_gpus train.py

'''

import os
import datetime
import time
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from birdcam.engine import train_one_epoch, evaluate
import birdcam.utils as utils
import birdcam.transforms as T
import torch.distributed as dist
from na_birds_dataset import NABirdsDataset

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
# from torchvision.transforms import InterpolationMode
# from transforms import SimpleCopyPaste

def get_transforms(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_dataset(path, split):
    ds = NABirdsDataset(root=path, transforms=get_transforms(train=True))
    ds_test = NABirdsDataset(root=path, transforms=get_transforms(train=False))
    
    num_classes = ds.num_classes
    
    split_idx = int(np.floor(split * len(ds)))

    indices = torch.randperm(len(ds)).tolist()
    ds = torch.utils.data.Subset(ds, indices[:split_idx])
    ds_test = torch.utils.data.Subset(ds_test, indices[split_idx:])
    
    return ds, ds_test, num_classes

def main(args):
       
    utils.init_distributed_mode(args)
    
    device = torch.device(args.device)
        
    print('Loading data')
    ds, ds_test, num_classes = get_dataset(args.data_path, args.split)
    
    print('Creating data loaders')
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ds)
        test_sampler = torch.utils.data.distributed.DistributedSampler(ds_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(ds)
        test_sampler = torch.utils.data.SequentialSampler(ds_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(ds, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, args.batch_size
        )
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True
        )

    data_loader = torch.utils.data.DataLoader(
        ds, batch_sampler=train_batch_sampler, 
        num_workers=args.workers, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        ds_test, batch_size=1, sampler=test_sampler, 
        num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print('Creating model')
    model = torchvision.models.detection.__dict__[args.model](pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        parameters, lr=args.lr, momentum=args.momentum, 
        weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma
    )
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        args.epochs += args.start_epoch

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            utils.save_on_master(
                checkpoint, 
                os.path.join(args.output_dir, args.model, f"epoch_{epoch}.pth")
            )

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', default='/media/nvme2/nabirds', type=str, help='dataset path')
    parser.add_argument('--split', default=0.8, type=float, help='training data split proportion')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', type=str, help='model name')
    parser.add_argument('--device', default='cuda', type=str, help='device (use cuda or cpu)')
    parser.add_argument('--batch-size', default=2, type=int, help='images per gpu')
    parser.add_argument('--epochs', default=26, type=int, help='number of total epochs to run')
    parser.add_argument( '--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--lr', default=0.00125, type=float, 
                        help='initial learning rate. equal to 0.02/8*$n_gpus')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--lr-step-size', default=8, type=int, 
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, 
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, 
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--print-freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./checkpoints', type=str, help='path to save outputs')
    parser.add_argument('--resume', default='', type=str, help='path of checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    args = parser.parse_args()
    
    if args.output_dir:
        os.makedirs(os.path.join(args.output_dir, args.model), exist_ok=True)
    
    main(args)