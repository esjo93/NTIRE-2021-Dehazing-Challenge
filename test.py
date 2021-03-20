import os
import time
from network import dehaze_net
from dataset import DehazeList

from utils import save_output_images, AverageMeter

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
import data_transforms as transforms


def test(eval_data_loader, model, output_dir='test', save_vis=True, logger=None):
    model.eval()

    batch_time, data_time, end = AverageMeter(), AverageMeter(), time.time()

    for iter, (image, name) in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        image_var = image.float().cuda()
        image_var = torch.autograd.Variable(image_var)
        _, _, h, w = image_var.size()
        
        is_proper_size = True if (h%8, w%8) == (0, 0) else False
        if not is_proper_size:
            image_var = F.interpolate(image_var, size=(h+8 - h%8, w+8 - w%8), mode='bilinear')
        with torch.no_grad():
            out, _, _, _ = model(image_var)

        if not is_proper_size:
            out = F.interpolate(out, size=(h, w), mode='bilinear')

        batch_time.update(time.time() - end)

        pred = out.clamp(0, 1)

        if save_vis:
            save_output_images(pred, name, output_dir)
        
        logger.info('Eval: [{0:04d}/{1:04d}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter+1, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
        end = time.time()


def test_dehaze(args, save_dir='.', logger=None):
    batch_size = args.batch_size
    num_workers = args.workers
    
    for k, v in args.__dict__.items():
        print(k, ':', v)

    model = dehaze_net(activation=nn.ReLU(inplace=True))
    model = nn.DataParallel(model).cuda()
    data_dir = args.data_dir

    dataset = DehazeList(data_dir, 'test', transforms.Compose([
        transforms.ToTensor(),
        ]), out_name=True)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
            return

    save_dir = os.path.dirname(args.resume)
    out_dir = os.path.join(save_dir, '{:03d}_{}'.format(start_epoch, 'test'))
    test(test_loader, model, save_vis=True, output_dir=out_dir, logger=logger)
