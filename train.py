import matplotlib    
matplotlib.use('Agg')

import os
import time
import shutil
from datetime import datetime
from network import dehaze_net
from dataset import DehazeList
from utils import adjust_learning_rate, save_output_images, save_checkpoint, psnr,\
    AverageMeter, draw_curves, dehazing_loss
    
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
import data_transforms as transforms


def train(train_loader, model, criterion, optimizer, epoch, eval_score=None, \
    print_freq=10, logger=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    net = model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = input.float()
        target_var = target.float()

        input_var = input_var.cuda()
        target_var = target_var.cuda(async=True)

        input_var = torch.autograd.Variable(input_var)
        target_var = torch.autograd.Variable(target_var)
        target_var_2 = F.interpolate(target_var, scale_factor=0.5, mode='bilinear')
        target_var_4 = F.interpolate(target_var, scale_factor=0.25, mode='bilinear')
        target_var_8 = F.interpolate(target_var, scale_factor=0.125, mode='bilinear')

        # set gradients to zero
        optimizer.zero_grad()

        out, out_2, out_4, out_8 = net(input_var)
        out_wo_brelu = out.clone()
        out = out.clamp(0, 1)

        # compute loss
        loss = criterion(out_wo_brelu, target_var) \
            + 0.5*criterion(out_2, target_var_2) + 0.25*criterion(out_4, target_var_4) \
                + 0.125*criterion(out_8, target_var_8)
        
        loss.backward()
        optimizer.step()
        losses.update(loss.data, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)

        # measure psnr
        if eval_score is not None:
            scores.update(eval_score(out, target_var), input.size(0))
        
        # log every (print_freq)th epoch
        if i % print_freq == (print_freq-1):
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                (epoch+1), (i+1), len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=scores))
        end = time.time()
    
    return losses.avg, scores.avg


def validate(val_loader, model, criterion, print_freq=10, output_dir='val', \
    save_vis=False, epoch=None, eval_score=None, logger=None, auto_save=True, best_score=0.0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    net = model
    net.eval()

    val_results = []
    end = time.time()
    for i, (input, target, name) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input_var = input.float()
        target_var = target.float()

        input_var = input_var.cuda()
        target_var = target_var.cuda(async=True)

        input_var = torch.autograd.Variable(input_var)
        target_var = torch.autograd.Variable(target_var)
        target_var_2 = F.interpolate(target_var, scale_factor=0.5, mode='bilinear')
        target_var_4 = F.interpolate(target_var, scale_factor=0.25, mode='bilinear')
        target_var_8 = F.interpolate(target_var, scale_factor=0.125, mode='bilinear')
                        
        # compute output
        with torch.no_grad():
            out, out_2, out_4, out_8 = net(input_var)
        
        out_wo_brelu = out.clone()
        out = out.clamp(0, 1)
        
        # compute loss
        loss = criterion(out_wo_brelu, target_var) \
            + 0.5*criterion(out_2, target_var_2) + 0.25*criterion(out_4, target_var_4) \
                + 0.125*criterion(out_8, target_var_8)
        losses.update(loss.data, input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        
        if eval_score is not None:
            scores.update(eval_score(out, target_var), input.size(0))
        
        img_save_dir = os.path.join(output_dir, 'val', 'epoch_{:04d}'.format(epoch+1))
        
        val_results.append(tuple([out, name, img_save_dir]))

        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time,
                loss=losses, top1=scores))
        end = time.time()
    logger.info(' * Score {top1.avg:.3f}'.format(top1=scores))
    
    # save the output images in every 10th epoch
    if save_vis == True:
        for items in val_results:
            pred, name, img_save_dir = items
            save_output_images(pred, name, img_save_dir)

    # save the output images if the model recorded the best score
    if auto_save and (scores.avg > best_score):
        for items in val_results:
            pred, name, img_save_dir = items
            img_save_dir = os.path.join(output_dir, 'val', 'best', 'epoch_{:04d}'.format(epoch+1))
            save_output_images(pred, name, img_save_dir)
        logger.info('Best model: {0}'.format(epoch+1))
    
    print()
    
    return losses.avg, scores.avg


def train_dehaze(args, save_dir='.', logger=None):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size
        
    # logging hyper-parameters
    for k, v in args.__dict__.items():
        logger.info('{0}:\t{1}'.format(k, v))
    
    # Dehazing model
    net = dehaze_net(3, 3, activation=nn.ReLU(inplace=True))
    net = nn.DataParallel(net).cuda()
        
    # criterion for updating weights
    criterion = dehazing_loss(coeff_l1=args.coeff_l1, coeff_cl=args.coeff_cl, coeff_ssim=args.coeff_ssim)
    criterion = criterion.cuda()

    # data-loading code
    data_dir = args.data_dir

    t = []

    if args.random_scale > 0:
        t.append(transforms.RandomScale(args.random_scale))

    t.append(transforms.RandomCrop(crop_size))

    if args.random_rotate > 0:
        t.append(transforms.RandomRotate(args.random_rotate))
    t.extend([transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              transforms.RandomIdentityMapping(p=args.random_identity_mapping),
              transforms.ToTensor(),
    ])

    # DataLoaders for training/validation dataset
    train_loader = torch.utils.data.DataLoader(
        DehazeList(data_dir, 'train', transforms.Compose(t), out_name=False),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        DehazeList(data_dir, 'val', transforms.Compose([
            transforms.ToTensor(),]), out_name=True),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False, drop_last=False
    )

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.Adam(net.parameters(),
                                args.lr,
                                betas=(0.5, 0.999),
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_score = 0
    start_epoch = 0
    train_losses, train_scores, val_losses, val_scores = [], [], [], []

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            optimizer = checkpoint['optimizer']
            train_losses, train_scores, val_losses, val_scores = checkpoint['training_log']
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    lr = args.lr
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch, lr)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch+1, lr))

        train_loss, train_score = train(train_loader, net, criterion, optimizer, epoch, eval_score=psnr, logger=logger)

        val_loss, val_score = 0, 0
        
        if epoch % 10 == 9:
            val_loss, val_score = validate(val_loader, net, criterion, eval_score=psnr, save_vis=True, output_dir=save_dir,\
                                epoch=epoch, logger=logger, best_score=best_score)
        else:
            val_loss, val_score = validate(val_loader, net, criterion, eval_score=psnr, epoch=epoch, output_dir=save_dir,\
                                logger=logger, best_score=best_score)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_scores.append(train_score)
        val_scores.append(val_score)
        
        if epoch == 0:
            best_score = val_score

        is_best = (val_score >= best_score)
        best_score = max(val_score, best_score)
        
        checkpoint_path = save_dir + '/'  + 'checkpoint_latest.pth.tar'

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer,
            'training_log': (train_losses, train_scores, val_losses, val_scores),
        }, is_best, filename=checkpoint_path)

        if epoch % 10 == 9:
            history_path = save_dir + '/' + 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)
            draw_curves(train_losses, train_scores, val_losses, val_scores, epoch+1, save_dir=save_dir+'/curves')
