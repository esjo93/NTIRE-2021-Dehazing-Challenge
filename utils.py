import os
import threading
import numpy as np
import shutil
from math import exp
from PIL import Image
import matplotlib.pyplot as plt

from network import VGG19
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image


class dehazing_loss(nn.Module):
    def __init__(self, coeff_l1=1.0, coeff_cl=0.5, coeff_ssim=0.1):
        super(dehazing_loss, self).__init__()
        self.content_loss = ContentLoss()
        self.coeff_l1 = coeff_l1
        self.coeff_cl = coeff_cl  # content loss coefficient
        self.coeff_ssim = coeff_ssim  # ssim loss coefficient
        self.ssim = 0 if self.coeff_ssim == 0 else SSIM(window_size=11)


    def forward(self, input_wo_brelu, target):
        input = input_wo_brelu.clone().clamp(0, 1)
        loss = self.coeff_l1 * F.l1_loss(input, target) + self.coeff_cl * self.content_loss(input, target) \
            + self.coeff_ssim * (1 - ssim(input, target)) 
        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


class ContentLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        self.add_module('vgg', VGG19().cuda())
        self.criterion = torch.nn.L1Loss().cuda()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Gaussiansmoothing(img, channel=3, window_size = 11):
    window = create_window(window_size, channel, sigma=5)
    
    if img.is_cuda:
        window = window.cuda(img.get_device())
    window = window.type_as(img)

    pad = window_size//2
    padded_img = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    x_smooth = F.conv2d(padded_img, window, padding=0, groups=channel)
    
    return x_smooth, img - x_smooth


def psnr(output, target):
    """
    Computes the PSNR.
    1 means the maximum value of intensity(255)
    """
    psnr = 0
    
    output_temp = output.clone().clamp(0, 1)

    with torch.no_grad():
        mse = torch.mean((output_temp - target)**2, dim=(1, 2, 3))
        psnr = 10 * torch.log10(1 / mse)
        psnr = torch.mean(psnr).item()
        
    return psnr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Saves the serialized current checkpoint
    
    Params

    state = 
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))


def adjust_learning_rate(args, optimizer, epoch, prev_lr):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.5 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    elif args.lr_mode == None:
        return optimizer.param_groups[0]['lr']
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    
    if lr != prev_lr:
        print('Learning rate has changed!')
        for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    return lr


def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for ind in range(len(filenames)):
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = os.path.split(fn)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        pred = predictions[ind]
        save_image(pred, fn)


def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILasdfINEAR))

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    return out


def draw_curves(training_loss, training_score, validation_loss, validation_score, epoch, save_dir='./curves'):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    x = np.arange(1, epoch+1, step=1)

    axes[0].plot(x, training_loss, label='train', alpha=0.8)
    axes[0].plot(x, validation_loss, label='val', alpha=0.8)
    axes[0].set_xlim(0, epoch+1)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylim(0, 1.2)
    axes[0].set_ylabel('Losses')
    axes[0].legend()
    axes[0].grid()
    
    axes[1].plot(x, training_score, label='train', alpha=0.8)
    axes[1].plot(x, validation_score, label='val', alpha=0.8)
    axes[1].set_xlim(0, epoch+1)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylim(5, 25.0)
    axes[1].set_ylabel('Scores')
    axes[1].legend()
    axes[1].grid()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(os.path.join(save_dir, 'epoch_{:04d}_curve.png'.format(epoch)))
    plt.close('all')
