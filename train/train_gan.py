import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys

import clip
import warnings

from utils import save_checkpoint
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse

import dataset_gan
import time
from models.model_vit_320_4_gan import ViT
parser = argparse.ArgumentParser()



def mspooling_loss(output, target):
    output1 = nn.AvgPool1d(3, stride=2)(output)
    target1 = nn.AvgPool1d(3, stride=2)(target)
    loss1 = F.mse_loss(output1, target1, reduction='mean')

    output2 = nn.AvgPool1d(5, stride=4)(output)
    target2 = nn.AvgPool1d(5, stride=4)(target)
    loss2 = F.mse_loss(output2, target2, reduction='mean')

    output3 = nn.AvgPool1d(7, stride=6)(output)
    target3 = nn.AvgPool1d(7, stride=6)(target)
    loss3 = F.mse_loss(output3, target3, reduction='mean')

    output4 = nn.AvgPool1d(9, stride=8)(output)
    target4 = nn.AvgPool1d(9, stride=8)(target)
    loss4 = F.mse_loss(output4, target4, reduction='mean')

    output5 = nn.AvgPool1d(11, stride=10)(output)
    target5 = nn.AvgPool1d(11, stride=10)(target)
    loss5 = F.mse_loss(output5, target5, reduction='mean')

    output6 = nn.AvgPool1d(13, stride=12)(output)
    target6 = nn.AvgPool1d(13, stride=12)(target)
    loss6 = F.mse_loss(output6, target6, reduction='mean')

    loss7 = F.mse_loss(output, target, reduction='mean')
    return loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7


def main():


    global args, best_prec1

    best_prec1 = 1e6
    args = parser.parse_args()
    args.original_lr = 1e-3
    args.lr = 1e-3
    args.batch_size = 24
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 100
    args.steps = [-1, 1, 200, 300]
    args.scales = [1, 1, 0.1, 0.1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 1

    pt = torch.load("vgg_new_total_3.pth", map_location='cuda:0')


    torch.cuda.manual_seed(args.seed)

    model = ViT(512, 512, 300, 8, 64)
    model = model.cuda()


    criterion = torch.nn.CrossEntropyLoss()  # nn.KLDivLoss(size_average=False).cuda()   #nn.SmoothL1Loss(size_average=False).cuda()
    mse_loss = torch.nn.MSELoss(size_average=True)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,

                                 weight_decay=args.decay)  # momentum=args.momentum,




    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)
        model.train()
        train(pt, model, criterion, mse_loss, optimizer, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, "pretrained_models", "GAN_checkpoint.pth.tar")



import math


def train(train_list, model, criterion, mse_loss, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset_gan.listDataset(train_list,
                                shuffle=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225]),
                                ]),
                                train=True,
                                seen=0,
                                batch_size=args.batch_size,
                                num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i, (audio_embed, target) in enumerate(train_loader):
        # print(audio_embed.sum())
        # (1,77,768)
        data_time.update(time.time() - end)

        audio_embed = audio_embed.cuda()
        audio_embed = Variable(audio_embed)
        output_ori = model(audio_embed).squeeze(dim=1)
        output = output_ori.flatten(1)

        target = target.type(torch.FloatTensor).cuda()
        target = Variable(target)
        target_ori = target.squeeze(1)
        target = target_ori.flatten(1)
        output1 = output / output.norm(dim=-1, keepdim=True)
        target1 = target / target.norm(dim=-1, keepdim=True)

        # must use contrastive loss
        out_tag = (output1 @ target1.T) * math.exp(0.07)
        tag_out = (target1 @ output1.T) * math.exp(0.07)
        label = torch.arange(output1.shape[0], dtype=torch.long).cuda()

        contrastive_loss = (criterion(out_tag, label) + criterion(tag_out, label)) / 2

        mspoolingloss = mspooling_loss(output, target)
        loss = contrastive_loss + mspoolingloss + mse_loss(output, target)

        losses.update(loss.item(), audio_embed.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def validate(val_list, model, criterion):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset_gan.listDataset(val_list,
                                shuffle=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225]),
                                ]), train=False),
        batch_size=args.batch_size)

    model.eval()

    mae = 0

    for i, (img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)

        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())

    mae = mae / len(test_loader)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr

    for i in range(len(args.steps)):

        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


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


if __name__ == '__main__':
    seed = 100
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.enabled = False
    main()