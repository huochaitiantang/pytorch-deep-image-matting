import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import net
import resnet_aspp
from data import MatTransform, MatDataset
from torchvision import transforms
import time
import os


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--size_h', type=int, required=True, help="height size of input image")
    parser.add_argument('--size_w', type=int, required=True, help="width size of input image")
    parser.add_argument('--crop_h', type=int, required=True, help="crop height size of input image")
    parser.add_argument('--crop_w', type=int, required=True, help="crop width size of input image")
    parser.add_argument('--imgDir', type=str, required=True, help="directory of image")
    parser.add_argument('--alphaDir', type=str, required=True, help="directory of alpha")
    parser.add_argument('--fgDir', type=str, required=True, help="directory of fg")
    parser.add_argument('--bgDir', type=str, required=True, help="directory of bg")
    parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--step', type=int, default=10, help='epoch of learning decay')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--resume', type=str, help="checkpoint that model resume from")
    parser.add_argument('--pretrain', type=str, help="checkpoint that model pretrain from")
    parser.add_argument('--saveDir', type=str, help="checkpoint that model save to")
    parser.add_argument('--printFreq', type=int, default=10, help="checkpoint that model save to")
    parser.add_argument('--ckptSaveFreq', type=int, default=10, help="checkpoint that model save to")
    parser.add_argument('--wl_weight', type=float, default=0.5, help="alpha loss weight")
    parser.add_argument('--stage', type=int, required=True, help="training stage: 1, 2, 3")
    parser.add_argument('--arch', type=str, required=True, choices=["vgg16","resnet50_aspp"], help="net backbone")
    parser.add_argument('--in_chan', type=int, default=4, choices=[3, 4], help="input channel 3(no trimap) or 4")
    args = parser.parse_args()
    print(args)
    return args


def get_dataset(args):
    train_transform = MatTransform(flip=True)
    
    train_set = MatDataset(args.imgDir, args.alphaDir, args.fgDir, args.bgDir, args.size_h, args.size_w, args.crop_h, args.crop_w, train_transform)
    train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)

    return train_loader

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def build_model(args):
    if args.arch == "resnet50_aspp":
        model = resnet_aspp.resnet50(args)
    else:
        model = net.DeepMatting(args)
        model.apply(weight_init)
    
    start_epoch = 1
    if args.pretrain and os.path.isfile(args.pretrain):
        print("=> loading pretrain '{}'".format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.load_state_dict(ckpt['state_dict'],strict=False)
        print("=> loaded pretrain '{}' (epoch {})".format(args.pretrain, ckpt['epoch']))
    
    if args.resume and os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        ckpt = torch.load(args.resume)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'],strict=True)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))
    
    return start_epoch, model    


def adjust_learning_rate(args, opt, epoch):
    if args.step > 0 and epoch >= args.step:
        lr = args.lr * 0.1
        for param_group in opt.param_groups:
            param_group['lr'] = lr


def format_second(secs):
    h = int(secs / 3600)
    m = int((secs % 3600) / 60)
    s = int(secs % 60)
    ss = "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format(h,m,s)
    return ss    


def gen_loss(img, alpha, fg, bg, trimap, pred_mattes):
    wi = torch.zeros(trimap.shape)
    wi[trimap == 128] = 1.
    t_wi = wi.cuda()
    t3_wi = torch.cat((wi, wi, wi), 1).cuda()
    unknown_region_size = t_wi.sum()

    #assert(t_wi.shape == pred_mattes.shape)
    #assert(t3_wi.shape == img.shape)

    # alpha diff
    alpha = alpha / 255.
    alpha_loss = torch.sqrt((pred_mattes - alpha)**2 + 1e-12)
    alpha_loss = (alpha_loss * t_wi).sum() / unknown_region_size

    # composite rgb loss
    pred_mattes_3 = torch.cat((pred_mattes, pred_mattes, pred_mattes), 1)
    comp = pred_mattes_3 * fg + (1. - pred_mattes_3) * bg
    comp_loss = torch.sqrt((comp - img) ** 2 + 1e-12) / 255.
    comp_loss = (comp_loss * t3_wi).sum() / unknown_region_size / 3.

    #print("Loss: AlphaLoss:{} CompLoss:{}".format(alpha_loss, comp_loss))
    return alpha_loss, comp_loss
   

def gen_alpha_pred_loss(alpha, pred_alpha, trimap):
    wi = torch.zeros(trimap.shape)
    wi[trimap == 128] = 1.
    t_wi = wi.cuda()
    unknown_region_size = t_wi.sum()

    # alpha diff
    alpha = alpha / 255.
    alpha_loss = torch.sqrt((pred_alpha - alpha)**2 + 1e-12)
    alpha_loss = (alpha_loss * t_wi).sum() / unknown_region_size
    
    return alpha_loss


def train(args, model, optimizer, train_loader, epoch):
    t0 = time.time()
    assert(args.stage in [1, 2, 3])
    for iteration, batch in enumerate(train_loader, 1):
        img = Variable(batch[0])
        alpha = Variable(batch[1])
        fg = Variable(batch[2])
        bg = Variable(batch[3])
        trimap = Variable(batch[4])
        img_info = batch[5]

        if args.cuda:
            img = img.cuda()
            alpha = alpha.cuda()
            fg = fg.cuda()
            bg = bg.cuda()
            trimap = trimap.cuda()

        #print("Shape: Img:{} Alpha:{} Fg:{} Bg:{} Trimap:{}".format(img.shape, alpha.shape, fg.shape, bg.shape, trimap.shape))
        #print("Val: Img:{} Alpha:{} Fg:{} Bg:{} Trimap:{} Img_info".format(img, alpha, fg, bg, trimap, img_info))

        adjust_learning_rate(args, optimizer, epoch)
        optimizer.zero_grad()

        if args.in_chan == 3:
            pred_mattes, pred_alpha = model(img)
        else:
            pred_mattes, pred_alpha = model(torch.cat((img, trimap), 1))

        if args.stage == 1:
            # stage1 loss
            alpha_loss, comp_loss = gen_loss(img, alpha, fg, bg, trimap, pred_mattes)
            loss = alpha_loss * args.wl_weight + comp_loss * (1. - args.wl_weight)
        elif args.stage == 2:
            # stage2 loss
            loss = gen_alpha_pred_loss(alpha, pred_alpha, trimap)
        else:
            # stage3 loss = stage1 loss + stage2 loss
            alpha_loss, comp_loss = gen_loss(img, alpha, fg, bg, trimap, pred_mattes)
            loss1 = alpha_loss * args.wl_weight + comp_loss * (1. - args.wl_weight)
            loss2 = gen_alpha_pred_loss(alpha, pred_alpha, trimap)
            loss = loss1 + loss2
        
        loss.backward()
        optimizer.step()

        if iteration % args.printFreq ==  0:
            t1 = time.time()
            num_iter = len(train_loader)
            speed = (t1 - t0) / iteration
            exp_time = format_second(speed * (num_iter * (args.nEpochs - epoch + 1) - iteration))

            if args.stage == 1:
                # stage 1
                print("Stage1-Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} Alpha:{:.5f} Comp:{:.5f} Speed:{:.5f}s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], loss.data[0], alpha_loss.data[0], comp_loss.data[0], speed, exp_time))
            elif args.stage == 2:
                # stage 2
                print("Stage2-Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} Speed:{:.5f}s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], loss.data[0], speed, exp_time))
            else:
                # stage 3
                print("Stage3-Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} Stage1:{:.5f} Stage2:{:.5f} Speed:{:.5f}s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], loss.data[0], loss1.data[0], loss2.data[0], speed, exp_time))
            

def checkpoint(epoch, save_dir, model):
    model_out_path = "{}/ckpt_e{}.pth".format(save_dir, epoch)
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
    }, model_out_path )
    print("Checkpoint saved to {}".format(model_out_path))

def main():

    print("===> Loading args")
    args = get_args()

    print("===> Environment init")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    print('===> Loading datasets')
    train_loader = get_dataset(args)

    print('===> Building model')
    start_epoch, model = build_model(args)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    if args.cuda:
        model = model.cuda()

    # training
    for epoch in range(start_epoch, args.nEpochs + 1):
        train(args, model, optimizer, train_loader, epoch)
        if epoch > 0 and epoch % args.ckptSaveFreq == 0:
            checkpoint(epoch, args.saveDir, model)


if __name__ == "__main__":
    main()
