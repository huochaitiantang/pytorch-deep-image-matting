import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import net
from data import MatTransform, MatDatasetOffline
from torchvision import transforms
import time
import os
import cv2
import numpy as np
from deploy import inference_img_by_crop, inference_img_by_resize, inference_img_whole
import math
import logging


def get_logger(fname):
    assert(fname != "")
    logger = logging.getLogger("DeepImageMatting")
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s-%(filename)s:%(lineno)d-%(levelname)s-%(message)s")

    # log file stream
    handler = logging.FileHandler(fname)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    # log console stream
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--size_h', type=int, required=True, help="height size of input image")
    parser.add_argument('--size_w', type=int, required=True, help="width size of input image")
    parser.add_argument('--crop_h', type=str, required=True, help="crop height size of input image")
    parser.add_argument('--crop_w', type=str, required=True, help="crop width size of input image")
    parser.add_argument('--alphaDir', type=str, required=True, help="directory of alpha")
    parser.add_argument('--fgDir', type=str, required=True, help="directory of fg")
    parser.add_argument('--bgDir', type=str, required=True, help="directory of bg")
    parser.add_argument('--imgDir', type=str, default="", help="directory of img")
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
    parser.add_argument('--stage', type=int, required=True, choices=[0, 1, 2, 3], help="training stage: 0(simple loss), 1, 2, 3")
    parser.add_argument('--testFreq', type=int, default=-1, help="test frequency")
    parser.add_argument('--testImgDir', type=str, default='', help="test image")
    parser.add_argument('--testTrimapDir', type=str, default='', help="test trimap")
    parser.add_argument('--testAlphaDir', type=str, default='', help="test alpha ground truth")
    parser.add_argument('--testResDir', type=str, default='', help="test result save to")
    parser.add_argument('--crop_or_resize', type=str, default="whole", choices=["resize", "crop", "whole"], help="how manipulate image before test")
    parser.add_argument('--max_size', type=int, default=1312, help="max size of test image")
    parser.add_argument('--log', type=str, default='tmplog.txt', help="log file")
    parser.add_argument('--arch', type=str, default='vgg', help="network structure")
    args = parser.parse_args()
    return args


def get_dataset(args):
    train_transform = MatTransform(flip=True)
    
    args.crop_h = [int(i) for i in args.crop_h.split(',')]
    args.crop_w = [int(i) for i in args.crop_w.split(',')]

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    ])

    train_set = MatDatasetOffline(args, train_transform, normalize)
    train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)

    return train_loader

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def build_model(args, logger):
    model = net.VGG16(args)
    model.apply(weight_init)
    
    start_epoch = 1
    best_sad = 100000000.
    if args.pretrain and os.path.isfile(args.pretrain):
        logger.info("loading pretrain '{}'".format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.load_state_dict(ckpt['state_dict'],strict=False)
        logger.info("loaded pretrain '{}' (epoch {})".format(args.pretrain, ckpt['epoch']))
    
    if args.resume and os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        ckpt = torch.load(args.resume)
        start_epoch = ckpt['epoch']
        best_sad = ckpt['best_sad']
        model.load_state_dict(ckpt['state_dict'],strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {} bestSAD {:.3f})".format(args.resume, ckpt['epoch'], ckpt['best_sad']))
    
    return start_epoch, model, best_sad


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


def gen_simple_alpha_loss(alpha, trimap, pred_mattes, args):
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128] = 1.
    alpha_f = alpha / 255.
    diff = pred_mattes - alpha_f
    diff = diff * weighted
    alpha_loss = torch.sqrt(diff ** 2 + 1e-12)

    alpha_loss_weighted = alpha_loss.sum() / (weighted.sum() + 1.)

    return alpha_loss_weighted



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
    alpha_loss = (alpha_loss * t_wi).sum() / (unknown_region_size + 1.)

    # composite rgb loss
    pred_mattes_3 = torch.cat((pred_mattes, pred_mattes, pred_mattes), 1)
    comp = pred_mattes_3 * fg + (1. - pred_mattes_3) * bg
    comp_loss = torch.sqrt((comp - img) ** 2 + 1e-12) / 255.
    comp_loss = (comp_loss * t3_wi).sum() / (unknown_region_size + 1.) / 3.

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
    alpha_loss = (alpha_loss * t_wi).sum() / (unknown_region_size + 1.)
    
    return alpha_loss


def ldata(loss):
    #return loss.data
    return loss.data[0]


def train(args, model, optimizer, train_loader, epoch, logger):
    model.train()
    t0 = time.time()
    #fout = open("train_loss.txt",'w')
    for iteration, batch in enumerate(train_loader, 1):
        torch.cuda.empty_cache()
        img = Variable(batch[0])
        alpha = Variable(batch[1])
        fg = Variable(batch[2])
        bg = Variable(batch[3])
        trimap = Variable(batch[4])
        img_norm = Variable(batch[6])
        img_info = batch[-1]

        if args.cuda:
            img = img.cuda()
            alpha = alpha.cuda()
            fg = fg.cuda()
            bg = bg.cuda()
            trimap = trimap.cuda()
            img_norm = img_norm.cuda()

        #print("Shape: Img:{} Alpha:{} Fg:{} Bg:{} Trimap:{}".format(img.shape, alpha.shape, fg.shape, bg.shape, trimap.shape))
        #print("Val: Img:{} Alpha:{} Fg:{} Bg:{} Trimap:{} Img_info".format(img, alpha, fg, bg, trimap, img_info))

        adjust_learning_rate(args, optimizer, epoch)
        optimizer.zero_grad()

        pred_mattes, pred_alpha = model(torch.cat((img_norm, trimap / 255.), 1))

        if args.stage == 0:
            # stage0 loss, simple alpha loss
            loss = gen_simple_alpha_loss(alpha, trimap, pred_mattes, args)
        elif args.stage == 1:
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

            if args.stage == 0:
                logger.info("Stage0-Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} Speed:{:.5f}s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], ldata(loss), speed, exp_time))
                # stage 2
            elif args.stage == 1:
                # stage 1
                logger.info("Stage1-Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} Alpha:{:.5f} Comp:{:.5f} Speed:{:.5f}s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], ldata(loss), ldata(alpha_loss), ldata(comp_loss), speed, exp_time))
            elif args.stage == 2:
                # stage 2
                logger.info("Stage2-Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} Speed:{:.5f}s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], ldata(loss), speed, exp_time))
            else:
                # stage 3
                logger.info("Stage3-Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} Stage1:{:.5f} Stage2:{:.5f} Speed:{:.5f}s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], ldata(loss), ldata(loss1), ldata(loss2), speed, exp_time))
        #fout.write("{:.5f} {} {}\n".format(loss.data[0], img_info[0][0], img_info[1][0]))
        #fout.flush()
    #fout.close()


def test(args, model, logger):
    model.eval()
    sample_set = []
    img_ids = os.listdir(args.testImgDir)
    img_ids.sort()
    cnt = len(img_ids)
    mse_diffs = 0.
    sad_diffs = 0.
    cur = 0
    t0 = time.time()
    for img_id in img_ids:
        img_path = os.path.join(args.testImgDir, img_id)
        trimap_path = os.path.join(args.testTrimapDir, img_id)

        assert(os.path.exists(img_path))
        assert(os.path.exists(trimap_path))

        img = cv2.imread(img_path)
        trimap = cv2.imread(trimap_path)[:, :, 0]

        assert(img.shape[:2] == trimap.shape[:2])

        img_info = (img_path.split('/')[-1], img.shape[0], img.shape[1])

        cur += 1
        logger.info('[{}/{}] {}'.format(cur, cnt, img_info[0]))        

        with torch.no_grad():
            torch.cuda.empty_cache()

            if args.crop_or_resize == "whole":
                origin_pred_mattes = inference_img_whole(args, model, img, trimap)
            elif args.crop_or_resize == "crop":
                origin_pred_mattes = inference_img_by_crop(args, model, img, trimap)
            else:
                origin_pred_mattes = inference_img_by_resize(args, model, img, trimap)

        # only attention unknown region
        origin_pred_mattes[trimap == 255] = 1.
        origin_pred_mattes[trimap == 0  ] = 0.

        # origin trimap 
        pixel = float((trimap == 128).sum())
        
        # eval if gt alpha is given
        if args.testAlphaDir != '':
            alpha_name = os.path.join(args.testAlphaDir, img_info[0])
            assert(os.path.exists(alpha_name))
            alpha = cv2.imread(alpha_name)[:, :, 0] / 255.
            assert(alpha.shape == origin_pred_mattes.shape)

            mse_diff = ((origin_pred_mattes - alpha) ** 2).sum() / pixel
            sad_diff = np.abs(origin_pred_mattes - alpha).sum()
            mse_diffs += mse_diff
            sad_diffs += sad_diff
            logger.info("sad:{} mse:{}".format(sad_diff, mse_diff))

        origin_pred_mattes = (origin_pred_mattes * 255).astype(np.uint8)
        if not os.path.exists(args.testResDir):
            os.makedirs(args.testResDir)
        cv2.imwrite(os.path.join(args.testResDir, img_info[0]), origin_pred_mattes)

    logger.info("Avg-Cost: {} s/image".format((time.time() - t0) / cnt))
    if args.testAlphaDir != '':
        logger.info("Eval-MSE: {}".format(mse_diffs / cnt))
        logger.info("Eval-SAD: {}".format(sad_diffs / cnt))
    return sad_diffs / cnt


def checkpoint(epoch, save_dir, model, best_sad, logger, best=False):

    epoch_str = "best" if best else "e{}".format(epoch)
    model_out_path = "{}/ckpt_{}.pth".format(save_dir, epoch_str)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_sad': best_sad
    }, model_out_path )
    logger.info("Checkpoint saved to {}".format(model_out_path))


def main():

    args = get_args()
    logger = get_logger(args.log)
    logger.info("Loading args: \n{}".format(args))

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    logger.info("Loading dataset:")
    train_loader = get_dataset(args)

    logger.info("Building model:")
    start_epoch, model, best_sad = build_model(args, logger)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    if args.cuda:
        model = model.cuda()

    # training
    for epoch in range(start_epoch, args.nEpochs + 1):
        train(args, model, optimizer, train_loader, epoch, logger)
        if epoch > 0 and args.testFreq > 0 and epoch % args.testFreq == 0:
            cur_sad = test(args, model, logger)
            if cur_sad < best_sad:
                best_sad = cur_sad
                checkpoint(epoch, args.saveDir, model, best_sad, logger, True)
        if epoch > 0 and epoch % args.ckptSaveFreq == 0:
            checkpoint(epoch, args.saveDir, model, best_sad, logger)


if __name__ == "__main__":
    main()
