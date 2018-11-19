import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import net
import net_nobn
import resnet_aspp
from data import MatTransform, MatDataset, MatDatasetOffline
from torchvision import transforms
import time
import os
import cv2
import numpy as np


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--size_h', type=int, required=True, help="height size of input image")
    parser.add_argument('--size_w', type=int, required=True, help="width size of input image")
    parser.add_argument('--crop_h', type=int, required=True, help="crop height size of input image")
    parser.add_argument('--crop_w', type=int, required=True, help="crop width size of input image")
    parser.add_argument('--alphaDir', type=str, required=True, help="directory of alpha")
    parser.add_argument('--fgDir', type=str, required=True, help="directory of fg")
    parser.add_argument('--bgDir', type=str, required=True, help="directory of bg")
    parser.add_argument('--imgDir', type=str, default="", help="directory of img")
    parser.add_argument('--dataOffline', action='store_true', help='use training data offline compoiste,  true require imDir not empty')
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
    parser.add_argument('--arch', type=str, required=True, choices=["vgg16","vgg16_nobn", "resnet50_aspp"], help="net backbone")
    parser.add_argument('--in_chan', type=int, default=4, choices=[3, 4], help="input channel 3(no trimap) or 4")
    parser.add_argument('--testFreq', type=int, default=-1, help="test frequency")
    parser.add_argument('--testImgDir', type=str, default='', help="test image")
    parser.add_argument('--testTrimapDir', type=str, default='', help="test trimap")
    parser.add_argument('--testAlphaDir', type=str, default='', help="test alpha ground truth")
    parser.add_argument('--testResDir', type=str, default='', help="test result save to")
    parser.add_argument('--addGrad', action='store_true', help='use grad as a input channel?')
    args = parser.parse_args()
    print(args)
    return args


def get_dataset(args):
    train_transform = MatTransform(flip=True)
    
    if(args.dataOffline):
        assert(args.imgDir != "")
        train_set = MatDatasetOffline(args.alphaDir, args.fgDir, args.bgDir, args.imgDir, args.size_h, args.size_w, args.crop_h, args.crop_w, train_transform)
    else:
        train_set = MatDataset(args.alphaDir, args.fgDir, args.bgDir, args.size_h, args.size_w, args.crop_h, args.crop_w, train_transform)
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
    elif args.arch == "vgg16_nobn":
        model = net_nobn.DeepMattingNobn(args)
        model.apply(weight_init)
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


def gen_simple_alpha_loss(alpha, trimap, pred_mattes):
    weighted = torch.zeros(trimap.shape).cuda()
    weighted[trimap == 128] = 1.
    alpha_f = alpha / 255.
    diff = pred_mattes - alpha_f
    diff = diff * weighted
    alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
    alpha_loss_weighted = alpha_loss.sum() / (weighted.sum() + 1e-6)

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
    alpha_loss = (alpha_loss * t_wi).sum() / (unknown_region_size + 1e-6)

    # composite rgb loss
    pred_mattes_3 = torch.cat((pred_mattes, pred_mattes, pred_mattes), 1)
    comp = pred_mattes_3 * fg + (1. - pred_mattes_3) * bg
    comp_loss = torch.sqrt((comp - img) ** 2 + 1e-12) / 255.
    comp_loss = (comp_loss * t3_wi).sum() / (unknown_region_size + 1e-6) / 3.

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
    alpha_loss = (alpha_loss * t_wi).sum() / (unknown_region_size + 1e-6)
    
    return alpha_loss


def train(args, model, optimizer, train_loader, epoch):
    model.train()
    t0 = time.time()
    assert(args.stage in [1, 2, 3])
    #fout = open("train_loss.txt",'w')
    for iteration, batch in enumerate(train_loader, 1):
        img = Variable(batch[0])
        alpha = Variable(batch[1])
        fg = Variable(batch[2])
        bg = Variable(batch[3])
        trimap = Variable(batch[4])
        grad = Variable(batch[5])
        img_info = batch[-1]

        if args.cuda:
            img = img.cuda()
            alpha = alpha.cuda()
            fg = fg.cuda()
            bg = bg.cuda()
            trimap = trimap.cuda()
            grad = grad.cuda()

        #print("Shape: Img:{} Alpha:{} Fg:{} Bg:{} Trimap:{}".format(img.shape, alpha.shape, fg.shape, bg.shape, trimap.shape))
        #print("Val: Img:{} Alpha:{} Fg:{} Bg:{} Trimap:{} Img_info".format(img, alpha, fg, bg, trimap, img_info))

        adjust_learning_rate(args, optimizer, epoch)
        optimizer.zero_grad()

        if args.in_chan == 3:
            pred_mattes, pred_alpha = model(img)
        else:
            if args.addGrad:
                pred_mattes, pred_alpha = model(torch.cat((img, trimap, grad), 1))
            else:
                pred_mattes, pred_alpha = model(torch.cat((img, trimap), 1))


        if args.stage == 1:
            # stage1 loss
            alpha_loss, comp_loss = gen_loss(img, alpha, fg, bg, trimap, pred_mattes)
            loss = alpha_loss * args.wl_weight + comp_loss * (1. - args.wl_weight)
            #loss = gen_simple_alpha_loss(alpha, trimap, pred_mattes)
            #alpha_loss = loss
            #comp_loss = loss
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
        #fout.write("{:.5f} {} {}\n".format(loss.data[0], img_info[0][0], img_info[1][0]))
        #fout.flush()
    #fout.close()


def compute_gradient(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    grad = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    grad=cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    return grad

def test(args, model):
    model.eval()
    sample_set = []
    img_ids = os.listdir(args.testImgDir)
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
        # resize for network input, to Tensor
        scale_img = cv2.resize(img, (args.size_w, args.size_h), interpolation=cv2.INTER_LINEAR)
        scale_grad = compute_gradient(scale_img)
        scale_trimap = cv2.resize(trimap, (args.size_w, args.size_h), interpolation=cv2.INTER_LINEAR)

        tensor_img = torch.from_numpy(scale_img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
        tensor_trimap = torch.from_numpy(scale_trimap.astype(np.float32)[np.newaxis, np.newaxis, :, :])
        tensor_grad = torch.from_numpy(scale_grad.astype(np.float32)[np.newaxis, np.newaxis, :, :])
    
        cur += 1
        if cur % 100 == 0:
            print('[{}/{}] {}'.format(cur, cnt, img_info[0]))
        if args.cuda:
            tensor_img = tensor_img.cuda()
            tensor_trimap = tensor_trimap.cuda()
            tensor_grad = tensor_grad.cuda()
        #print('Img Shape:{} Trimap Shape:{}'.format(img.shape, trimap.shape))
        assert(args.stage in [1, 2, 3])
        if args.in_chan == 3:
            input_t = tensor_img
        else:
            if args.addGrad:
                input_t = torch.cat((tensor_img, tensor_trimap, tensor_grad), 1)
            else:
                input_t = torch.cat((tensor_img, tensor_trimap), 1)

        # forward
        if args.stage == 1:
            # stage 1
            pred_mattes, _ = model(input_t)
        else:
            # stage 2, 3
            _, pred_mattes = model(input_t)
        pred_mattes = pred_mattes.data
        if args.cuda:
            pred_mattes = pred_mattes.cpu()
        pred_mattes = pred_mattes.numpy()[0, 0, :, :]

        # resize to origin size
        origin_pred_mattes = cv2.resize(pred_mattes, (img_info[2], img_info[1]), interpolation = cv2.INTER_LINEAR)
        assert(origin_pred_mattes.shape == trimap.shape)

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
            if cur % 100 == 0:
                print("sad:{} mse:{}".format(sad_diff, mse_diff))

        origin_pred_mattes = (origin_pred_mattes * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.testResDir, img_info[0]), origin_pred_mattes)

    print("Avg-Cost: {} s/image".format((time.time() - t0) / cnt))
    if args.testAlphaDir != '':
        print("Eval-MSE: {}".format(mse_diffs / cnt))
        print("Eval-SAD: {}".format(sad_diffs / cnt))


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
        if epoch > 0 and args.testFreq > 0 and epoch % args.testFreq == 0:
            test(args, model)


if __name__ == "__main__":
    main()
