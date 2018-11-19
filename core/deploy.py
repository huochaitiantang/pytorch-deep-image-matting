import torch
import argparse
import torch.nn as nn
import net
import net_nobn
import resnet_aspp
import cv2
import os
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import time

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--size_h', type=int, required=True, help="height size of input image")
    parser.add_argument('--size_w', type=int, required=True, help="width size of input image")
    parser.add_argument('--imgDir', type=str, required=True, help="directory of image")
    parser.add_argument('--trimapDir', type=str, required=True, help="directory of trimap")
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--resume', type=str, required=True, help="checkpoint that model resume from")
    parser.add_argument('--saveDir', type=str, required=True, help="where prediction result save to")
    parser.add_argument('--alphaDir', type=str, default='', help="directory of gt")
    parser.add_argument('--stage', type=int, required=True, help="backbone stage")
    parser.add_argument('--not_strict', action='store_true', help='not copy ckpt strict?')
    parser.add_argument('--arch', type=str, required=True, choices=["vgg16","vgg16_nobn", "resnet50_aspp"], help="net backbone")
    parser.add_argument('--in_chan', type=int, default=4, choices=[3, 4], help="input channel 3(no trimap) or 4")
    parser.add_argument('--bilateralfilter', action='store_true', help='use bilateralfilter before image input?')
    parser.add_argument('--guidedfilter', action='store_true', help='use guidedfilter after prediction?')
    parser.add_argument('--addGrad', action='store_true', help='use grad as a input channel?')
    args = parser.parse_args()
    print(args)
    return args

def gen_dataset(imgdir, trimapdir):
        sample_set = []
        img_ids = os.listdir(imgdir)
        cnt = len(img_ids)
        cur = 1
        for img_id in img_ids:
            img_name = os.path.join(imgdir, img_id)
            trimap_name = os.path.join(trimapdir, img_id)

            assert(os.path.exists(img_name))
            assert(os.path.exists(trimap_name))

            sample_set.append((img_name, trimap_name))

        return sample_set

def compute_gradient(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    grad = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    grad=cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    return grad


def main():

    print("===> Loading args")
    args = get_args()

    print("===> Environment init")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    if args.arch == "resnet50_aspp":
        model = resnet_aspp.resnet50(args)
    elif args.arch == "vgg16_nobn":
        model = net_nobn.DeepMattingNobn(args)
    else:     
        model = net.DeepMatting(args)
    ckpt = torch.load(args.resume)
    if args.not_strict:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt['state_dict'], strict=True)

    if args.cuda:
        model = model.cuda()

    print("===> Load dataset")
    dataset = gen_dataset(args.imgDir, args.trimapDir)

    mse_diffs = 0.
    sad_diffs = 0.
    cnt = len(dataset)
    cur = 0
    t0 = time.time()
    for img_path, trimap_path in dataset:
        img = cv2.imread(img_path)
        trimap = cv2.imread(trimap_path)[:, :, 0]

        assert(img.shape[:2] == trimap.shape[:2])

        img_info = (img_path.split('/')[-1], img.shape[0], img.shape[1])
        # resize for network input, to Tensor
        scale_img = cv2.resize(img, (args.size_w, args.size_h), interpolation=cv2.INTER_LINEAR)
        scale_trimap = cv2.resize(trimap, (args.size_w, args.size_h), interpolation=cv2.INTER_LINEAR)

        if args.bilateralfilter:
            #cv2.imwrite("result/debug/{}_before.png".format(img_info[0][:-4]), scale_img)
            scale_img = cv2.bilateralFilter(scale_img, d=9, sigmaColor=100, sigmaSpace=100)
            #cv2.imwrite("result/debug/{}_after.png".format(img_info[0][:-4]), scale_img)

        scale_grad = compute_gradient(scale_img)
        tensor_img = torch.from_numpy(scale_img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
        tensor_trimap = torch.from_numpy(scale_trimap.astype(np.float32)[np.newaxis, np.newaxis, :, :])
        tensor_grad = torch.from_numpy(scale_grad.astype(np.float32)[np.newaxis, np.newaxis, :, :])
    
        cur += 1
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

        # origin trimap 
        pixel = float((trimap == 128).sum())
        
        # eval if gt alpha is given
        if args.alphaDir != '':
            alpha_name = os.path.join(args.alphaDir, img_info[0])
            assert(os.path.exists(alpha_name))
            alpha = cv2.imread(alpha_name)[:, :, 0] / 255.
            assert(alpha.shape == origin_pred_mattes.shape)

            #x1 = (alpha[trimap == 255] == 1.0).sum() # x3
            #x2 = (alpha[trimap == 0] == 0.0).sum() # x5
            #x3 = (trimap == 255).sum()
            #x4 = (trimap == 128).sum()
            #x5 = (trimap == 0).sum()
            #x6 = trimap.size # sum(x3,x4,x5)
            #x7 = (alpha[trimap == 255] < 1.0).sum() # 0
            #x8 = (alpha[trimap == 0] > 0).sum() #

            #print(x1, x2, x3, x4, x5, x6, x7, x8)
            #assert(x1 == x3)
            #assert(x2 == x5)
            #assert(x6 == x3 + x4 + x5)
            #assert(x7 == 0)
            #assert(x8 == 0)

            mse_diff = ((origin_pred_mattes - alpha) ** 2).sum() / pixel
            sad_diff = np.abs(origin_pred_mattes - alpha).sum()
            mse_diffs += mse_diff
            sad_diffs += sad_diff
            print("sad:{} mse:{}".format(sad_diff, mse_diff))

        origin_pred_mattes = (origin_pred_mattes * 255).astype(np.uint8)
        res = origin_pred_mattes.copy()
        if args.guidedfilter:
            radius = 50
            eps = 1e-6
            GF = cv2.ximgproc.createGuidedFilter(img, radius, eps)
            GF.filter(origin_pred_mattes, res)
            
        # only attention unknown region
        res[trimap == 255] = 255
        res[trimap == 0  ] = 0

        cv2.imwrite(os.path.join(args.saveDir, img_info[0]), res)

    print("Avg-Cost: {} s/image".format((time.time() - t0) / cnt))
    if args.alphaDir != '':
        print("Eval-MSE: {}".format(mse_diffs / cur))
        print("Eval-SAD: {}".format(sad_diffs / cur))

if __name__ == "__main__":
    main()
