import torch
import cv2
import os
import random
import numpy as np
from torchvision import transforms

def gen_trimap(alpha):
    k_size = random.choice(range(1, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(alpha, kernel, iterations=np.random.randint(1, 20))
    #eroded = cv2.erode(alpha, kernel)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[alpha >= 255] = 255
    trimap[dilated <= 0] = 0

    return trimap


def compute_gradient(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    grad = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    grad=cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    return grad

class MatTransform(object):
    def __init__(self, flip=False):
        self.flip = flip

    def __call__(self, img, alpha, fg, bg, trimap, crop_h, crop_w):
        h, w = alpha.shape

        # random crop in the unknown region center
        target = np.where(trimap == 128)
        cropx, cropy = 0, 0
        if len(target[0]) > 0:
            rand_ind = np.random.randint(len(target[0]), size = 1)[0]
            cropy, cropx = target[0][rand_ind], target[1][rand_ind]
            cropx = min(max(cropx, 0), w - crop_w)
            cropy = min(max(cropy, 0), h - crop_h)

        img    = img   [cropy : cropy + crop_h, cropx : cropx + crop_w]
        fg     = fg    [cropy : cropy + crop_h, cropx : cropx + crop_w]
        bg     = bg    [cropy : cropy + crop_h, cropx : cropx + crop_w]
        alpha  = alpha [cropy : cropy + crop_h, cropx : cropx + crop_w]
        trimap = trimap[cropy : cropy + crop_h, cropx : cropx + crop_w]

        # random flip
        if self.flip and random.random() < 0.5:
            img = cv2.flip(img, 1)
            alpha = cv2.flip(alpha, 1)
            fg = cv2.flip(fg, 1)
            bg = cv2.flip(bg, 1)
            trimap = cv2.flip(trimap, 1)

        return img, alpha, fg, bg, trimap


def get_files(mydir):
    res = []
    for root, dirs, files in os.walk(mydir, followlinks=True):
        for f in files:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG"):
                res.append(os.path.join(root, f))
    return res


# Dataset not composite online
class MatDatasetOffline(torch.utils.data.Dataset):
    def __init__(self, args, transform=None, normalize=None):
        self.samples=[]
        self.transform = transform
        self.normalize = normalize
        self.args = args
        self.size_h = args.size_h
        self.size_w = args.size_w
        self.crop_h = args.crop_h
        self.crop_w = args.crop_w
        assert(len(self.crop_h) == len(self.crop_w))
        
        fg_paths = get_files(self.args.fgDir)

        self.cnt = len(fg_paths)

        for fg_path in fg_paths:
            alpha_path = fg_path.replace(self.args.fgDir, self.args.alphaDir)
            img_path = fg_path.replace(self.args.fgDir, self.args.imgDir)
            bg_path = fg_path.replace(self.args.fgDir, self.args.bgDir)
            assert(os.path.exists(alpha_path))
            assert(os.path.exists(fg_path))
            assert(os.path.exists(bg_path))
            assert(os.path.exists(img_path))
            self.samples.append((alpha_path, fg_path, bg_path, img_path))
        print("\t--Valid Samples: {}".format(self.cnt))
        assert(self.cnt > 0)

    def __getitem__(self,index):
        alpha_path, fg_path, bg_path, img_path = self.samples[index]

        img_info = [fg_path, alpha_path, bg_path, img_path]

        # read fg, alpha
        fg = cv2.imread(fg_path)[:, :, :3]
        bg = cv2.imread(bg_path)[:, :, :3]
        img = cv2.imread(img_path)[:, :, :3]
        alpha = cv2.imread(alpha_path)[:, :, 0]

        assert(bg.shape == fg.shape and bg.shape == img.shape)
        img_info.append(fg.shape)
        bh, bw, bc, = fg.shape

        rand_ind = random.randint(0, len(self.crop_h) - 1)
        cur_crop_h = self.crop_h[rand_ind]
        cur_crop_w = self.crop_w[rand_ind]

        # if ratio!=1: make the img (h==croph and w>=cropw)or(w==cropw and h>=croph)
        wratio = float(cur_crop_w) / bw
        hratio = float(cur_crop_h) / bh
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            nbw = int(bw * ratio + 1.0)
            nbh = int(bh * ratio + 1.0)
            fg = cv2.resize(fg, (nbw, nbh), interpolation=cv2.INTER_LINEAR)
            bg = cv2.resize(bg, (nbw, nbh), interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(img, (nbw, nbh), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(alpha, (nbw, nbh), interpolation=cv2.INTER_LINEAR)
        trimap = gen_trimap(alpha)

        # random crop(crop_h, crop_w) and flip
        if self.transform:
            img, alpha, fg, bg, trimap = self.transform(img, alpha, fg, bg, trimap, cur_crop_h, cur_crop_w)

        # resize to (size_h, size_w)
        if self.size_h != img.shape[0] or self.size_w != img.shape[1]:
            # resize
            img   =cv2.resize(img,    (self.size_w, self.size_h), interpolation=cv2.INTER_LINEAR)
            fg    =cv2.resize(fg,     (self.size_w, self.size_h), interpolation=cv2.INTER_LINEAR)
            bg    =cv2.resize(bg,     (self.size_w, self.size_h), interpolation=cv2.INTER_LINEAR)
            alpha =cv2.resize(alpha,  (self.size_w, self.size_h), interpolation=cv2.INTER_LINEAR)

        trimap = gen_trimap(alpha)
        grad = compute_gradient(img)

        if self.normalize:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # first, 0-255 to 0-1
            # second, x-mean/std and HWC to CHW
            img_norm = self.normalize(img_rgb)
        else:
            img_norm = None

        #img_id = img_info[0].split('/')[-1]
        #cv2.imwrite("result/debug/{}_img.png".format(img_id), img)
        #cv2.imwrite("result/debug/{}_alpha.png".format(img_id), alpha)
        #cv2.imwrite("result/debug/{}_fg.png".format(img_id), fg)
        #cv2.imwrite("result/debug/{}_bg.png".format(img_id), bg)
        #cv2.imwrite("result/debug/{}_trimap.png".format(img_id), trimap)
        #cv2.imwrite("result/debug/{}_grad.png".format(img_id), grad)
        alpha = torch.from_numpy(alpha.astype(np.float32)[np.newaxis, :, :])
        trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :])
        grad = torch.from_numpy(grad.astype(np.float32)[np.newaxis, :, :])
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        fg = torch.from_numpy(fg.astype(np.float32)).permute(2, 0, 1)
        bg = torch.from_numpy(bg.astype(np.float32)).permute(2, 0, 1)

        return img, alpha, fg, bg, trimap, grad, img_norm, img_info
    
    def __len__(self):
        return len(self.samples)
