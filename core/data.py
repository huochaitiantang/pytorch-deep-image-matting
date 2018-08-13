import torch
import cv2
import os
import random
import numpy as np
from torchvision import transforms

def gen_trimap(alpha):
    k_size = random.choice(range(10, 20))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    dilated = cv2.dilate(alpha, kernel)
    #eroded = cv2.erode(alpha, kernel)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    delta = 5
    undelta = 255 - delta
    trimap[np.where(dilated >= undelta)] = 255
    trimap[np.where(dilated <= delta  )] = 0
    return trimap


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
            cropx, cropy = target[0][rand_ind], target[1][rand_ind]
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


class MatDataset(torch.utils.data.Dataset):
    def __init__(self, imgdir, alphadir, fgdir, bgdir, size_h, size_w, crop_h, crop_w, transform=None):
        self.sample_set=[]
        self.transform = transform
        self.size_h = size_h
        self.size_w = size_w
        self.crop_h = crop_h
        self.crop_w = crop_w
        img_ids = os.listdir(imgdir)
        cnt = len(img_ids)
        cur = 1
        for img_id in img_ids:
            img_name = '{}/{}'.format(imgdir, img_id)
            alpha_name = '{}/{}'.format(alphadir, img_id)
            fg_name = '{}/{}'.format(fgdir, img_id)
            bg_name = '{}/{}'.format(bgdir, img_id)

            assert(os.path.exists(img_name))
            assert(os.path.exists(alpha_name))
            assert(os.path.exists(fg_name))
            assert(os.path.exists(bg_name))

            self.sample_set.append((img_name, alpha_name, fg_name, bg_name, img_id))

        print('\t--Valid Samples: {}'.format(len(self.sample_set)))


    def __getitem__(self,index):
        img_name, alpha_name, fg_name, bg_name, img_id = self.sample_set[index]
        img = cv2.imread(img_name)
        fg = cv2.imread(fg_name)
        bg = cv2.imread(bg_name)
        alpha = cv2.imread(alpha_name)[:, :, 0]
        h, w = img.shape[0], img.shape[1]
        img_info = (img_id, h, w)

        # img size is too small
        if h < self.crop_h or w < self.crop_w:
            upratio = max((self.crop_h + 1)/float(h), (self.crop_w + 1)/float(w))
            new_w, new_h = int(w * upratio), int(h * upratio)
            assert(new_w >= self.crop_w and new_h >= self.crop_w)
            # resize
            img   = cv2.resize(img,   (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            fg    = cv2.resize(fg,    (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            bg    = cv2.resize(bg,    (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        trimap = gen_trimap(alpha)

        # random crop(crop_h, crop_w) and flip
        if self.transform:
            img, alpha, fg, bg, trimap = self.transform(img, alpha, fg, bg, trimap, self.crop_h, self.crop_w)

        # resize to (size_h, size_w)
        if self.size_h != img.shape[0] or self.size_w != img.shape[1]:
            # resize
            img   =cv2.resize(img,    (self.size_w, self.size_h), interpolation=cv2.INTER_LINEAR)
            fg    =cv2.resize(fg,     (self.size_w, self.size_h), interpolation=cv2.INTER_LINEAR)
            bg    =cv2.resize(bg,     (self.size_w, self.size_h), interpolation=cv2.INTER_LINEAR)
            alpha =cv2.resize(alpha,  (self.size_w, self.size_h), interpolation=cv2.INTER_LINEAR)
            trimap=cv2.resize(trimap, (self.size_w, self.size_h), interpolation=cv2.INTER_LINEAR)
       
        #cv2.imwrite("result/debug/{}_img.png".format(img_info[0]), img)
        #cv2.imwrite("result/debug/{}_alpha.png".format(img_info[0]), alpha)
        #cv2.imwrite("result/debug/{}_fg.png".format(img_info[0]), fg)
        #cv2.imwrite("result/debug/{}_bg.png".format(img_info[0]), bg)
        #cv2.imwrite("result/debug/{}_trimap.png".format(img_info[0]), trimap)

        alpha = torch.from_numpy(alpha.astype(np.float32)[np.newaxis, :, :])
        trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :])
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        fg = torch.from_numpy(fg.astype(np.float32)).permute(2, 0, 1)
        bg = torch.from_numpy(bg.astype(np.float32)).permute(2, 0, 1)

        return img, alpha, fg, bg, trimap, img_info
    
    def __len__(self):
        return len(self.sample_set)
