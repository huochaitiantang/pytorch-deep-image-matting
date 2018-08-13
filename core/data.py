import torch
import cv2
import os
import random
import numpy as np
from torchvision import transforms

def gen_trimap(alpha):
    k_size = random.choice(range(20, 40))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    dilated = cv2.dilate(alpha, kernel)
    eroded = cv2.erode(alpha, kernel)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[np.where((dilated == 255) & (eroded == 255))] = 255
    trimap[np.where((dilated == 0) & (eroded == 0))] = 0
    return trimap


class MatTransform(object):
    def __init__(self, flip=False):
        self.flip = flip

    def __call__(self, img, alpha, fg, bg):
        if self.flip and random.random() < 0.5:
            img = cv2.flip(img, 1)
            alpha = cv2.flip(alpha, 1)
            fg = cv2.flip(fg, 1)
            bg = cv2.flip(bg, 1)
        return img, alpha, fg, bg


class MatDataset(torch.utils.data.Dataset):
    def __init__(self, imgdir, alphadir, fgdir, bgdir, size_h, size_w, transform=None):
        self.sample_set=[]
        self.transform = transform
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

            img = cv2.imread(img_name)
            scale_h = float(size_h) / img.shape[0]
            scale_w = float(size_w) / img.shape[1]
            img_info = (img_id, scale_h, scale_w)

            # resize
            img = cv2.resize(img, (size_w, size_h), interpolation=cv2.INTER_LINEAR)
            fg = cv2.imread(fg_name)
            fg = cv2.resize(fg, (size_w, size_h), interpolation=cv2.INTER_LINEAR)
            bg = cv2.imread(bg_name)
            bg = cv2.resize(bg, (size_w, size_h), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.imread(alpha_name)[:, :, 0]
            alpha = cv2.resize(alpha, (size_w, size_h), interpolation=cv2.INTER_LINEAR)

            self.sample_set.append((img, alpha, fg, bg, img_info))
            #if cur >= 100:
            #    break
            if cur % 100 == 0:
                print("\t--{}/{}".format(cur, cnt))
            cur += 1
        print('\t--Valid Samples: {}'.format(len(self.sample_set)))

    def __getitem__(self,index):
        img, alpha, fg, bg, img_info = self.sample_set[index]
        # resize and flip
        if self.transform:
            img, alpha, fg, bg = self.transform(img, alpha, fg, bg)
        trimap = gen_trimap(alpha)

        #cv2.imwrite("{}_img.png".format(img_info[0]), img)
        #cv2.imwrite("{}_alpha.png".format(img_info[0]), alpha)
        #cv2.imwrite("{}_fg.png".format(img_info[0]), fg)
        #cv2.imwrite("{}_bg.png".format(img_info[0]), bg)
        #cv2.imwrite("{}_trimap.png".format(img_info[0]), trimap)

        alpha = torch.from_numpy(alpha.astype(np.float32)[np.newaxis, :, :])
        trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :])
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        fg = torch.from_numpy(fg.astype(np.float32)).permute(2, 0, 1)
        bg = torch.from_numpy(bg.astype(np.float32)).permute(2, 0, 1)

        return img, alpha, fg, bg, trimap, img_info
    
    def __len__(self):
        return len(self.sample_set)
