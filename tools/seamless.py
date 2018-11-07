import os
import cv2
import time
import numpy as np

fg_dir = "/home/liuliang/Desktop/dataset/matting/xuexin/complex"
trimap_dir = "/home/liuliang/Desktop/dataset/matting/xuexin/complex_trimap_deeplabv3"
#trimap_dir = "/home/liuliang/Desktop/dataset/matting/xuexin/complex_segmentation_deeplabv3_png"
save_dir = "/home/liuliang/Desktop/dataset/matting/xuexin/tmp"

fgs = os.listdir(fg_dir)
cnt = len(fgs)
print("Cnt:{}".format(cnt))
t0 = time.time()

def get_corner(mask):
    x1 = y1 = 0
    h , w = mask.shape
    x2 = w - 1
    y2 = h - 1
    hs = np.sum(mask, axis = 0) # len = w
    ws = np.sum(mask, axis = 1) # len = h
    x = np.where(hs > 0) 
    y = np.where(ws > 0)

    x1 = x[0][0] + 1
    x2 = x[0][-1] + 1
    y1 = y[0][0] + 1
    y2 = y[0][-1] + 1
    print(x1, y1, x2, y2)

    return x1, y1, x2, y2

for k in range(cnt):
    print("{}".format(k + 1))
    fg = fgs[k]
    fg_path = os.path.join(fg_dir, fg)
    #trimap_path = os.path.join(trimap_dir, fg).replace(".JPG", ".png")
    trimap_path = os.path.join(trimap_dir, fg)
    assert(os.path.exists(fg_path) and os.path.exists(trimap_path))

    src = cv2.imread(fg_path)
    trimap = cv2.imread(trimap_path)
    h, w, c = src.shape
    
    mask =  np.zeros_like(src)
    mask[trimap >= 128] = 255
    #mask = trimap
    #cv2.imwrite(os.path.join(save_dir, "mask.png"), mask)

    x1, y1, x2, y2 = get_corner(mask[:, :, 0])
    center = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))

    assert(src.shape == trimap.shape and src.shape == mask.shape)
    
    bg = np.zeros((h + 10, w + 10, c), src.dtype)
    bg[:, :, 2] = 67
    bg[:, :, 1] = 142
    bg[:, :, 0] = 219

    #center = (int(w / 2), int(h / 2))
    #center = (0, 0)
    print(src.shape, bg.shape, center)

    output = cv2.seamlessClone(src, bg, mask, center, cv2.NORMAL_CLONE)
    output = output[5: h + 5, 5: w + 5, :]
    #cv2.imwrite(os.path.join(save_dir, "output.png"), output)
    #cv2.imwrite(os.path.join(save_dir, "src.png"), src)

    #k_size = 3
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    #eroded = cv2.erode(mask, kernel, iterations=10)
    #cv2.imwrite(os.path.join(save_dir, "eroded.png"), eroded)
    mask =  np.zeros_like(src)
    mask[trimap >= 255] = 255

    alpha_f =  mask / 255.
    output = alpha_f * src + (1. - alpha_f) * output

    save_path = os.path.join(save_dir, fg).replace(".JPG", ".png")
    cv2.imwrite(save_path, output)
    #break

t1 = time.time()

print("Avg Cost: {}".format((t1 - t0) / cnt))
