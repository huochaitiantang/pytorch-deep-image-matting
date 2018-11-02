import os
import shutil

fin = open("../train_loss.txt")
lines = fin.readlines()
cnt = len(lines)
overthresh = 0.10
k = 0
des_dir = "/home/liuliang/Desktop/dataset/matting/big_loss_sample"

for line in lines:
    #print(line)
    loss, fg_path, alpha_path = line.split(' ')
    alpha_path = alpha_path.strip('\n')
    loss = float(loss)
    if(loss > overthresh):
        k += 1
        fg_des = "{}/image/{}.{}".format(des_dir, k, fg_path.split('.')[-1])
        alpha_des = "{}/alpha/{}.{}".format(des_dir, k, alpha_path.split('.')[-1])
        print(loss)
        print(fg_path, alpha_path)
        print(fg_des, alpha_des)
        shutil.copyfile(fg_path, fg_des)
        shutil.copyfile(alpha_path, alpha_des)
print("Ratio: {}".format(float(k) / cnt))
