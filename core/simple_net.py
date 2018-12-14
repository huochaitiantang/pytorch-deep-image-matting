import torch
import torch.nn as nn
import math
import cv2
import torch.nn.functional as F

class DeepMattingSimple(nn.Module):
    def __init__(self, args):
        super(DeepMattingSimple, self).__init__()

        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride = 1, padding=1,bias=True)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride = 1, padding=1,bias=True)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride = 1, padding=1,bias=True)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, stride = 1, padding=1,bias=True)
        self.conv5 = nn.Conv2d(8, 1, kernel_size=3, stride = 1, padding=1,bias=True)


    def forward(self, x):
        # Stage 1
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        pred_mattes = F.sigmoid(self.conv5(x4))

        return pred_mattes, 0
