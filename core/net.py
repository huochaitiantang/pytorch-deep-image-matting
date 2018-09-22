import torch
import torch.nn as nn
import math
import cv2
import torch.nn.functional as F

class DeepMatting(nn.Module):
    def __init__(self, args):
        super(DeepMatting, self).__init__()
        batchNorm_momentum = 0.1
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3,stride = 1, padding=1,bias=True)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding=1,bias=True)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1,bias=True)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=True)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=True)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=True)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=True)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=True)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=1, padding=0,bias=True)
        self.bn61 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        
        self.deconv6_1 = nn.Conv2d(512, 512, kernel_size=1,bias=True)
        self.bn61d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.deconv5_1 = nn.Conv2d(512, 512, kernel_size=5, padding=2,bias=True)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.deconv4_1 = nn.Conv2d(512, 256, kernel_size=5, padding=2,bias=True)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.deconv3_1 = nn.Conv2d(256, 128, kernel_size=5, padding=2,bias=True)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.deconv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2,bias=True)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2,bias=True)
        self.bn11d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        
        self.deconv1 = nn.Conv2d(64, 1, kernel_size=5, padding=2,bias=True)

        assert(args.stage in [1, 2, 3])
        if args.stage == 2:
            # for stage2 training
            for p in self.parameters():
                p.requires_grad=False
        
        self.refine_conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=True)
        self.refine_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.refine_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.refine_pred = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv1_1(x)))
        x12 = F.relu(self.bn12(self.conv1_2(x11)))
        x1p, id1 = F.max_pool2d(x12,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv2_1(x1p)))
        x22 = F.relu(self.bn22(self.conv2_2(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv3_1(x2p)))
        x32 = F.relu(self.bn32(self.conv3_2(x31)))
        x33 = F.relu(self.bn33(self.conv3_3(x32)))
        x3p, id3 = F.max_pool2d(x33,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv4_1(x3p)))
        x42 = F.relu(self.bn42(self.conv4_2(x41)))
        x43 = F.relu(self.bn43(self.conv4_3(x42)))
        x4p, id4 = F.max_pool2d(x43,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv5_1(x4p)))
        x52 = F.relu(self.bn52(self.conv5_2(x51)))
        x53 = F.relu(self.bn53(self.conv5_3(x52)))
        x5p, id5 = F.max_pool2d(x53,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 6
        x61 = F.relu(self.bn61(self.conv6_1(x5p)))

        # Stage 6d
        x61d = F.relu(self.bn61d(self.deconv6_1(x61)))

        # Stage 5d
        x5d = F.max_unpool2d(x61d,id5, kernel_size=2, stride=2)
        x51d = F.relu(self.bn51d(self.deconv5_1(x5d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x41d = F.relu(self.bn41d(self.deconv4_1(x4d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x31d = F.relu(self.bn31d(self.deconv3_1(x3d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x21d = F.relu(self.bn21d(self.deconv2_1(x2d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn11d(self.deconv1_1(x1d)))

        # Should add sigmoid? github repo add so.
        raw_alpha = self.deconv1(x12d)
        pred_mattes = F.sigmoid(raw_alpha)

        # Stage2 refine conv1
        refine0 = torch.cat((x[:, :3, :, :], pred_mattes * 256),  1)
        refine1 = F.relu(self.refine_conv1(refine0))
        refine2 = F.relu(self.refine_conv2(refine1))
        refine3 = F.relu(self.refine_conv3(refine2))
        # Should add sigmoid?
        # sigmoid lead to refine result all converge to 0... 
        #pred_refine = F.sigmoid(self.refine_pred(refine3))
        pred_refine = self.refine_pred(refine3)

        pred_alpha = F.sigmoid(raw_alpha + pred_refine)

        #print(pred_mattes.mean(), pred_alpha.mean(), pred_refine.sum())

        return pred_mattes, pred_alpha
