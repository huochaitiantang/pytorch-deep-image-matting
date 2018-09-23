import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# upsample x like size of y
def F_upsample(x, y):
    N, C, H, W = y.shape
    return F.upsample(x, size=(H, W), mode='bilinear', align_corners=True)


class ResNet(nn.Module):

    def __init__(self, block, layers, args):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        # deeplabv3+ atrous spatial pyramid pooling
        feat_chan = 1024
        depth = 256
        atrous_rates = [6, 12, 18]
        self.low_level_conv = nn.Conv2d(64, depth, kernel_size=1, stride=1, padding=0)
        self.aspp_conv1x1   = nn.Conv2d(feat_chan, depth, kernel_size=1, stride=1, padding=0)
        self.aspp_conv3x3_1 = nn.Conv2d(feat_chan, depth, kernel_size=3, stride=1, padding=atrous_rates[0], dilation=atrous_rates[0])
        self.aspp_conv3x3_2 = nn.Conv2d(feat_chan, depth, kernel_size=3, stride=1, padding=atrous_rates[1], dilation=atrous_rates[1])
        self.aspp_conv3x3_3 = nn.Conv2d(feat_chan, depth, kernel_size=3, stride=1, padding=atrous_rates[2], dilation=atrous_rates[2])
        self.aspp_pooling_conv = nn.Conv2d(feat_chan, depth, kernel_size=1, stride=1, padding=0)
        self.aspp_cat_conv = nn.Conv2d(depth * 5, depth, kernel_size=1, stride=1, padding=0)

        num_classes = 1
        self.pred_conv1 = nn.Conv2d(depth * 2, 256, kernel_size=3, stride=1, padding=1)
        self.pred_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pred_conv3 = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1)
        #self.pred_trimap = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1, bias=False)

        assert(args.stage in [1, 2, 3])
        if args.stage == 2:
            # for stage2 training
            for p in self.parameters():
                p.requires_grad=False
        
        self.refine_conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=True)
        self.refine_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.refine_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.refine_pred = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        c0 = x
        x = self.new_conv1(c0) # stride 2
        x = self.bn1(x)
        c1 = self.relu(x)
        c2 = self.maxpool(c1) # stride 4
        x = self.layer1(c2)
        c3 = self.layer2(x) # stride 8
        c4 = self.layer3(c3) # stride 16

        # aspp feature 1/16
        aspp_conv1 = self.aspp_conv1x1(c4)
        aspp_conv3_1 = self.aspp_conv3x3_1(c4)
        aspp_conv3_2 = self.aspp_conv3x3_2(c4)
        aspp_conv3_3 = self.aspp_conv3x3_3(c4)
        # global avg pooling
        aspp_pooling = F.avg_pool2d(c4, kernel_size=c4.size()[2:])
        aspp_pooling = self.aspp_pooling_conv(aspp_pooling)
        aspp_pooling = F_upsample(aspp_pooling, c4)

        # pyramid fusion
        aspp = torch.cat((aspp_conv1, aspp_conv3_1, aspp_conv3_2, aspp_conv3_3, aspp_pooling), 1)
        aspp = self.aspp_cat_conv(aspp)

        # cat with low feature 1/4
        low_feat = self.low_level_conv(c2)
        aspp_up = F_upsample(aspp, low_feat)
        cat_feat = torch.cat((aspp_up, low_feat), 1)

        # pred raw alpha 1/1
        pred_conv1 = self.pred_conv1(cat_feat)
        pred_conv2 = self.pred_conv2(pred_conv1)
        raw_alpha = F_upsample(self.pred_conv3(pred_conv2), c0)
        pred_mattes = F.sigmoid(raw_alpha)

        # Stage2 refine conv1
        refine0 = torch.cat((c0[:, :3, :, :], pred_mattes * 256),  1)
        refine1 = F.relu(self.refine_conv1(refine0))
        refine2 = F.relu(self.refine_conv2(refine1))
        refine3 = F.relu(self.refine_conv3(refine2))
        # Should add sigmoid?
        # sigmoid lead to refine result all converge to 0... 
        #pred_refine = F.sigmoid(self.refine_pred(refine3))
        pred_refine = self.refine_pred(refine3)

        pred_alpha = F.sigmoid(raw_alpha + pred_refine)

        return pred_mattes, pred_alpha


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(args, pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(Bottleneck, [3, 4, 6, 3], args)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
