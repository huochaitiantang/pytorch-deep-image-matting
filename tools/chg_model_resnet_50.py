import torch
import torchvision
import collections
import os

HOME = os.environ['HOME']
model_path = "{}/.torch/models/resnet50-19c8e357.pth".format(HOME)
if not os.path.exists(model_path):
    model = torchvision.models.vgg16(pretrained=True)
assert(os.path.exists(model_path))

x = torch.load(model_path)
val = collections.OrderedDict()

for key in x.keys():
    val[key] = x[key]
val['conv1.weight'] = torch.cat((x['conv1.weight'], torch.zeros(64, 1, 7, 7)), 1)

print(x['conv1.weight'].shape)
print(val['conv1.weight'].shape)

y = {}
y['state_dict'] = val
y['epoch'] = 0
if not os.path.exists('./model'):
    os.makedirs('./model')
torch.save(y, './model/resnet50_state_dict.pth')
