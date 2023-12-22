"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

from taming.util import get_ckpt_path


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        # self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = torch.hub.load("Warvito/MedicalNet-models", model="medicalnet_resnet10_23datasets", verbose=False,)
        # self.conv1 = nn.Conv3d(1, self.chns[0], kernel_size=(7,7,7), stride=(2,2,2), padding=(3,3,3), bias=False)
        # self.bn1 = nn.BatchNorm3d(self.chns[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # self.lin1 = BasicBlock(self.chns[0],self.chns[1]) #NetLinLayer(self.chns[1], use_dropout=use_dropout)
        # self.lin2 = BasicBlock(self.chns[1],self.chns[2], downsample=True) #NetLinLayer(self.chns[2], use_dropout=use_dropout)
        # self.lin3 = BasicBlock(self.chns[2],self.chns[3], downsample=True) #NetLinLayer(self.chns[3], use_dropout=use_dropout)
        # self.lin4 = BasicBlock(self.chns[3],self.chns[4], downsample=True) #NetLinLayer(self.chns[4], use_dropout=use_dropout)
        # self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False
            

    # def load_from_pretrained(self, name="vgg_lpips"):
    #     # ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
    #     # print(torch.load(ckpt, map_location=torch.device("cpu")))
    #     # print(torch.hub.load("Warvito/MedicalNet-models", 
    #     #                    model="medicalnet_resnet10_23datasets", 
    #     #                    verbose=False,).state_dict())
    #     # self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
    #     pretrained_dict = torch.hub.load("Warvito/MedicalNet-models", model="medicalnet_resnet10_23datasets", verbose=False).state_dict()

    #     model_dict = self.state_dict()

    #     mapping = {
    #         "layer1.0": "lin1",
    #         "layer2.0": "lin2",
    #         "layer3.0": "lin3",
    #         "layer4.0": "lin4"
    #     }

    #     pretrained_dict = {f"{mapping[k] if k in mapping else k}{sub_k}": v 
    #                for k, v in pretrained_dict.items() 
    #                for sub_k in model_dict 
    #                if f"{mapping[k] if k in mapping else k}{sub_k}" in model_dict}

    #     model_dict.update(pretrained_dict)

    #     self.load_state_dict(model_dict)
        # self.load_state_dict(
        #     torch.hub.load("Warvito/MedicalNet-models", 
        #                    model="medicalnet_resnet10_23datasets", 
        #                    verbose=False,).state_dict())
        # print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        # ckpt = get_ckpt_path(name)
        # model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        model.load_state_dict(
            torch.hub.load("Warvito/MedicalNet-models", 
                           model="medicalnet_resnet10_23datasets", 
                           verbose=False,).state_dict(), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = normalize_tensor(input), normalize_tensor(target) #(self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        # feats0, feats1, diffs = {}, {}, {}
        # lin0 = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
        # lins = [lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        # outs0.append(lin0(in0_input))
        # outs1.append(lin0(in1_input))
        # for kk in range(1,len(self.chns)):
        #     outs0.append(lins[kk](outs0[kk-1]))
        #     outs1.append(lins[kk](outs1[kk-1]))
        # for kk in range(len(self.chns)):
        #     feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
        #     diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        # res = [spatial_average_3d(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        feats0, feats1 = normalize_tensor(outs0), normalize_tensor(outs1)
        diffs = (feats0 - feats1) ** 2
        res = spatial_average_3d(diffs, keepdim=True)
        val = res
        # for l in range(1, len(self.chns)):
            # val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

class BasicBlock(nn.Module):
    def __init__(self, chn_in, chn_out, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(chn_in, chn_out, kernel_size=3, stride=stride, padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm3d(chn_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(chn_out, chn_out, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm3d(chn_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if downsample is True:
            self.downsample = nn.Sequential(
                nn.Conv3d(chn_in, chn_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm3d(chn_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv3d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)


def spatial_average_3d(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3, 4], keepdim=keepdim)