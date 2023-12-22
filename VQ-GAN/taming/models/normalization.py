import re
import torch.nn as nn
import torch.nn.functional as F

class SourceNorm(nn.Module):
    def __init__(self, norm_nc):
        super().__init__()

        nhidden = 128

        self.conv_t1 = nn.Sequential(
                        nn.Conv3d(norm_nc, nhidden, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv3d(nhidden, norm_nc, kernel_size=1)
        )
        self.conv_t1ce = nn.Sequential(
                        nn.Conv3d(norm_nc, nhidden, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv3d(nhidden, norm_nc, kernel_size=1)
        )
        self.conv_t2 = nn.Sequential(
                        nn.Conv3d(norm_nc, nhidden, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv3d(nhidden, norm_nc, kernel_size=1)
        )
        self.conv_flair = nn.Sequential(
                        nn.Conv3d(norm_nc, nhidden, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv3d(nhidden, norm_nc, kernel_size=1)
        )

    def forward(self, x, source):
        if source == 't1': x = self.conv_t1(x)
        elif source == 't1ce': x = self.conv_t1ce(x)
        elif source == 't2': x = self.conv_t2(x)
        elif source == 'flair': x = self.conv_flair(x)

        return x

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, kernel_size=3, norm_type='instance'):
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)

    def forward(self, x):
        normalized = self.param_free_norm(x)

        x = F.interpolate(x, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(x)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
    
class SPADE_Multimodal(nn.Module):
    def __init__(self, norm_nc, label_nc, kernel_size, norm_type='instance'):
        super().__init__()
        self.spade_t1 = SPADE(norm_nc, label_nc, kernel_size, norm_type)
        self.spade_t1ce = SPADE(norm_nc, label_nc, kernel_size, norm_type)
        self.spade_t2 = SPADE(norm_nc, label_nc, kernel_size, norm_type)
        self.spade_flair = SPADE(norm_nc, label_nc, kernel_size, norm_type)

    def forward(self, x, modality):
        if modality == 't1': x = self.spade_t1(x)
        elif modality == 't1ce': x = self.spade_t1ce(x)
        elif modality == 't2': x = self.spade_t2(x)
        elif modality == 'flair': x = self.spade_flair(x)
        else: raise ValueError('%s is not a recognized modality in SPADE_Block' % modality)
        
        return x
    
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv3d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv3d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv3d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        self.norm_0 = SPADE_Multimodal(fin, fin, kernel_size=3, norm_type='instance')
        self.norm_1 = SPADE_Multimodal(fmiddle, fmiddle, kernel_size=3, norm_type='instance')
        if self.learned_shortcut:
            self.norm_s = SPADE_Multimodal(fin, fin, kernel_size=3, norm_type='instance')

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, modality):
        x_s = self.shortcut(x, modality)

        dx = self.conv_0(self.actvn(self.norm_0(x, modality)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, modality)))

        out = x_s + dx

        return out

    def shortcut(self, x, modality):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, modality))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
    
class SPADEGenerator(nn.Module):
    def __init__(self, z_dim=3):
        super().__init__()
        nf = 128

        self.in_spade = SPADEResnetBlock(z_dim, nf)
        self.out_spade = SPADEResnetBlock(nf, z_dim)


    def forward(self, x, modality):
        
        x = self.in_spade(x, modality)
        x = self.out_spade(x, modality)

        return x