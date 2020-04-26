import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from . import block as B
from . import spectral_norm as SN
import functools
import numpy as np
import os
import models.modules.archs_util as arch_util
import torch.nn.functional as F
import re

####################
# Generator
####################
class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


class SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv',latent_input=None,num_latent_channels=None):
        super(RRDBNet, self).__init__()
        self.latent_input = latent_input
        if num_latent_channels is not None and num_latent_channels>0:
            num_latent_channels_HR = 1 * num_latent_channels
            if 'HR_rearranged' in latent_input:
                num_latent_channels *= upscale**2
        self.num_latent_channels = 1*num_latent_channels
        self.upscale = upscale
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        if latent_input is not None:
            in_nc += num_latent_channels
        if latent_input is None or 'all_layers' not in latent_input:
            num_latent_channels,num_latent_channels_HR = 0,0

        USE_MODULE_LISTS = True
        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None,return_module_list=USE_MODULE_LISTS)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA',latent_input_channels=num_latent_channels) for _ in range(nb)]
        LR_conv = B.conv_block(nf+num_latent_channels, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode,return_module_list=USE_MODULE_LISTS)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        if latent_input is not None and 'all_layers' in latent_input:
            if 'LR' in latent_input:
                self.latent_upsampler = nn.Upsample(scale_factor=upscale if upscale==3 else 2)
        HR_conv0 = B.conv_block(nf+num_latent_channels_HR, nf, kernel_size=3, norm_type=None, act_type=act_type,return_module_list=USE_MODULE_LISTS)
        HR_conv1 = B.conv_block(nf+num_latent_channels_HR, out_nc, kernel_size=3, norm_type=None, act_type=None,return_module_list=USE_MODULE_LISTS)

        if USE_MODULE_LISTS:
            self.model = nn.ModuleList(fea_conv+\
                [B.ShortcutBlock(B.sequential(*(rb_blocks+LR_conv),return_module_list=USE_MODULE_LISTS),latent_input_channels=num_latent_channels,use_module_list=True)]+\
                                       upsampler+HR_conv0+HR_conv1)
        else:
            self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
                *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        if self.latent_input is not None:
            if 'HR_downscaled' in self.latent_input:
                # latent_input_HR = 1*self.Z
                latent_input_HR,x = torch.split(x,split_size_or_sections=[x.size(1)-3,3],dim=1)
                latent_input_HR = latent_input_HR.view([latent_input_HR.size(0)]+[-1]+[self.upscale*val for val in list(latent_input_HR.size()[2:])])
                latent_input = torch.nn.functional.interpolate(input=latent_input_HR,scale_factor=1/self.upscale,mode='bilinear',align_corners=False)
            else:
                latent_input = 1*self.Z
            x = torch.cat([latent_input, x], dim=1)
        for i,module in enumerate(self.model):
            module_children = [str(type(m)) for m in module.children()]
            if i>0 and self.latent_input is not None and 'all_layers' in self.latent_input:
                if len(module_children)>0 and 'Upsample' in module_children[0]:
                    if 'LR' in self.latent_input:
                        latent_input = self.latent_upsampler(latent_input)
                    elif 'HR_rearranged' in self.latent_input:
                        raise Exception('Unsupported yet')
                        latent_input = latent_input.view()
                    elif 'HR_downscaled' in self.latent_input:
                        latent_input = 1*latent_input_HR
                elif 'ReLU' not in str(type(module)):
                    x = torch.cat([latent_input,x],1)
            x = module(x)
        return x


####################
# Discriminator
####################

# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA',input_patch_size=128):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        FC_end_patch_size = 1*input_patch_size
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type,mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type,act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/2)
        # 64, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type,act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type,act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/2)
        # 32, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type,act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type,act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/2)
        # 16, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type,act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type,act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/2)
        # 8, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type,act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type,act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/2)
        # 4, 512
        self.features = B.sequential(*([conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,conv9]))

        # classifier
        self.classifier = nn.Sequential(nn.Linear(base_nf*8 * int(FC_end_patch_size)**2, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        if self.last_FC_layers:
            x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Discriminator_VGG_128_(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA',input_patch_size=128,num_2_strides=5,nb=10):
        super(Discriminator_VGG_128_, self).__init__()
        assert num_2_strides<=5,'Can be modified by adding more stridable layers, if needed.'
        self.num_2_strides = 1*num_2_strides
        # features
        # hxw, c
        # 128, 64
        FC_end_patch_size = 1*input_patch_size
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type,mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2 if num_2_strides>0 else 1, norm_type=norm_type,act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/(2 if num_2_strides>0 else 1))
        num_2_strides -= 1
        # 64, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2 if num_2_strides>0 else 1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/(2 if num_2_strides>0 else 1))
        num_2_strides -= 1
        # 32, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2 if num_2_strides>0 else 1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/(2 if num_2_strides>0 else 1))
        num_2_strides -= 1
        # 16, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2 if num_2_strides>0 else 1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/(2 if num_2_strides>0 else 1))
        num_2_strides -= 1
        # 8, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2 if num_2_strides>0 else 1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/(2 if num_2_strides>0 else 1))
        num_2_strides -= 1
        # 4, 512
        self.features = B.sequential(*([conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,conv9][:nb]))

        self.last_FC_layers = self.num_2_strides==5 #Replacing the FC layers with convolutions, which means using a patch discriminator:
        self.last_FC_layers = False
        # classifier
        # FC_end_patch_size = input_patch_size//(2**self.num_2_strides)
        if self.last_FC_layers:
            self.classifier = nn.Sequential(nn.Linear(base_nf*8 * int(FC_end_patch_size)**2, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))
        else:
            # num_feature_channels = base_nf*8
            num_feature_channels = [l for l in self.features.children()][-2].num_features
            pseudo_FC_conv0 = B.conv_block(num_feature_channels,min(100,num_feature_channels),kernel_size=8,stride=1,norm_type=norm_type,act_type=act_type, mode=mode,pad_type=None)
            pseudo_FC_conv1 = B.conv_block(min(100,num_feature_channels),1,kernel_size=1,stride=1,norm_type=norm_type,act_type=act_type, mode=mode)
            self.classifier = nn.Sequential(pseudo_FC_conv0, nn.LeakyReLU(0.2, False),pseudo_FC_conv1) # Changed the LeakyRelu inplace arg to False here, because it caused a bug for some reason.

    def forward(self, x):
        x = self.features(x)
        if self.last_FC_layers:
            x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



####################
# Perceptual Network
####################
RETRAINING_OBLIGING_MODIFICATIONS = ['num_channel_factor_\d(\.\d)?$','patches_init_first']

# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,feature_layer=34,use_bn=False,use_input_norm=True,device=torch.device('cpu'),**kwargs):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.__dict__[arch+'_bn'](pretrained='untrained' not in arch_config)
        else:
            model = torchvision.models.__dict__[arch](pretrained='untrained' not in arch_config)
        # I now remove all unnecessary layers before changing the model configuration, because this change may make alter the number of layers, thus necessitating changing the feature_layer parameter.
        model.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = model.features
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def _initialize_weights(self):#This function was copied from the torchvision.models.vgg code:
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# Encoder:
def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,padding=1, bias=True)

def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out
