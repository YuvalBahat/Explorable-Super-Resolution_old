import functools
import torch
import torch.nn as nn
from torch.nn import init

import models.modules.architecture as arch

####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    if 'filter_layer' in m.__dict__ and m.__getattribute__('filter_layer'):
        return
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt,CEM=None,num_latent_channels=None):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    opt_net['latent_input'] = opt_net['latent_input'] if opt_net['latent_input']!="None" else None
    if which_model == 'RRDB_net':  # RRDB
        netG = arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv',
            latent_input=(opt_net['latent_input']+'_'+opt_net['latent_input_domain']) if opt_net['latent_input'] is not None else None,num_latent_channels=num_latent_channels)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    if opt_net['CEM_arch']:
        netG = CEM.WrapArchitecture_PyTorch(netG,opt['datasets']['train']['patch_size'] if opt['is_train'] else None)
    if opt['is_train'] and which_model != 'MSRResNet':# and which_model != 'DnCNN':
        init_weights(netG, init_type='kaiming', scale=0.1)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG

# Discriminator
def define_D(opt,CEM=None):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']
    input_patch_size = opt['datasets']['train']['patch_size']
    # in_nc = opt_net['in_nc']*(2 if opt['network_D']['decomposed_input'] else 1)
    in_nc = opt_net['in_nc']
    assert not ((opt_net['pre_clipping'] or opt_net['decomposed_input']) and which_model!='PatchGAN'),'Unsupported yet'
    if CEM is not None:
        input_patch_size -= 2*CEM.invalidity_margins_HR
    if which_model == 'discriminator_vgg_128':
        kwargs = {}
        if 'num_2_strides' in opt_net:
            kwargs['num_2_strides'] = opt_net['num_2_strides']
        netD = arch.Discriminator_VGG_128(in_nc=in_nc, base_nf=opt_net['nf'], nb=opt_net['n_layers'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'],input_patch_size=input_patch_size,**kwargs)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD


def define_F(opt, use_bn=False,**kwargs):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if 'arch' in kwargs.keys()and 'vgg11' in kwargs['arch']:
        feature_layer = int(kwargs['arch'][len('vgg11_'):])
        kwargs['arch'] = 'vgg11'
    else:
        if use_bn:
            feature_layer = 49
        else:
            feature_layer = 34
    netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,use_input_norm=True, device=device,**kwargs)
    # netF = arch.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF

