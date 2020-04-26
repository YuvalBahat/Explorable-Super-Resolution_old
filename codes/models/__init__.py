def create_model(opt,*kargs,**kwargs):
    model = opt['model']

    if model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt,*kargs,**kwargs)
    print('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
