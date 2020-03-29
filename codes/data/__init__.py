import torch.utils.data


def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt['phase']
    if phase == 'train':
        batch_size = dataset_opt['batch_size']
        shuffle = dataset_opt['use_shuffle']
        num_workers = dataset_opt['n_workers']
    else:
        batch_size = 1
        shuffle = False
        num_workers = 0#1 ybahat changed num_workers to prevent job falling, following https://github.com/pytorch/pytorch/issues/1355
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def create_dataset(dataset_opt,**kwargs):
    mode = dataset_opt['mode']
    if mode == 'LRHR':
        from data.LRHR_dataset import LRHRDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt,**kwargs)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                     dataset_opt['name']))
    return dataset
