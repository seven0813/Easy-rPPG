# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------

import importlib
import numpy as np
import random
import torch
import torch.utils.data
from functools import partial
from os import path as osp

from model.utils import get_root_logger, scandir, get_dist_info
from model.utils.options import get_dataset_phase


__all__ = ['create_dataset', 'create_dataloader']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder)
    if v.endswith('_dataset.py')
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'dataloader.{file_name}')
    for file_name in dataset_filenames
]


def create_dataset(dataset_opt):
    dataset_type = dataset_opt['type']

    # dynamic instantiation
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    dataset = dataset_cls(dataset_opt)

    logger = get_root_logger()
    logger.info(
        f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} '
        'is created.')
    return dataset


def create_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):

    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    dataloader_args = {}
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader_args['generator'] = generator

    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_workers_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_workers_per_gpu'] * multiplier
        
        dataloader_args.update(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=dataset_opt['shuffle'] if sampler is None else False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
        )
        if seed is not None and num_workers > 0:
            dataloader_args['worker_init_fn'] = partial(
                worker_init_fn,
                num_workers=num_workers,
                rank=rank,
                seed=seed,
                
            )

    elif phase in ['val', 'test']:  # validation
            ## TODO: 暂时不支持多卡验证和测试，后续再添加
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_workers_per_gpu']  
                  
            dataloader_args.update(
                dataset=dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                drop_last=False
            )
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. '
                         "Supported ones are 'train', 'val' and 'test'.")

    dataloader = torch.utils.data.DataLoader(**dataloader_args)
    logger = get_root_logger()
    logger.info(
        '%s: dataset_len=%s, num_batches=%s, '
        'batch_size=%s, num_workers=%s, shuffle=%s, drop_last=%s.',
        dataset_opt['name'],
        len(dataset),
        len(dataloader),
        dataloader_args['batch_size'],
        dataloader_args['num_workers'],
        dataloader_args.get('shuffle', False),
        dataloader_args['drop_last'],
    )
    return dataloader


def worker_init_fn(worker_id, num_workers, rank, seed):
    '''多线程加载数据时，保证每个线程的随机数不一样'''
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(opt):
    # create train and val dataloaders
    data_loader_dict = dict()
    seed = opt.get('seed', opt.get('manual_seed'))
    for dataset_key, dataset_opt in opt['datasets'].items():
        phase = get_dataset_phase(dataset_key)
        if phase is None:
            continue
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            data_loader_dict[dataset_key] = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=seed)

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            data_loader_dict[dataset_key] = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=seed)
            
        elif phase == 'test':   
            test_set = create_dataset(dataset_opt)
            data_loader_dict[dataset_key] = create_dataloader(
                test_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=seed)

    
    return data_loader_dict
