# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import yaml
from copy import deepcopy
from collections import OrderedDict
from os import path as osp
from typing import Any


class AttrDict(OrderedDict):
    """Ordered dict with attribute-style access.

    It keeps normal dict behavior, so both opt['train'] and opt.train work.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = to_attr_dict(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def to_attr_dict(obj: Any) -> Any:
    """Recursively convert dict-like config nodes to AttrDict."""
    if isinstance(obj, AttrDict):
        return obj
    if isinstance(obj, dict):
        return AttrDict((key, to_attr_dict(value)) for key, value in obj.items())
    if isinstance(obj, list):
        return [to_attr_dict(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(to_attr_dict(item) for item in obj)
    return obj


def get_dataset_phase(dataset_key: str) -> str | None:
    """Return train/val/test for dataset config keys, including train_1 style keys."""
    phase = dataset_key.split('_')[0]
    if phase in {'train', 'val', 'test'}:
        return phase
    return None


def root_has_output_dir(root: str) -> bool:
    """Return whether root already points inside an output directory."""
    root_parts = set(osp.normpath(root).split(osp.sep))
    return bool({'experiments', 'results'} & root_parts)


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path: str) -> AttrDict:
    """Parse option file.

    Args:
        opt_path (str): Option file path.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    # 将 datasets 下的公共配置作为各 train/val/test 数据集的默认值。
    # 数据集子配置中的同名字段优先，便于为某个 phase 单独覆盖。
    if 'datasets' in opt:
        common_dataset_options = {
            key: deepcopy(value)
            for key, value in opt['datasets'].items()
            if get_dataset_phase(key) is None
        }
        for dataset_key, dataset in opt['datasets'].items():
            # for several datasets, e.g., test_1, test_2
            phase = get_dataset_phase(dataset_key)
            if phase is None:
                continue
            for key, value in common_dataset_options.items():
                dataset.setdefault(key, deepcopy(value))
            dataset['phase'] = phase

    # paths
    root = opt['path']['root']
    if opt['mode'] == 'train_and_test':
        if root_has_output_dir(root):
            experiments_root = osp.join(root, opt['name'])
        else:
            experiments_root = osp.join(root, 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')
        opt['path']['tensorboard'] = osp.join(experiments_root, 'tensorboard')

    else:  # test
        if root_has_output_dir(root):
            results_root = osp.join(root, opt['name'])
        else:
            results_root = osp.join(root, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return to_attr_dict(opt)


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg
