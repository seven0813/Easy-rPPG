# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
from os import path as osp

from model.utils import get_root_logger, scandir

# automatically scan and import model modules
# scan all the files under the 'models' folder and collect files ending with
# '_Trainer.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder)
    if v.endswith('Trainer.py')
]
# import all the model modules
_model_modules = [
    importlib.import_module(f'model.trainer.{file_name}')
    for file_name in model_filenames
]


def create_trainer(opt,data_loader_dict):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt['trainer_type']

    # dynamic instantiation
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model = model_cls(opt,data_loader_dict)

    logger = get_root_logger()
    logger.info(f'Trainer [{model.__class__.__name__}] is created.')
    return model
