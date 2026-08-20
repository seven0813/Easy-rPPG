
""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import numpy as np
import torch
import os.path as osp
import logging

from model.trainer import create_trainer
from dataloader import get_dataloader
from model.utils import (backup_config_file, get_root_logger, parse,
                         make_exp_dirs, get_time_str, dict2str, get_env_info)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument(
        '--config_file',required=True,type=str,help='Path to the YAML configuration file.')
    return parser

def init_logger(opt):
    log_file = osp.join(opt['path']['log'],
                        f"{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='rppg', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    
    return logger


if __name__ == "__main__":
    # Parse the config file path, then load the YAML configuration.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    config = parse(args.config_file)
    
    if config.mode == "train_and_test":
        resume_state = config.train.get('resume_state')
    else:
        resume_state = None
    if resume_state:
        if not osp.isfile(resume_state):
            raise FileNotFoundError(f"Resume state does not exist: {resume_state}")
    else:
        make_exp_dirs(config)
    config_backup_path = backup_config_file(args.config_file, config)
    set_random_seed(config.seed)
    logger = init_logger(config)
    
    logger.info('Loading config file from: {}'.format(args.config_file))
    logger.info('Backed up config file to: {}'.format(config_backup_path))
    logger.info(config.name)
    logger.info(f"Set random seed to {config.seed}")
    if resume_state:
        logger.info("Resume training state: %s", resume_state)
    else:
        logger.info("Make exp dirs ready.")
    
    ## create dataloaders
    data_loader_dict = get_dataloader(config)
    
    ## create trainer
    trainer = create_trainer(config, data_loader_dict)
    
    
    if config.mode == "train_and_test":
        trainer.train(data_loader_dict)
        trainer.test(data_loader_dict)
    elif config.mode == "only_test":
        trainer.test(data_loader_dict)
    else:
        logger.info("Mode only support train_and_test or only_test !")
