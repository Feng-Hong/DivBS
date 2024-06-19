import yaml
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import secrets
from utils import custom_logger,random_str, get_date, re_nest_configs
import wandb

import torch.multiprocessing as mp
import methods



def init_seeds(seed):
    print('=====> Using fixed random seed: ' + str(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def main():
    # ============================================================================
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, type=str, help='Indicate the config file used for the training.')
    parser.add_argument('--seed', default=None, type=int, help='Fix the random seed for reproduction. Default is None(random seed).')
    parser.add_argument('--output_dir', default=None, type=str, help='Output directory that saves everything.')
    parser.add_argument('--log_file', default=None, type=str, help='Logger file name.')
    # notes
    parser.add_argument('--notes', default=None, type=str, help='Notes for the experiment.')
    parser.add_argument('--wandb_not_upload', action='store_true', help='Do not upload the result to wandb.')

    args = parser.parse_args()

    # ============================================================================
    # load config file
    print('=====> Loading config file: ' + args.cfg)
    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    print('=====> Config file loaded.')

    if args.seed is None:
        args.seed = secrets.randbelow(5000) 

    if args.seed is not None:
        config["seed"] = args.seed
    if args.log_file is not None:
        config['log_file'] = args.log_file
    
    if args.output_dir is None:
        args.output_dir = './exp/'
        # dataset
        args.output_dir = os.path.join(args.output_dir, config['dataset']['name'])
        # method
        args.output_dir = args.output_dir + '_' + config['method']
        # model
        args.output_dir = args.output_dir + '_' + config['networks']['type'] + '-' + config['networks']['params']['m_type']
        # bs
        args.output_dir = args.output_dir + '_bs' + str(config['training_opt']['batch_size'])
        # epochs
        args.output_dir = args.output_dir + '_ep' + str(config['training_opt']['num_epochs'])
        # lr
        args.output_dir = args.output_dir + '_lr' + str(config['training_opt']['optim_params']['lr'])
        # optimizer
        args.output_dir = args.output_dir + '_' + config['training_opt']['optimizer']
        # scheduler
        args.output_dir = args.output_dir + '_' + config['training_opt']['scheduler']
        # seed
        args.output_dir = args.output_dir + '_seed' + str(config['seed'])
        # ratio
        if 'method_opt' in config:
            if 'ratio' in config['method_opt']:
                args.output_dir = args.output_dir + '_r' + str(config['method_opt']['ratio'])
        # notes
        if args.notes is not None:
            args.output_dir = args.output_dir + '_' + args.notes
        

    # args.output_dir = os.path.join(args.output_dir, get_date()+ '_' + random_str(6))
    args.output_dir = os.path.join(args.output_dir, get_date())

    config['output_dir'] = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # wandb_not_upload
    if args.wandb_not_upload:
        os.environ["WANDB_MODE"] = "dryrun"
    else:
        os.environ["WANDB_MODE"] = "run"
    
    if args.log_file is None:
        logger = custom_logger(args.output_dir)
    else:
        logger = custom_logger(args.output_dir, args.log_file)

    logger.info('========================= Start Main =========================')


    # save config file
    logger.info('=====> Saving config file')
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info('=====> Config file saved')

    init_seeds(config["seed"])
    # logger.info(f'=====> Random seed initialized to {config["seed"]}')
    logger.info(f'=====> Wandb initialized')
    run = wandb.init(config=config,project="Efficient Selection")
    re_nest_configs(run.config)
    wandb.define_metric('acc', 'max')
    run.name = config['dataset']['name'] + '_' + config['output_dir'].split('/')[-2]

    wandb_local_path = wandb.run.dir
    # save wandb_local_path to wandb_local_path.txt
    with open(os.path.join(args.output_dir, 'wandb_local_path.txt'), 'w') as f:
        f.write(wandb_local_path)
        f.close()

    config['num_gpus'] = torch.cuda.device_count()
    logger.info(f'=====> Number of GPUs: {config["num_gpus"]}')

    Method = getattr(methods, config['method'])(config, logger)
    Method.run()

    logger.info('========================= End Main =========================')


if __name__ == '__main__':
    main()