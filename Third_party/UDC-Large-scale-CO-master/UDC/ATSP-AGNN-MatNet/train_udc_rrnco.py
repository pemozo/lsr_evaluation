##########################################################################################
# UDC ATSP fine-tuning on on-the-fly RRNCO instances

import argparse
import logging
import os
import random
import re
import sys
from datetime import datetime

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from ATSPTrainerRRNCO import ATSPTrainerRRNCO as Trainer
from utils.utils import copy_all_src, create_logger


env_params = {
    'problem_size_low': 250,
    'problem_size_high': 500,
    'sub_size': 50,
    'pomo_size': 50,
    'sample_size': 30,
    'problem_gen_params': {
        'int_min': 0,
        'int_max': 1000 * 1000,
        'scaler': 1000 * 1000,
    },
}

model_p_params = {
    'embedding_dim': 64,
    'depth': 12,
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256 ** (1 / 2),
    'encoder_layer_num': 5,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16 ** (1 / 2),
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1 / 2) ** (1 / 2),
    'ms_layer2_init': (1 / 16) ** (1 / 2),
    'eval_type': 'argmax',
    'one_hot_seed_cnt': 50,
}

optimizer_params = {
    'optimizer': {'lr': 1e-5, 'weight_decay': 0},
    'optimizer_p': {'lr': 1e-5, 'weight_decay': 0},
    'scheduler': {'milestones': [40], 'gamma': 0.1},
}

trainer_params = {
    'use_cuda': True,
    'cuda_device_num': 0,
    'epochs': 1,
    'train_episodes': 20,
    'train_batch_size': 1,
    'validation_interval': 1,
    'model_load': {
        't_enable': False,
        'p_enable': False,
    },
    'validation_test_episodes': 2,
    'validation_test_batch_size': 1,
    'validation_aug_factor': 16,
}

rrnco_params = {
    'data_dir': '',
    'train_seed': 1234,
    'validation_seed': 3333,
    'size_seed': 4321,
    'cache_size': 10,
}

checkpoint_params = {
    'source_t_file': '',
    'source_p_file': '',
    'output_dir': '',
    'kind': 'smoke',
}

logger_params = {
    'log_file': {
        'desc': 'train__udc_rrnco',
        'filename': 'log.txt',
    }
}


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../../..'))
    checkpoint_root = os.path.abspath(os.path.join(script_dir, '../Checkpoints'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--train-episodes', type=int)
    parser.add_argument('--validation-episodes', type=int)
    parser.add_argument('--validation-interval', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument(
        '--rrnco-data-dir',
        default=os.path.join(
            project_root,
            'data/ATSP_data/RRNCO_atsp250_64instances/dataset',
        ),
    )
    parser.add_argument(
        '--t-file',
        default=os.path.join(checkpoint_root, 'checkpoint-tsp-300.pt'),
    )
    parser.add_argument(
        '--p-file',
        default=os.path.join(checkpoint_root, 'checkpoint-partition-300.pt'),
    )
    parser.add_argument('--checkpoint-root', default=checkpoint_root)
    parser.add_argument('--run-id')
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = 1 if args.mode == 'smoke' else 50
    if args.train_episodes is None:
        args.train_episodes = 20 if args.mode == 'smoke' else 1000
    if args.validation_episodes is None:
        args.validation_episodes = 2 if args.mode == 'smoke' else 16
    for name in ('epochs', 'train_episodes', 'validation_episodes', 'validation_interval'):
        if getattr(args, name) < 1:
            parser.error('--{} must be at least 1'.format(name.replace('_', '-')))
    if args.learning_rate <= 0:
        parser.error('--learning-rate must be positive')

    if args.run_id is None:
        args.run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')
    if not re.match(r'^[A-Za-z0-9_.-]+$', args.run_id):
        parser.error('--run-id may only contain letters, digits, dot, underscore, and hyphen')
    return args


def apply_args(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    trainer_params['epochs'] = args.epochs
    trainer_params['train_episodes'] = args.train_episodes
    trainer_params['validation_test_episodes'] = args.validation_episodes
    trainer_params['validation_interval'] = args.validation_interval
    optimizer_params['optimizer']['lr'] = args.learning_rate
    optimizer_params['optimizer_p']['lr'] = args.learning_rate
    rrnco_params['data_dir'] = args.rrnco_data_dir
    rrnco_params['train_seed'] = args.seed
    rrnco_params['size_seed'] = args.seed + 1
    rrnco_params['validation_seed'] = args.seed + 2
    checkpoint_params['source_t_file'] = args.t_file
    checkpoint_params['source_p_file'] = args.p_file
    checkpoint_params['kind'] = 'smoke' if args.mode == 'smoke' else 'final'
    output_group = 'rrnco_smoke' if args.mode == 'smoke' else 'rrnco_finetune'
    checkpoint_params['output_dir'] = os.path.join(args.checkpoint_root, output_group, args.run_id)
    logger_params['log_file']['desc'] = 'train__udc_rrnco_{}_{}'.format(args.mode, args.run_id)


def print_config():
    logger = logging.getLogger('root')
    for name in (
            'env_params',
            'model_p_params',
            'model_params',
            'optimizer_params',
            'trainer_params',
            'rrnco_params',
            'checkpoint_params'):
        logger.info('%s%s', name, globals()[name])


def main():
    args = parse_args()
    apply_args(args)
    create_logger(**logger_params)
    print_config()
    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        model_p_params=model_p_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
        rrnco_params=rrnco_params,
        checkpoint_params=checkpoint_params,
    )
    copy_all_src(trainer.result_folder)
    trainer.run()


if __name__ == '__main__':
    main()
