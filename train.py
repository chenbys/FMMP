import argparse
import copy
import os
import os.path as osp
import time
import sys
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from capeformer.apis import train_model
from capeformer.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.utils import collect_env, get_root_logger
import warnings

warnings.filterwarnings('ignore')

from cape_extloc import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--val-only', action='store_true')
    parser.add_argument('--viz-only', action='store_true')
    parser.add_argument('--config', default=None, help='train config file path')
    parser.add_argument('--work-dir', default=None, help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--auto-resume', type=bool, default=False)
    parser.add_argument('--no-validate', action='store_true')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int)
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        default={}, help='override some settings in the used config, the key-value pair '
                                         'in xxx=yyy format will be merged into config file. For example, '
                                         "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.val_only = args.val_only
    cfg.viz_only = args.viz_only

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg_name = osp.splitext(osp.basename(args.config))[0]
        cfg.work_dir = f'./output/{cfg_name}'

    # auto resume
    if args.auto_resume:
        checkpoint = os.path.join(args.work_dir, 'latest.pth')
        if os.path.exists(checkpoint):
            cfg.resume_from = checkpoint

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
        assert cfg.batch_size % torch.cuda.device_count() == 0
        cfg.data.samples_per_gpu = cfg.batch_size // torch.cuda.device_count()
    else:
        cfg.batch_size = cfg.data.samples_per_gpu * torch.cuda.device_count()

    set_outdir_by_exp_name(cfg)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file_name = f'{os.path.basename(cfg.work_dir)}.log'
    log_file = osp.join(cfg.work_dir, log_file_name)
    cfg.log_file = log_file
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    logger.info('\n------------------------------------------------------------------------------------------')
    logger.info(sys.argv)
    logger.info('\n------------------------------------------------------------------------------------------')
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    logger.info('\n------------------------------------------------------------------------------------------')
    logger.info(cfg.lr_config)
    logger.info('\n------------------------------------------------------------------------------------------')
    logger.info(cfg.data)
    logger.info('\n------------------------------------------------------------------------------------------')
    logger.info(cfg.model)
    logger.info('\n------------------------------------------------------------------------------------------')

    # set random seeds
    # args.seed = 1
    # args.deterministic = True
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    model = build_posenet(cfg.model)
    train_datasets = [build_dataset(cfg.data.train)]

    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset = build_dataset(val_dataset, dict(test_mode=True))

    viz_dataset = val_dataset

    logger.info(f'Total params: {sum([p.nelement() for p in model.parameters()]) / 1e6:.2f}M')
    model.dataset_cat_info = val_dataset.cats
    train_model(model, train_datasets, cfg,
                distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, meta=meta)


if __name__ == '__main__':
    main()
