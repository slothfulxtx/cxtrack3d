import argparse
import os
import os.path as osp
from datetime import datetime
from torch.utils.data import DataLoader
import yaml
from addict import Dict
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
import torch
from tqdm import tqdm
import pynvml

from utils import Logger, pl_ddp_rank
from tasks import create_task
from datasets import create_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description='3d point cloud single object tracking task')
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--workspace', type=str,
                        default='./workspace', help='path to workspace')
    parser.add_argument('--run_name', type=str,
                        default=None, help='name of this run')
    parser.add_argument('--gpus', type=int, nargs='+',
                        default=None, help='specify gpu devices')
    parser.add_argument('--resume_from', type=str,
                        default=None, help='path to checkpoint file')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--phase', type=str, default='train', choices=[
                        'train', 'test'], help='choose which phase to run point cloud single object tracking')
    parser.add_argument('--debug', action='store_true',
                        help='choose which state to run point cloud single object tracking')
    args = parser.parse_args()
    return args


def add_args_to_cfg(args, cfg):
    run_name = osp.splitext(osp.basename(args.config))[
        0] if args.run_name is None else args.run_name
    cfg.work_dir = osp.abspath(osp.join(
        args.workspace, run_name, datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')))
    if args.gpus is None:
        pynvml.nvmlInit()
        gpu_ids = list(range(torch.cuda.device_count()))
        mem_free = []
        for gpu_id in gpu_ids:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_free.append(dict(free=meminfo.free, gpu_id=gpu_id))
        mem_free.sort(key=lambda x: x['free'], reverse=True)
        cfg.gpus = [mem_free[0]['gpu_id']]
    else:
        cfg.gpus = args.gpus
    cfg.resume_from = osp.abspath(
        args.resume_from) if args.resume_from is not None else None
    cfg.phase = args.phase
    cfg.debug = args.debug
    cfg.dataset_cfg.debug = args.debug
    if cfg.debug:
        cfg.seed = 1234
        cfg.train_cfg.val_per_epoch = 1
        cfg.train_cfg.max_epochs = 10
        cfg.train_cfg.batch_size = 2*len(cfg.gpus)
    else:
        cfg.seed = args.seed

    if cfg.phase == 'test':
        assert len(cfg.gpus) == 1
        cfg.save_test_result = True if not cfg.debug else False


class CustomProgressBar(TQDMProgressBar):

    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)
        self.state = None

    def get_metrics(self, trainer, pl_module):
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


class CustomTensorBoardLogger(TensorBoardLogger):

    @property
    def root_dir(self) -> str:
        return self.save_dir

    @property
    def log_dir(self) -> str:
        return self.save_dir

    @rank_zero_only
    def save(self) -> None:
        return


def main():

    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = Dict(cfg)
    add_args_to_cfg(args, cfg)

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    if not cfg.debug and pl_ddp_rank() == 0:
        os.makedirs(cfg.work_dir, exist_ok=True)
        with open(osp.join(cfg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(cfg.to_dict(), f)

    log_file_dir = osp.join(
        cfg.work_dir, '3DSOT.log') if not cfg.debug else None
    log = Logger(name='3DSOT', log_file=log_file_dir)

    task = create_task(cfg, log)
    if cfg.phase == 'train':
        train_dataset, val_dataset = create_datasets(
            cfg=cfg.dataset_cfg,
            split_types=['train', 'test'],  # ['train', 'val']
            log=log
        )
        # train_dataset[34]
        # return
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.train_cfg.batch_size,
            pin_memory=True,
            num_workers=cfg.train_cfg.num_workers,
            shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=cfg.eval_cfg.batch_size,
            pin_memory=False,
            num_workers=cfg.eval_cfg.num_workers,
            collate_fn=lambda x: x
        )
        regular_ckpt_callback = ModelCheckpoint(
            filename='{epoch}',
            save_top_k=-1,
            save_last=True,
            every_n_epochs=cfg.train_cfg.save_per_epoch
        )
        best_ckpt_callback = ModelCheckpoint(
            filename='best_{epoch}_{precesion}_{success}',
            monitor='precesion',
            mode='max',
            save_top_k=cfg.train_cfg.save_top_k
        )

        progress_bar_callback = CustomProgressBar()
        logger = CustomTensorBoardLogger(
            save_dir=cfg.work_dir, version='', name='')
        # init trainer
        if cfg.strategy:
            strategy = cfg.strategy
            assert strategy in ['dp', 'ddp_find_unused_parameters_false']
        else:
            strategy = 'ddp_find_unused_parameters_false'

        trainer = pl.Trainer(
            gpus=cfg.gpus,
            strategy=strategy,
            max_epochs=cfg.train_cfg.max_epochs,
            callbacks=[regular_ckpt_callback,
                       best_ckpt_callback, progress_bar_callback] if not cfg.debug else [progress_bar_callback],
            default_root_dir=cfg.work_dir,
            check_val_every_n_epoch=cfg.train_cfg.val_per_epoch,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            enable_checkpointing=not cfg.debug,
            log_every_n_steps=1,
            logger=logger,
            gradient_clip_val=0.0
        )
        trainer.fit(task, train_dataloader, val_dataloader,
                    ckpt_path=cfg.resume_from)

    elif cfg.phase == 'test':
        assert cfg.resume_from is not None
        test_dataset = create_datasets(
            cfg=cfg.dataset_cfg,
            split_types='test',
            log=log
        )
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=cfg.eval_cfg.batch_size,
            pin_memory=False,
            num_workers=cfg.eval_cfg.num_workers,
            collate_fn=lambda x: x
        )
        progress_bar_callback = CustomProgressBar()
        logger = CustomTensorBoardLogger(
            save_dir=cfg.work_dir, version='', name='')

        trainer = pl.Trainer(
            gpus=cfg.gpus,
            strategy='ddp',
            log_every_n_steps=1,
            callbacks=[progress_bar_callback],
            default_root_dir=cfg.work_dir,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            logger=logger,
        )
        trainer.test(task, test_dataloader, ckpt_path=cfg.resume_from)
    else:
        raise NotImplementedError(
            '{} has not been implemented!'.format(cfg.phase))


if __name__ == '__main__':
    main()
