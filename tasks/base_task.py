import torch
import pytorch_lightning as pl
import os.path as osp
import json
import time

from optimizers import create_optimizer
from schedulers import create_scheduler
from models import create_model
from utils import TorchPrecision, TorchSuccess, TorchRuntime, estimateAccuracy, estimateOverlap, estimateWaymoOverlap, AverageMeter


class BaseTask(pl.LightningModule):

    def __init__(self, cfg, log):
        super().__init__()
        self.cfg = cfg
        self.model = create_model(cfg.model_cfg, log)
        log.info('Model size = %.2f MB' % self.compute_model_size())
        if 'Waymo' in cfg.dataset_cfg.dataset_type:
            self.succ_total = AverageMeter()
            self.prec_total = AverageMeter()
            self.succ_easy = AverageMeter()
            self.prec_easy = AverageMeter()
            self.succ_medium = AverageMeter()
            self.prec_medium = AverageMeter()
            self.succ_hard = AverageMeter()
            self.prec_hard = AverageMeter()
            self.n_frames_total = 0
            self.n_frames_easy = 0
            self.n_frames_medium = 0
            self.n_frames_hard = 0
        else:
            self.prec = TorchPrecision()
            self.succ = TorchSuccess()
            self.runtime = TorchRuntime()
            if cfg.save_test_result:
                self.pred_bboxes = []

        self.txt_log = log

    def compute_model_size(self):
        num_param = sum([p.numel() for p in self.model.parameters()])
        param_size = num_param * 4 / 1024 / 1024  # MB
        return param_size

    def configure_optimizers(self):
        optimizer = create_optimizer(self.cfg.optimizer_cfg, self.parameters())
        scheduler = create_scheduler(self.cfg.scheduler_cfg, optimizer)
        return dict(
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

    def training_step(self, *args, **kwargs):
        raise NotImplementedError(
            'Training_step has not been implemented!')

    def on_validation_epoch_start(self):
        self.prec.reset()
        self.succ.reset()
        self.runtime.reset()

    def forward_on_tracklet(self, tracklet):
        raise NotImplementedError(
            'Forward_on_tracklet has not been implemented!')

    def validation_step(self, batch, batch_idx):
        tracklet = batch[0]
        start_time = time.time()
        pred_bboxes, gt_bboxes = self.forward_on_tracklet(tracklet)
        end_time = time.time()
        runtime = end_time-start_time
        n_frames = len(tracklet)
        overlaps, accuracies = [], []
        for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
            overlap = estimateOverlap(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
            accuracy = estimateAccuracy(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
            overlaps.append(overlap)
            accuracies.append(accuracy)

        self.succ(torch.tensor(overlaps, device=self.device))
        self.prec(torch.tensor(accuracies, device=self.device))
        self.runtime(torch.tensor(runtime, device=self.device),
                     torch.tensor(n_frames, device=self.device))

    def on_validation_epoch_end(self):

        self.log('precesion', self.prec.compute(), prog_bar=True)
        self.log('success', self.succ.compute(), prog_bar=True)
        self.log('runtime', self.runtime.compute(), prog_bar=True)

    def _on_test_epoch_start_kitti_format(self):
        self.prec.reset()
        self.succ.reset()
        self.runtime.reset()

    def _on_test_epoch_start_waymo_format(self):
        self.succ_total.reset()
        self.prec_total.reset()
        self.succ_easy.reset()
        self.prec_easy.reset()
        self.succ_medium.reset()
        self.prec_medium.reset()
        self.succ_hard.reset()
        self.prec_hard.reset()
        self.n_frames_total = 0
        self.n_frames_easy = 0
        self.n_frames_medium = 0
        self.n_frames_hard = 0

    def on_test_epoch_start(self):
        if 'Waymo' in self.cfg.dataset_cfg.dataset_type:
            self._on_test_epoch_start_waymo_format()
        else:
            self._on_test_epoch_start_kitti_format()

    def _test_step_kitti_format(self, batch, batch_idx):
        # if batch_idx != 0:
        #     return
        tracklet = batch[0]
        start_time = time.time()
        pred_bboxes, gt_bboxes = self.forward_on_tracklet(tracklet)
        end_time = time.time()
        runtime = end_time-start_time
        n_frames = len(tracklet)
        # print()
        # print()
        # print(pred_bboxes)
        # assert False
        if self.cfg.save_test_result:
            self.pred_bboxes.append((batch_idx, pred_bboxes))
        overlaps, accuracies = [], []
        for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
            overlap = estimateOverlap(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
            accuracy = estimateAccuracy(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
            overlaps.append(overlap)
            accuracies.append(accuracy)
        self.succ(torch.tensor(overlaps, device=self.device))
        self.prec(torch.tensor(accuracies, device=self.device))
        self.runtime(torch.tensor(runtime, device=self.device),
                     torch.tensor(n_frames, device=self.device))

    def _test_step_waymo_format(self, batch, batch_idx):
        # if batch_idx != 0:
        #     return
        tracklet = batch[0]
        tracklet_length = len(tracklet) - 1
        pred_bboxes, gt_bboxes = self.forward_on_tracklet(tracklet)
        n_frames = len(tracklet)

        success = TorchSuccess()
        precision = TorchPrecision()

        overlaps, accuracies = [], []
        for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
            overlap = estimateWaymoOverlap(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space)
            accuracy = estimateAccuracy(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
            overlaps.append(overlap)
            accuracies.append(accuracy)
        success(torch.tensor(overlaps, device=self.device))
        precision(torch.tensor(accuracies, device=self.device))
        success = success.compute() if type(
            success.compute()) == float else success.compute().item()
        precision = precision.compute() if type(
            precision.compute()) == float else precision.compute().item()

        self.succ_total.update(success, n=tracklet_length)
        self.prec_total.update(precision, n=tracklet_length)
        self.n_frames_total += n_frames
        if tracklet[0]['mode'] == 'easy':
            self.succ_easy.update(success, n=tracklet_length)
            self.prec_easy.update(precision, n=tracklet_length)
            self.n_frames_easy += n_frames
        elif tracklet[0]['mode'] == 'medium':
            self.succ_medium.update(success, n=tracklet_length)
            self.prec_medium.update(precision, n=tracklet_length)
            self.n_frames_medium += n_frames
        elif tracklet[0]['mode'] == 'hard':
            self.succ_hard.update(success, n=tracklet_length)
            self.prec_hard.update(precision, n=tracklet_length)
            self.n_frames_hard += n_frames

        self.txt_log.info('Total: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_total.avg, self.succ_total.avg,  self.n_frames_total))
        self.txt_log.info('easy: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_easy.avg, self.succ_easy.avg,  self.n_frames_easy))
        self.txt_log.info('medium: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_medium.avg, self.succ_medium.avg,  self.n_frames_medium))
        self.txt_log.info('hard: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_hard.avg, self.succ_hard.avg,  self.n_frames_hard))

    def test_step(self, batch, batch_idx):
        if 'Waymo' in self.cfg.dataset_cfg.dataset_type:
            return self._test_step_waymo_format(batch, batch_idx)
        else:
            return self._test_step_kitti_format(batch, batch_idx)

    def _on_test_epoch_end_kitti_format(self):
        self.log('precesion', self.prec.compute(), prog_bar=True)
        self.log('success', self.succ.compute(), prog_bar=True)
        self.log('runtime', self.runtime.compute(), prog_bar=True)
        if self.cfg.save_test_result:
            self.pred_bboxes.sort(key=lambda x: x[0])
            data = []
            for idx, bbs in self.pred_bboxes:
                pred_bboxes = []
                for bb in bbs:
                    pred_bboxes.append(bb.encode())
                data.append(pred_bboxes)
            with open(osp.join(self.cfg.work_dir, 'result.json'), 'w') as f:
                json.dump(data, f)

    def _on_test_epoch_end_waymo_format(self):
        self.txt_log.info('============ Final ============')
        self.txt_log.info('Total: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_total.avg, self.succ_total.avg,  self.n_frames_total))
        self.txt_log.info('easy: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_easy.avg, self.succ_easy.avg,  self.n_frames_easy))
        self.txt_log.info('medium: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_medium.avg, self.succ_medium.avg,  self.n_frames_medium))
        self.txt_log.info('hard: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_hard.avg, self.succ_hard.avg,  self.n_frames_hard))

    def on_test_epoch_end(self):
        if 'Waymo' in self.cfg.dataset_cfg.dataset_type:
            return self._on_test_epoch_end_waymo_format()
        else:
            return self._on_test_epoch_end_kitti_format()
