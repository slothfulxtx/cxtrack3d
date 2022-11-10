import argparse
import yaml
from addict import Dict
import json
import open3d
import numpy as np
import torch

from utils import Logger, estimateAccuracy, estimateOverlap, TorchSuccess, TorchPrecision, IO
from datasets import create_datasets
from datasets.utils import BoundingBox, crop_and_center_pcd


def parse_args():
    parser = argparse.ArgumentParser(
        description='visualize 3d point cloud single object tracking result')
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--result_dir', type=str,
                        default=None, required=True, help='path to result file')
    parser.add_argument('--debug', action='store_true',
                        help='choose which state to run visualization')
    args = parser.parse_args()
    return args


def add_args_to_cfg(args, cfg):
    cfg.result_dir = args.result_dir
    cfg.debug = args.debug


def get_color_from_xyz(point_xyz, low=(0.5, 0.5, 0.3), high=(0.5, 0.5, 0.7), reverse=False):
    assert len(point_xyz.shape) == 2
    if reverse:
        low, high = high, low
    low_color = np.tile(np.array(low), (point_xyz.shape[0], 1))
    high_color = np.tile(np.array(high), (point_xyz.shape[0], 1))
    h = point_xyz[:, 1:2]
    h_mean = np.mean(h)
    h_std = np.std(h)
    h = np.clip(h, h_mean - 1.5*h_std, h_mean+1.5*h_std)
    alpha = (h - h.min())/(h.max()-h.min()+1e-6)
    alpha = np.tile(alpha, (1, 3))
    color = alpha * high_color + (1-alpha)*low_color
    return color


def swap_xy_axis(rot_mat):
    assert rot_mat.shape == (3, 3)
    new_rot_mat = np.copy(rot_mat)
    new_rot_mat[0, :] = rot_mat[2, :]
    new_rot_mat[2, :] = rot_mat[0, :]
    return new_rot_mat


def visualize_tracklet(tracklet_id, tracklet, result, cfg, log):
    pred_bboxes = []
    for item in result:
        pred_bbox = BoundingBox.decode(item)
        pred_bboxes.append(pred_bbox)

    gt_bboxes = []
    pcds = []
    for frame in tracklet:
        gt_bbox = frame['bbox']
        gt_bboxes.append(gt_bbox)
        pcd = frame['pcd']
        pcds.append(pcd)

    prec = TorchPrecision()
    succ = TorchSuccess()
    overlaps, accuracies = [], []
    for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
        overlap = estimateOverlap(
            gt_bbox, pred_bbox, dim=cfg.eval_cfg.iou_space, up_axis=cfg.dataset_cfg.up_axis)
        accuracy = estimateAccuracy(
            gt_bbox, pred_bbox, dim=cfg.eval_cfg.iou_space, up_axis=cfg.dataset_cfg.up_axis)
        overlaps.append(overlap)
        accuracies.append(accuracy)

    succ(torch.tensor(overlaps))
    prec(torch.tensor(accuracies))
    log.info('  Succ=%f Prec=%f ' % (succ.compute(), prec.compute()))
    # if prec.compute() > 80:
    #     return

    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1600, height=800)
    vis.get_render_option().load_from_json("./open3d_render_option.json")
    current_pcd = None
    current_pred_bbox = None
    current_gt_bbox = None
    gt_bbox_vis = True
    pred_bbox_vis = True

    def visualize_frame(vis, frame_id):
        nonlocal pcds, gt_bboxes, pred_bboxes, current_pcd, current_pred_bbox, current_gt_bbox, gt_bbox_vis, pred_bbox_vis
        if current_pcd is not None:
            vis.remove_geometry(current_pcd, False)
        if current_pred_bbox is not None:
            vis.remove_geometry(current_pred_bbox, False)
        if current_gt_bbox is not None:
            vis.remove_geometry(current_gt_bbox, False)

        pcd = open3d.geometry.PointCloud()
        # print(pcds[frame_id].points.T.shape)
        point_xyz = pcds[frame_id].points.T
        point_color = get_color_from_xyz(point_xyz, low=(
            0.6, 0.6, 0.6), high=(0.2, 0.2, 1.0), reverse=True)
        pcd.points = open3d.utility.Vector3dVector(point_xyz)
        pcd.colors = open3d.utility.Vector3dVector(point_color)

        pred_bb = pred_bboxes[frame_id]
        gt_bb = gt_bboxes[frame_id]

        # Rot = open3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_quaternion(gt_bb.)
        # print(gt_bb.rotation_matrix)

        gt_bbox = open3d.geometry.OrientedBoundingBox(
            gt_bb.center, swap_xy_axis(gt_bb.rotation_matrix), gt_bb.wlh)
        gt_bbox.color = (0, 1.0, 0)
        pred_bbox = open3d.geometry.OrientedBoundingBox(
            pred_bb.center, swap_xy_axis(pred_bb.rotation_matrix), pred_bb.wlh)
        pred_bbox.color = (0.6, 0.6, 0)
        # gt
        if current_pcd is None:
            vis.add_geometry(pcd)
        else:
            vis.add_geometry(pcd, False)
        if gt_bbox_vis:
            vis.add_geometry(gt_bbox, False)
        if pred_bbox_vis:
            vis.add_geometry(pred_bbox, False)
        current_pcd = pcd
        current_gt_bbox = gt_bbox
        current_pred_bbox = pred_bbox
        # vis.update_geometry(pcd)

    current_frame_id = 0
    visualize_frame(vis, current_frame_id)

    def switch_to_next_frame(vis):
        nonlocal current_frame_id, pcds, log
        if current_frame_id < len(pcds) - 1:
            current_frame_id += 1
            visualize_frame(vis, current_frame_id)
            log.info(
                '  Switch to the next frame[%d/%d] of tracklet...' % (current_frame_id, len(pcds)))
        else:
            log.info('  At the last frame of tracklet!')

    def switch_to_next_10_frame(vis):
        nonlocal current_frame_id, pcds, log
        if current_frame_id < len(pcds) - 1:
            current_frame_id += 10
            current_frame_id = min(current_frame_id, len(pcds)-1)
            visualize_frame(vis, current_frame_id)
            log.info(
                '  Switch to the next frame[%d/%d] of tracklet...' % (current_frame_id, len(pcds)))
        else:
            log.info('  At the last frame of tracklet!')

    def switch_to_prev_frame(vis):
        nonlocal current_frame_id, pcds, log
        if current_frame_id > 0:
            current_frame_id -= 1
            visualize_frame(vis, current_frame_id)
            log.info(
                '  Switch to the prev frame[%d/%d] of tracklet...' % (current_frame_id, len(pcds)))
        else:
            log.info('  At the first frame of tracklet!')

    def switch_to_prev_10_frame(vis):
        nonlocal current_frame_id, pcds, log
        if current_frame_id > 0:
            current_frame_id -= 10
            current_frame_id = max(current_frame_id, 0)
            visualize_frame(vis, current_frame_id)
            log.info(
                '  Switch to the prev frame[%d/%d] of tracklet...' % (current_frame_id, len(pcds)))
        else:
            log.info('  At the first frame of tracklet!')

    def switch_gt_bbox(vis):
        nonlocal gt_bbox_vis, current_frame_id
        gt_bbox_vis = not gt_bbox_vis
        visualize_frame(vis, current_frame_id)

    def switch_pred_bbox(vis):
        nonlocal pred_bbox_vis, current_frame_id
        pred_bbox_vis = not pred_bbox_vis
        visualize_frame(vis, current_frame_id)

    def save_target_pcd(vis):
        nonlocal current_frame_id, pcds, gt_bboxes
        tgt = crop_and_center_pcd(
            pcds[current_frame_id], gt_bboxes[current_frame_id])
        IO.put('./tmp/tgt_pcd_%d.xyz' % current_frame_id, tgt.points.T)
        log.info('#### save tgt pcd (frame = %d) to /tmp/ ###' %
                 current_frame_id)

    def exit_vis(vis):
        vis.destroy_window()

    vis.register_key_callback(65, switch_to_prev_frame)
    vis.register_key_callback(68, switch_to_next_frame)

    vis.register_key_callback(87, switch_to_prev_10_frame)
    vis.register_key_callback(83, switch_to_next_10_frame)

    vis.register_key_callback(90, switch_pred_bbox)
    vis.register_key_callback(88, switch_gt_bbox)
    vis.register_key_callback(80, save_target_pcd)
    vis.register_key_callback(27, exit_vis)
    # vis.poll_events()
    vis.run()
    # vis.destroy_window()


def main():

    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = Dict(cfg)
    add_args_to_cfg(args, cfg)

    log = Logger(name='3DSOT', log_file=None)

    test_dataset = create_datasets(
        cfg=cfg.dataset_cfg,
        split_types='test',
        log=log
    )
    with open(cfg.result_dir, 'r') as f:
        results = json.load(f)
    assert len(test_dataset) == len(results)
    for tracklet_id, (tracklet, result) in enumerate(zip(test_dataset, results)):
        log.info('Visualizing tracklet_id = %d/%d...' %
                 (tracklet_id, len(test_dataset)))
        visualize_tracklet(tracklet_id, tracklet, result, cfg, log)


if __name__ == '__main__':
    main()
