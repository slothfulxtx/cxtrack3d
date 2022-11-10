
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DGCNN
from .utils import pytorch_utils as pt_utils
from .transformer import Transformer
from .exrpn import EXRPN
from .rpn import RPN


class CXTrack(nn.Module):
    def __init__(self, cfg, log):
        super().__init__()
        self.cfg = cfg
        self.log = log
        self.backbone_net = DGCNN(cfg.backbone_cfg)
        self.transformer = Transformer(cfg.transformer_cfg)

        if not cfg.transformer_cfg.layers_cfg[-1].mask_pred:
            self.fc_mask = (
                pt_utils.Seq(cfg.transformer_cfg.feat_dim)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(1, activation=None)
            )
        else:
            self.fc_mask = None

        if not cfg.transformer_cfg.layers_cfg[-1].center_pred:
            self.fc_center = (
                pt_utils.Seq(3 + cfg.transformer_cfg.feat_dim)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(3 + cfg.transformer_cfg.feat_dim, activation=None)
            )
        else:
            self.fc_center = (
                pt_utils.Seq(3 + cfg.transformer_cfg.feat_dim)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(cfg.transformer_cfg.feat_dim, activation=None)
            )

        if cfg.exrpn_cfg:
            self.rpn_type = 'exrpn'
            self.exrpn = EXRPN(cfg.exrpn_cfg)
        elif cfg.rpn_cfg:
            self.rpn_type = 'rpn'
            self.rpn = RPN(cfg.rpn_cfg)

    def forward(self, input_dict):
        template_pcd = input_dict['template_pcd']
        search_pcd = input_dict['search_pcd']
        template_npts = template_pcd.shape[1]
        search_npts = search_pcd.shape[1]
        template_mask_ref = input_dict['template_mask_ref']
        search_mask_ref = input_dict['search_mask_ref']
        device = template_pcd.device
        bs = template_pcd.shape[0]

        output_dict = {}

        b_output_dict = self.backbone_net(template_pcd)

        template_xyz = b_output_dict['xyz']
        template_feat = b_output_dict['feat']
        template_idx = b_output_dict['idx']
        template_geo_feat = b_output_dict['geo_feat']

        b_output_dict = self.backbone_net(search_pcd)

        search_xyz = b_output_dict['xyz']
        search_feat = b_output_dict['feat']
        search_idx = b_output_dict['idx']
        search_geo_feat = b_output_dict['geo_feat']

        search_mask_ref = torch.gather(
            search_mask_ref, 1, search_idx)
        template_mask_ref = torch.gather(
            template_mask_ref, 1, template_idx)

        output_dict.update(
            search_xyz=search_xyz,
            template_xyz=template_xyz
        )

        if self.training:
            output_dict['search_mask_gt'] = torch.gather(
                input_dict['search_mask_gt'], 1, search_idx)
            output_dict['template_mask_gt'] = torch.gather(
                input_dict['template_mask_gt'], 1, template_idx)

        trfm_input_dict = dict(
            search_xyz=search_xyz,
            template_xyz=template_xyz,
            search_feat=search_feat,
            template_feat=template_feat,
            search_mask_ref=search_mask_ref,
            template_mask_ref=template_mask_ref,
        )

        if self.training:
            trfm_input_dict.update(
                search_mask_gt=output_dict['search_mask_gt'],
                template_mask_gt=output_dict['template_mask_gt'],
            )

        trfm_output_dict = self.transformer(trfm_input_dict)

        search_feat = trfm_output_dict.pop('search_feat')
        template_feat = trfm_output_dict.pop('template_feat')

        output_dict.update(trfm_output_dict)

        if not self.cfg.transformer_cfg.layers_cfg[-1].mask_pred:
            search_mask_pred = self.fc_mask(search_feat).squeeze(1)
            output_dict.update(
                search_mask_pred_9=search_mask_pred
            )
            search_mask_score = search_mask_pred.sigmoid()
        else:
            search_mask_score = trfm_output_dict.pop('search_mask_ref')

        if not self.cfg.transformer_cfg.layers_cfg[-1].center_pred:
            search_xyz_feat = torch.cat(
                (search_xyz.transpose(1, 2).contiguous(), search_feat),
                dim=1
            )
            offset = self.fc_center(search_xyz_feat)
            search_center_xyz = search_xyz + \
                offset[:, :3, :].transpose(1, 2).contiguous()
            search_feat = search_feat + offset[:, 3:, :]
            output_dict.update(
                search_center_pred_9=search_center_xyz
            )
        else:
            search_xyz_feat = torch.cat(
                (search_xyz.transpose(1, 2).contiguous(), search_feat),
                dim=1
            )

            offset = self.fc_center(search_xyz_feat)
            search_feat = search_feat + offset
            search_center_xyz = trfm_output_dict.pop('search_center_ref')

        output_dict.update(
            center_xyz=search_center_xyz
        )

        rpn_input_dict = dict(
            search_xyz=search_xyz,
            search_mask_score=search_mask_score,
            search_feat=search_feat,
            search_center_xyz=search_center_xyz
        )

        if self.rpn_type == 'exrpn':
            rpn_output_dict = self.exrpn(rpn_input_dict)
        elif self.rpn_type == 'rpn':
            rpn_output_dict = self.rpn(rpn_input_dict)
        output_dict.update(rpn_output_dict)

        # output_dict.update(s_xyz=search_xyz, t_xyz=template_xyz)
        return output_dict
