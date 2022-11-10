import torch.nn as nn
import torch
import pytorch3d.ops

from .utils import pytorch_utils as pt_utils
from .utils import pointnet2_utils


class EdgeConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mlps = cfg.mlps
        mlps[0] = mlps[0]*2
        if cfg.use_xyz:
            mlps[0] += 6
        self.shared_mlp = pt_utils.SharedMLP(mlps, bn=True)
        self.cfg = cfg

    def get_graph_feature(self, new_xyz, new_feat, xyz, feat, k, use_xyz=False):
        bs = xyz.size(0)
        device = torch.device('cuda')
        feat = feat.permute(0, 2, 1).contiguous() if feat is not None else None
        new_feat = new_feat.permute(
            0, 2, 1).contiguous() if new_feat is not None else None
        if use_xyz:
            feat = torch.cat([feat, xyz], dim=-1) if feat is not None else xyz
            new_feat = torch.cat([new_feat, new_xyz],
                                 dim=-1) if new_feat is not None else new_xyz
        # b,n,c
        _, knn_idx, _ = pytorch3d.ops.knn_points(
            new_xyz, xyz, K=k, return_nn=True)

        knn_feat = pytorch3d.ops.knn_gather(feat, knn_idx)
        # b,n1,k,c
        feat_tiled = new_feat.unsqueeze(-2).repeat(1, 1, k, 1)
        edge_feat = torch.cat([knn_feat-feat_tiled, feat_tiled], dim=-1)

        return edge_feat.permute(0, 3, 1, 2).contiguous()

    def forward(self, xyz, feat, npoints):
        """
        Args:
            xyz : b,n,3
            feat : b,c,n
        """
        device = xyz.device
        if self.cfg.sample_method == 'FPS':
            sample_idx = pointnet2_utils.furthest_point_sample(
                xyz, self.npoint)
        elif self.cfg.sample_method == 'Range':
            sample_idx = torch.arange(npoints).repeat(
                xyz.size(0), 1).int().to(device)
        else:
            raise NotImplementedError(
                'Sample method %s has not been implemented' % self.cfg.sample_method)
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz.transpose(1, 2).contiguous(), sample_idx)
            .transpose(1, 2)
            .contiguous()
        )

        new_feat = pointnet2_utils.gather_operation(
            feat, sample_idx) if feat is not None else None

        edge_feat = self.get_graph_feature(
            new_xyz, new_feat, xyz, feat, self.cfg.nsample, use_xyz=self.cfg.use_xyz)
        # b, 2*c(+6), npoints, nsample
        new_feat = self.shared_mlp(edge_feat)
        new_feat = new_feat.max(dim=-1, keepdim=False)[0]
        return new_xyz, new_feat, sample_idx.long()


class DGCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        for layer_cfg in cfg.layers_cfg:
            self.SA_modules.append(EdgeConv(layer_cfg))

        self.fc = (
            pt_utils.Seq(cfg.layers_cfg[-1].mlps[-1])
            # .conv1d(cfg.out_channels, bn=True)
            .conv1d(cfg.out_channels, activation=None)
        )
        self.downsample_ratios = cfg.downsample_ratios
        assert len(self.downsample_ratios) == 3
        self.cfg = cfg

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(
            1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pcd):
        npts = pcd.shape[1]

        xyz, features = self._break_up_pc(pcd)

        l_xyz, l_features, l_idxs = [xyz], [features], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_idxs = self.SA_modules[i](
                l_xyz[i], l_features[i], npts // self.downsample_ratios[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_idxs.append(li_idxs)

        idxs_list = [[] for _ in range(len(l_idxs))]
        bs = l_idxs[0].shape[0]
        for i in range(len(l_idxs)):
            for bi in range(bs):
                if i == 0:
                    idxs_list[i].append(l_idxs[0][bi])
                else:
                    idxs_list[i].append(
                        idxs_list[i - 1][bi][l_idxs[i][bi].long()])

        idx = torch.stack(idxs_list[len(l_idxs) - 1], dim=0)

        return dict(
            xyz=l_xyz[-1],
            feat=self.fc(l_features[-1]),
            idx=idx.long(),
            geo_feat=l_features[-1]
        )
