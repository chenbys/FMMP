import torch
import torch.nn.functional as F
from torch import nn
from .deform_att import MDeformAttn


class StructrualDeformAttnW(nn.Module):
    def __init__(self, side_head, d_model=256, n_levels=4, n_heads=8, n_points=4, **kwargs):
        super().__init__()
        self.side_head = side_head
        self.attn_layers = nn.ModuleList()
        for hid in range(self.side_head):
            self.attn_layers.append(MDeformAttn(d_model=d_model, n_levels=n_levels,
                                                n_heads=n_heads, n_points=n_points))
        if self.side_head >= 2:
            self.projector = nn.Sequential(
                nn.Linear(d_model * (self.side_head - 1), d_model),
                nn.ReLU()
            )
            nn.init.constant_(self.projector[0].weight.data, 0)
            nn.init.constant_(self.projector[0].bias.data, 0)
        return

    def _reset_parameters(self):
        for shead in self.attn_layers:
            shead._reset_parameters()

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index, side_refpoint_idxs_and_order):
        """
            :param side_refpoint_idxs       (batch_size, keypoint_num, max_head_num)
            :param side_refpoint_idxs_order (batch_size, keypoint_num, max_head_num)
        """
        side_refpoint_idxs, side_refpoint_idxs_order = side_refpoint_idxs_and_order.transpose(0, 1)
        shead_attviz_dicts = []
        atted_feats = []
        for hid, shead in enumerate(self.attn_layers):
            batch_keypoints = reference_points[:, :, 0]

            batch_side_refpoints = []
            for b in range(len(batch_keypoints)):
                side_refidx = side_refpoint_idxs[b][:, hid]
                side_refpoint = batch_keypoints[b][side_refidx]
                batch_side_refpoints.append(side_refpoint)
            side_refpoint = torch.stack(batch_side_refpoints)

            atted_feat = shead(query, side_refpoint.unsqueeze(2).expand_as(reference_points), input_flatten,
                               input_spatial_shapes, input_level_start_index)
            atted_feats.append(atted_feat)
            shead_attviz_dicts.append(shead.attviz_dict)

        self.attviz_dict = {}
        for key in shead_attviz_dicts[0].keys():
            self.attviz_dict[key] = torch.stack([d[key] for d in shead_attviz_dicts], dim=1)

        result = atted_feats[0]
        if self.side_head >= 2:
            result = result + self.projector(torch.cat(atted_feats[1:], -1))
        return result
