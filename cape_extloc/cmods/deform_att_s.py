from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn.functional as F

from .deform_ops.functions import MSDeformAttnFunction
from .deform_ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from .deform_ops.modules.ms_deform_attn import MSDeformAttn
from torch.nn.init import xavier_uniform_, constant_
from torch import nn
import math


class StructrualDeformAttn(MSDeformAttn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index, headwise_refpoint_idxs,
                input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :param headwise_refpoint_idxs      (N, Length_{query}, >=n_head)

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        assert reference_points.shape[-1] == 2
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        batch_keypoints = reference_points[:, :, 0]
        batch_headwise_refpoints = []
        for b in range(N):
            # headwise_refpoint_idxs      (N, Length_{query}, >=n_head)
            headwise_refpoint_idx = headwise_refpoint_idxs[b][:, :self.n_heads]
            headwise_refpoint = batch_keypoints[b][headwise_refpoint_idx]
            batch_headwise_refpoints.append(headwise_refpoint)
        refpoints_structural = torch.stack(batch_headwise_refpoints)[:, :, :, None, None]
        refpoints_default = reference_points[:, :, None, :, None, :]
        refpoints_used = refpoints_structural
        sampling_locations = refpoints_used + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        try:
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
                self.im2col_step)
        except:
            # CPU
            output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        if self.training:
            self.attviz_dict = {}
        else:
            self.attviz_dict = {}
            self.attviz_dict['reference_points'] = reference_points
            self.attviz_dict['sampling_locations'] = sampling_locations
            self.attviz_dict['attention_weights'] = attention_weights
        return output
