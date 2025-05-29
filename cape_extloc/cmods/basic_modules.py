import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .deform_ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from mmpose.models.utils.ops import resize
from mmpose.core.post_processing import transform_preds
import numpy as np


def get_ref_feat(values_flatten, ref_points, spatial_shapes):
    n_heads, n_points = 1, 1
    N, Len_q, _ = ref_points.shape
    N, Len_in, d_model = values_flatten.shape
    n_levels = spatial_shapes.size(0)

    sampling_locations = ref_points.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).expand(
        -1, -1, n_heads, n_levels, n_points, -1).contiguous()
    attention_weights = (torch.ones(N, Len_q, n_heads, n_levels, n_points).to(values_flatten) /
                         (n_levels * n_points)).contiguous()
    values_split = values_flatten.view(N, Len_in, n_heads, d_model // n_heads)
    feat_from = ms_deform_attn_core_pytorch(values_split, spatial_shapes, sampling_locations, attention_weights)
    return feat_from


def get_mean_of_deep_support(feature_s, target_s):
    query_embed_list = []
    for feat_pyramid, target in zip(feature_s, target_s):
        resized_feature = resize(input=feat_pyramid[-1], size=target.shape[-2:],
                                 mode='bilinear', align_corners=False)
        target = target / (target.sum(dim=-1).sum(dim=-1)[:, :, None, None] + 1e-8)
        query_embed = target.flatten(2) @ resized_feature.flatten(2).permute(0, 2, 1)
        query_embed_list.append(query_embed)
    query_embed = torch.mean(torch.stack(query_embed_list, dim=0), 0)
    query_embed = query_embed
    return query_embed


def get_query_mask_and_order(mask_s, pe_layer):
    query_order = pe_layer(mask_s.new_zeros((mask_s.shape[0], 1, mask_s.shape[1])).to(torch.bool))
    query_mask = (~mask_s.to(torch.bool)).squeeze(-1)
    tgt_key_padding_mask_remove_all_true = query_mask.clone().to(query_mask.device)
    tgt_key_padding_mask_remove_all_true[query_mask.logical_not().sum(dim=-1) == 0, 0] = False
    query_mask = tgt_key_padding_mask_remove_all_true
    return query_mask, query_order.flatten(2).permute(0, 2, 1)


def make_pyramid_projs(backbone_dims, embed_dims, num_feature_levels):
    input_proj_list = []
    for in_channels in backbone_dims[::-1][:num_feature_levels]:
        input_proj_list.append(nn.Sequential(
            nn.Conv2d(in_channels, embed_dims, kernel_size=1),
        ))
    projs = nn.ModuleList(input_proj_list)
    for proj in projs:
        nn.init.xavier_uniform_(proj[0].weight, gain=1)
        nn.init.constant_(proj[0].bias, 0)
    return projs


def proj_pyramid(feat_pyramid, projs, upsample):
    srcs = []
    for idx in range(len(projs)):
        x = feat_pyramid[-idx - 1]
        src = projs[idx](x)
        srcs.append(src)
    output_features = [srcs[0]]
    for i in range(1, len(srcs)):
        upsampled_feature = upsample(output_features[-1])
        combined_feature = upsampled_feature + srcs[i]
        output_features.append(combined_feature)
    return output_features


class TokenDecodeMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels=2,
                 num_layers=3):
        super(TokenDecodeMLP, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_channels, hidden_channels))
                layers.append(nn.GELU())
            else:
                layers.append(nn.Linear(hidden_channels, hidden_channels))
                layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def decode_output(img_metas, output, img_size):
    batch_size = len(img_metas)
    W, H = img_size
    output = output * np.array([W, H])[None, None, :]  # [bs, query, 2], coordinates with recovered shapes.

    if 'bbox_id' or 'query_bbox_id' in img_metas[0]:
        bbox_ids = []
    else:
        bbox_ids = None

    c = np.zeros((batch_size, 2), dtype=np.float32)
    s = np.zeros((batch_size, 2), dtype=np.float32)
    image_paths = []
    score = np.ones(batch_size)
    for i in range(batch_size):
        c[i, :] = img_metas[i]['center']
        s[i, :] = img_metas[i]['scale']
        image_paths.append(img_metas[i]['image_file'])

        if 'query_bbox_score' in img_metas[i]:
            score[i] = np.array(
                img_metas[i]['bbox_score']).reshape(-1)
        if 'bbox_id' in img_metas[i]:
            bbox_ids.append(img_metas[i]['bbox_id'])
        elif 'query_bbox_id' in img_metas[i]:
            bbox_ids.append(img_metas[i]['bbox_id'])

    preds = np.zeros(output.shape)
    for idx in range(output.shape[0]):
        preds[i] = transform_preds(output[i], c[i], s[i], [W, H], use_udp=False)

    all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
    all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = 1.0  # NOTE: Currently, assume all predicted points are of 100% confidence.
    all_boxes[:, 0:2] = c[:, 0:2]
    all_boxes[:, 2:4] = s[:, 0:2]
    all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
    all_boxes[:, 5] = score
    result = {}
    result['preds'] = all_preds
    result['boxes'] = all_boxes
    result['image_paths'] = image_paths
    result['bbox_ids'] = bbox_ids
    return result
