import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from .cmods.support_encoder import SupportEncoder
from .cmods.query_decoder import QueryDecoder
from .cmods.basic_modules import get_query_mask_and_order, MLP, inverse_sigmoid, get_ref_feat, TokenDecodeMLP
from capeformer.models.utils.two_stage_support_refine_transformer import TransformerEncoder, TransformerEncoderLayer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from copy import deepcopy


class ExtLocPredictor(nn.Module):
    def __init__(self,
                 num_feature_levels=3,
                 side_head=0,
                 use_joint_encoder=False,
                 use_self_att=False,
                 top_down_fuse=False,
                 point_init='cst',
                 embed_init='ref',
                 num_decoder_layers=3,
                 d_model=256,
                 nhead=8,
                 npoint=4,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 **kwargs):
        super().__init__()
        self.d_model = d_model
        self.kwargs = kwargs
        self.num_decoder_layers = num_decoder_layers
        if kwargs['reuse_layers']:
            slayer = SupportEncoder(num_feature_levels=num_feature_levels, use_self_att=use_self_att,
                                    n_points=npoint, n_heads=nhead, d_ffn=dim_feedforward, side_head=side_head)
            self.support_layers = nn.ModuleList([slayer for _ in range(num_decoder_layers)])
            qlayer = QueryDecoder(num_feature_levels=num_feature_levels, use_self_att=use_self_att,
                                  n_points=npoint, n_heads=nhead, d_ffn=dim_feedforward, side_head=side_head)
            self.decoder_layers = nn.ModuleList([qlayer for _ in range(num_decoder_layers)])
        else:
            self.support_layers = nn.ModuleList([
                SupportEncoder(num_feature_levels=num_feature_levels, use_self_att=use_self_att,
                               n_points=npoint, n_heads=nhead, d_ffn=dim_feedforward, side_head=side_head)
                for _ in range(num_decoder_layers)
            ])
            self.decoder_layers = nn.ModuleList([
                QueryDecoder(num_feature_levels=num_feature_levels, use_self_att=use_self_att,
                             n_points=npoint, n_heads=nhead, d_ffn=dim_feedforward, side_head=side_head)
                for _ in range(num_decoder_layers)
            ])
        self.pos_embed_proj = MLP(d_model, d_model, d_model, 2)

        self.use_joint_encoder = use_joint_encoder
        if self.use_joint_encoder:
            self.joint_encoder_layers = nn.ModuleList([
                TransformerEncoder(TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation), 1, )
                for _ in range(num_decoder_layers)
            ])
        self.top_down_fuse = top_down_fuse
        if self.top_down_fuse:
            self.top_down_projs = nn.ModuleList([
                nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size=1), nn.ReLU())
                for _ in range(num_feature_levels - 1)
            ])

        self.point_init = point_init
        self.embed_init = embed_init
        self.pe_layer = build_positional_encoding(dict(type='SinePositionalEncoding', num_feats=128, normalize=True))

        kpt_branch = TokenDecodeMLP(in_channels=self.d_model, hidden_channels=self.d_model)
        self.kpt_branch = nn.ModuleList([deepcopy(kpt_branch) for i in range(num_decoder_layers)])

        self.init_weights()
        if kwargs['ref_feat_aware']:
            self.ref_feat_weight = 1
        else:
            self.ref_feat_weight = 0
        return

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        for layer in self.support_layers:
            layer._reset_parameters()
        for layer in self.decoder_layers:
            layer._reset_parameters()
        for mlp in self.kpt_branch:
            nn.init.constant_(mlp.mlp[-1].weight.data, 0)
            nn.init.constant_(mlp.mlp[-1].bias.data, 0)
        return

    def get_inits(self, point_s, visible_s, ref_feat_s):
        query_order = self.pe_layer(point_s[0, ..., 0][:, None]).flatten(2).permute(0, 2, 1)

        if self.embed_init == 'cst':
            query_embed = torch.zeros_like(query_order)
        elif self.embed_init == 'ref':
            query_embed = (torch.stack(ref_feat_s) * visible_s).sum(0) / visible_s.sum(0).clamp(min=0.1)
        else:
            raise NotImplementedError

        if self.point_init == 'cst':
            latest_point = torch.zeros_like(query_order)[..., :2].sigmoid()
        elif self.point_init == 'sm':
            raise NotImplementedError
            vw = visible_s.permute(2, 0, 1).unsqueeze(-1)
            latest_point = (point_s * vw).sum(0) / vw.sum(0).clamp(min=0.01)
        else:
            raise NotImplementedError
        return query_embed, latest_point, query_order

    def forward(self, pyramid_q, pyramid_s, point_s, visible_s, headwise_refpoint_idxs_s):
        device = point_s.device
        num_shots, B, K, _ = point_s.shape
        shapes = torch.as_tensor([p.shape[-2:] for p in pyramid_q], dtype=torch.long).to(device)
        level_start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
        ref_pos_s = [self.pos_embed_proj(self.pe_layer.forward_coordinates(p)) for p in point_s]
        values_s = [torch.cat([p.flatten(2).transpose(1, 2) for p in pd], 1) for pd in pyramid_s]
        ref_feat_s = [get_ref_feat(values_s[s], point_s[s], shapes) * visible_s[s] for s in range(num_shots)]

        keypoint_embed, latest_point, query_order = self.get_inits(point_s, visible_s, ref_feat_s)
        latest_point_unsig = inverse_sigmoid(latest_point)

        pred_points = []
        for lid in range(self.num_decoder_layers):
            keypoint_embed_ups = []
            support_attviz_dict = []
            for sid in range(num_shots):
                keypoint_embed_up = self.support_layers[lid](keypoint_embed, ~visible_s[sid].bool().squeeze(-1),
                                                             query_order, values_s[sid],
                                                             point_s[sid], ref_pos_s[sid],
                                                             ref_feat_s[sid] * self.ref_feat_weight,
                                                             shapes, level_start, headwise_refpoint_idxs_s[sid])
                keypoint_embed_ups.append(keypoint_embed_up)
                support_attviz_dict.append(self.support_layers[lid].deform_attn.attviz_dict)

            keypoint_embed = (torch.stack(keypoint_embed_ups, 0) * visible_s).sum(0) / visible_s.sum(0).clamp(min=0.1)

            if self.use_joint_encoder:
                B, C, H, W = pyramid_q[0].shape
                map_mask = keypoint_embed.new_zeros((B, H, W)).to(torch.bool)
                pos_embed = self.pe_layer(map_mask).flatten(2).permute(2, 0, 1)
                pos_embed_cat = torch.cat([pos_embed, query_order.transpose(0, 1)])
                memory, keypoint_embed = self.joint_encoder_layers[lid](
                    pyramid_q[0].flatten(2).permute(2, 0, 1),
                    keypoint_embed.transpose(0, 1),
                    src_key_padding_mask=map_mask.flatten(1),
                    query_key_padding_mask=~visible_s.max(0)[0].bool().squeeze(-1),
                    pos=pos_embed_cat
                )
                keypoint_embed = keypoint_embed.transpose(0, 1)
                memory_pyramid = [memory.permute(1, 2, 0).reshape(B, C, H, W)] + pyramid_q[1:]
                values_q = torch.cat([p.flatten(2).transpose(1, 2) for p in memory_pyramid], 1)
            else:
                memory_pyramid = pyramid_q
                values_q = torch.cat([p.flatten(2).transpose(1, 2) for p in pyramid_q], 1)

            if self.top_down_fuse:
                tp_pyramid = [memory_pyramid[0]]
                for i, featmap in enumerate(pyramid_q[1:]):
                    fused = self.top_down_projs[i](featmap + F.upsample_bilinear(tp_pyramid[-1], featmap.shape[-2:]))
                    tp_pyramid.append(fused)
                values_q = torch.cat([p.flatten(2).transpose(1, 2) for p in tp_pyramid], 1)

            headwise_refpoint_idxs_q = headwise_refpoint_idxs_s[sid]
            ref_pos = self.pos_embed_proj(self.pe_layer.forward_coordinates(latest_point))
            ref_feat = get_ref_feat(values_q, latest_point, shapes) * self.ref_feat_weight
            keypoint_embed = self.decoder_layers[lid](keypoint_embed, None, query_order, values_q,
                                                      latest_point, ref_pos, ref_feat,
                                                      shapes, level_start, headwise_refpoint_idxs_q)
            pred_point_unsig = latest_point_unsig + self.kpt_branch[lid](keypoint_embed)
            pred_point = pred_point_unsig.sigmoid()
            pred_points.append(pred_point)
            latest_point = pred_point.detach()
            latest_point_unsig = pred_point_unsig.detach()

        return pred_points
