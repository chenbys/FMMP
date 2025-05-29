import torch
import torch.nn.functional as F
from mmpose.models import builder
from mmpose.models.detectors.base import BasePose
from mmpose.models.builder import POSENETS
import torch.nn as nn
from .main_predictor import ExtLocPredictor
from .cmods.basic_modules import proj_pyramid, make_pyramid_projs, decode_output


@POSENETS.register_module()
class ExtLocDetector(BasePose):
    def __init__(self, encoder_config, predictor, train_cfg=None, test_cfg=None, viz_cfg=None, pretrained=None):
        super().__init__()

        self.backbone = builder.build_backbone(encoder_config)
        self.backbone.init_weights(pretrained)
        self.predictor = ExtLocPredictor(**predictor)
        self.backbone_projs = make_pyramid_projs([256, 512, 1024, 2048], 256, num_feature_levels=3)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.target_type = test_cfg.get('target_type', 'GaussianHeatMap')

        self.test_iter = 0
        self.train_iter = 0
        self.test_cfg = test_cfg
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward_basic(self, **kwargs):
        B, G, C, H, W = kwargs['imgs'].shape
        featpyramid = self.backbone(kwargs['imgs'].flatten(0, 1))
        featpyramid = proj_pyramid(featpyramid, self.backbone_projs, self.upsample)
        point_s = kwargs['points'][:, :-1].transpose(0, 1)
        visible_s = kwargs['visibles'][:, :-1].transpose(0, 1).float().unsqueeze(-1)
        headwise_refpoint_idxs_s = kwargs['headwise_refpoint_idxs'][:, :-1].transpose(0, 1)

        featpyramid_s, featpyramid_q = [], []
        for fpy in featpyramid:
            c, h, w = fpy.shape[-3:]
            fpy = fpy.reshape(B, G, c, h, w)
            featpyramid_s.append(fpy[:, :-1])
            featpyramid_q.append(fpy[:, -1])
        featpyramid_sr = []
        for s in range(G - 1):
            featpyramid_sr.append([fpy[:, s] for fpy in featpyramid_s])

        pred_points = self.predictor(featpyramid_q, featpyramid_sr, point_s, visible_s, headwise_refpoint_idxs_s)
        return pred_points

    def forward(self, **kwargs):
        kwargs['highreso_imgs'] = kwargs['imgs']
        pred_points = self.forward_basic(**kwargs)
        if self.training:
            self.train_iter += 1
            self.test_iter = 0
            gt_points = kwargs['points'][:, -1]
            gt_visibles = kwargs['visibles'][:, -1].float()
            loss_dict = {}

            real_point_indicator = torch.zeros_like(gt_visibles)
            for b, K_c in enumerate(kwargs['category_keypoint_num']):
                real_point_indicator[b, :K_c.item()] = 1
            mix_point_indicator = 1 - real_point_indicator

            gv_r = (gt_visibles * real_point_indicator).unsqueeze(1)
            gv_m = (gt_visibles * mix_point_indicator).unsqueeze(1)
            bt_pred_points = torch.stack(pred_points, dim=1)
            l1_raw = F.l1_loss(bt_pred_points, gt_points.unsqueeze(1), reduction="none").sum(-1)

            loss_dict[f'l1_loss_real'] = ((l1_raw * gv_r).sum(-1) / gv_r.sum(-1).clamp(min=0.1)).sum(-1).mean()
            mw = self.train_cfg.mix_weight
            loss_dict[f'l1_loss_mix'] = mw * ((l1_raw * gv_m).sum(-1) / gv_m.sum(-1).clamp(min=0.1)).sum(-1).mean()

            return loss_dict
        else:
            self.test_iter += 1
            raw_num = kwargs['meta_q'][0]['raw_keypoint_num']
            result = {}
            for lid in range(len(pred_points)):
                p = decode_output(kwargs['meta_q'], pred_points[lid][:, :raw_num].cpu().numpy(),
                                  img_size=kwargs['imgs'].shape[-2:])
                if lid == (len(pred_points) - 1):
                    result['major'] = p
                else:
                    result[f'layer{lid}'] = p
            result.update({"sample_image_file": kwargs['meta_q'][0]['image_file']})
            return result

    def forward_test(self, **kwargs):
        return

    def show_result(self):
        return

    def forward_train(self, **kwargs):
        return

    @property
    def with_keypoint(self):
        return hasattr(self, 'keypoint_head')
