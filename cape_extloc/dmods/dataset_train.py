from mmpose.datasets import DATASETS
import random
import numpy as np
from capeformer.datasets.datasets.mp100.transformer_dataset import TransformerPoseDataset
import copy
import torch
from .side_func import get_side_ref_idx_and_orders
from .pad_func import *


@DATASETS.register_module()
class CJCTrainSet(TransformerPoseDataset):
    def __init__(self, ann_file, img_prefix, data_cfg, pipeline, valid_class_ids,
                 max_kpt_num=None, num_shots=1, num_queries=100, num_episodes=1, test_mode=False,
                 epoch_sample_num=None, keypoint_padding={}):
        self.epoch_sample_num = epoch_sample_num
        super().__init__(ann_file=ann_file, img_prefix=img_prefix, data_cfg=data_cfg, pipeline=pipeline,
                         valid_class_ids=valid_class_ids, max_kpt_num=max_kpt_num, num_shots=num_shots,
                         num_queries=num_queries, num_episodes=num_episodes, test_mode=test_mode)
        self.keypoint_padding = keypoint_padding
        self.is_train = True
        self.cid_to_K_max = {}
        for cid, d in self.cats.items():
            all_ids = []
            for se in d['skeleton']:
                all_ids += se
            self.cid_to_K_max[cid] = max(all_ids) + 1
        return

    def random_paired_samples(self):
        # Will be called every epoch.
        num_datas = [len(self.cat2obj[self._class_to_ind[cls]]) for cls in self.valid_classes]
        if self.epoch_sample_num is None:
            samples_per_class_in_this_epoch = max(num_datas)
        else:
            samples_per_class_in_this_epoch = int(self.epoch_sample_num / len(self.valid_classes))

        all_samples = []
        for cls in self.valid_class_ids:
            for i in range(samples_per_class_in_this_epoch):
                shot = random.sample(self.cat2obj[cls], self.num_shots + 1)
                all_samples.append(shot)
        self.paired_samples = np.array(all_samples)
        np.random.shuffle(self.paired_samples)

    def pad_pose(self, **kwargs):
        kwargs.update(dict(K_max=self.keypoint_padding.num))
        if self.keypoint_padding.type == 'zero':
            return get_zero_padding(**kwargs)
        elif self.keypoint_padding.type == 'eqdiv':
            return get_eqdiv_padding(**kwargs)
        elif self.keypoint_padding.type == 'mixup':
            return get_mixup_padding(**kwargs)
        else:
            raise NotImplementedError

    def get_single_data(self, obj_id, **kwargs):
        obj = copy.deepcopy(self.db[obj_id])
        obj['ann_info'] = copy.deepcopy(self.ann_info)
        ori_data = self.pipeline(obj)

        _, img_H, img_W = ori_data['img'].shape
        assert img_H == img_W
        img_meta = ori_data['img_metas'].data
        cid = img_meta['category_id']
        keypoint = torch.tensor(img_meta['joints_3d'][:, :2]) / img_H
        visible = torch.tensor(ori_data['target_weight']).bool().squeeze(1)
        se_pairs = self.cats[cid]['skeleton']
        category_keypoint_num = self.cid_to_K_max[cid]
        img_meta['category_keypoint_num'] = category_keypoint_num
        img_meta['raw_keypoint_num'] = len(keypoint)

        assert torch.where(visible)[0].max().item() <= (category_keypoint_num + 1)
        keypoint, visible, se_pairs = self.pad_pose(keypoint=keypoint, visible=visible, se_pairs=se_pairs,
                                                    category_keypoint_num=category_keypoint_num, **kwargs)

        link = torch.zeros(len(visible), len(visible))
        for s, e in se_pairs:
            if visible[s] and visible[e]:
                link[s, e] = 1
                link[e, s] = 1
        # keypoint: [K,2]
        # head-wise refpoint [K,N_max_pad,2]
        # [P;linked(P); P*(N_max_pad-1)][:N_max_pad].
        headwise_refpoint_idx_and_order = get_side_ref_idx_and_orders(visible, se_pairs, link)
        return ori_data['img'], keypoint, visible, link, ori_data['target'], img_meta, headwise_refpoint_idx_and_order

    def __getitem__(self, idx):
        pair_ids = self.paired_samples[idx]
        assert len(pair_ids) == self.num_shots + 1
        imgs, points, visibles, links, cids, metas = [], [], [], [], [], []
        headwise_refpoint_idxs = []
        K_max = self.keypoint_padding.num
        mixup_lam = np.random.beta(self.keypoint_padding.alpha, self.keypoint_padding.alpha, K_max)
        nonzero_padding = np.random.rand(K_max) < self.keypoint_padding.prob
        se_pairs = self.cats[self.db[pair_ids[0]]['category_id']]['skeleton']
        selected_se_pair_idx = np.random.randint(len(se_pairs), size=K_max)

        for did in pair_ids:
            img, point, visible, link, heatmap, meta, headwise_refpoint_idx = self.get_single_data(
                did, mixup_lam=mixup_lam, nonzero_padding=nonzero_padding, selected_se_pair_idx=selected_se_pair_idx)
            imgs.append(img)
            points.append(point)
            visibles.append(visible)
            links.append(link)
            metas.append(meta)
            cids.append(meta['category_id'])
            headwise_refpoint_idxs.append(headwise_refpoint_idx)

        data = dict()
        data['imgs'] = torch.stack(imgs)
        data['points'] = torch.stack(points)
        data['visibles'] = torch.stack(visibles)
        data['links'] = torch.stack(links)
        data['cid'] = torch.tensor(cids)
        data['headwise_refpoint_idxs'] = torch.stack(headwise_refpoint_idxs)
        data['category_keypoint_num'] = torch.tensor(meta['category_keypoint_num'])
        return data
