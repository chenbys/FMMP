from mmpose.datasets import DATASETS
import random
import numpy as np
import os
from collections import OrderedDict
from xtcocotools.coco import COCO
from capeformer.datasets.datasets.mp100.test_dataset import TestPoseDataset
import copy
import torch
from mmcv.parallel import DataContainer as DC
from .side_func import get_side_ref_idx_and_orders
from .pad_func import *


@DATASETS.register_module()
class CJCTestSet(TestPoseDataset):
    def __init__(self, ann_file, img_prefix, data_cfg, pipeline, valid_class_ids,
                 max_kpt_num=None, num_shots=1, num_queries=100, num_episodes=1,
                 pck_threshold_list=[0.05, 0.1, 0.15, 0.20], test_mode=True, keypoint_padding={}):

        super().__init__(ann_file=ann_file, img_prefix=img_prefix, data_cfg=data_cfg, pipeline=pipeline,
                         valid_class_ids=valid_class_ids, max_kpt_num=max_kpt_num, num_shots=num_shots,
                         num_queries=num_queries, num_episodes=num_episodes, pck_threshold_list=pck_threshold_list)
        self.keypoint_padding = keypoint_padding
        self.is_train = False
        self.cid_to_K_max = {}
        for cid, d in self.cats.items():
            all_ids = []
            for se in d['skeleton']:
                all_ids += se
            self.cid_to_K_max[cid] = max(all_ids) + 1
        # [len(v['skeleton']) for v in self.cats.values()]
        return

    def make_paired_samples(self):
        random.seed(1)
        np.random.seed(0)
        all_samples = []
        for cls in self.valid_class_ids:
            for _ in range(self.num_episodes):
                shots = random.sample(self.cat2obj[cls], self.num_shots + self.num_queries)
                sample_ids = shots[:self.num_shots]
                query_ids = shots[self.num_shots:]
                for query_id in query_ids:
                    all_samples.append(sample_ids + [query_id])

        self.paired_samples = np.array(all_samples)

    def random_paired_samples(self):
        raise NotImplementedError

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
        keypoint, visible, se_pairs = self.pad_pose(keypoint=keypoint, visible=visible, se_pairs=se_pairs,
                                                    category_keypoint_num=category_keypoint_num, **kwargs)

        link = torch.zeros(len(visible), len(visible))
        for s, e in se_pairs:
            if visible[s] and visible[e]:
                link[s, e] = 1
                link[e, s] = 1
        # keypoint: [K,2]
        # head-wise refpoint [K,N,2]
        # [P;linked(P); P*(N-1)][:N].
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

        dids = []
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
            dids.append(did)

        data = dict()
        data['imgs'] = torch.stack(imgs)
        data['points'] = torch.stack(points)
        data['visibles'] = torch.stack(visibles)
        data['links'] = torch.stack(links)
        data['cid'] = torch.tensor(cids)
        data['meta_q'] = DC(meta, cpu_only=True)
        data['headwise_refpoint_idxs'] = torch.stack(headwise_refpoint_idxs)
        data['did'] = torch.tensor(dids)
        data['category_keypoint_num'] = torch.tensor(meta['category_keypoint_num'])

        # img_metas = dict()
        # for key in xq_img_metas.keys():
        #     img_metas['sample_' + key] = [xs_img_meta[key] for xs_img_meta in xs_img_metas]
        #     img_metas['query_' + key] = xq_img_metas[key]
        # img_metas['bbox_id'] = idx
        # Xall['img_metas'] = DC(img_metas, cpu_only=True)
        return data

    def evaluate(self, outputs, res_folder, metric='PCK', respostfix='', **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'AUC', 'EPE', 'NME']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, f'result{respostfix}_keypoints.json')

        kpts = []
        for output in outputs:
            preds = output[f'preds']
            boxes = output[f'boxes']
            image_paths = output[f'image_paths']
            bbox_ids = output[f'bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]

                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        # kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)
        return name_value
