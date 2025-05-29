import torch
import numpy as np


def get_zero_padding(**kwargs):
    K_max = kwargs['K_max']
    ori_keypoint = kwargs['keypoint']
    ori_visible = kwargs['visible']
    K_pad = K_max - len(ori_keypoint)
    all_keypoints = torch.cat([ori_keypoint] + [ori_keypoint.new_zeros(K_pad, 2)])
    all_visibles = torch.cat([ori_visible] + [ori_visible.new_zeros(K_pad)])
    return all_keypoints, all_visibles, kwargs['se_pairs']


def get_eqdiv_padding(**kwargs):
    K_c = kwargs['category_keypoint_num']
    K_max = kwargs['K_max']
    ori_keypoint = kwargs['keypoint'][:K_c]
    ori_visible = kwargs['visible'][:K_c]
    ori_se_pairs = kwargs['se_pairs']
    nonzero_padding = kwargs['nonzero_padding']
    K_pad = K_max - K_c
    L_c = len(ori_se_pairs)
    eqdiv_point_num, tail_zero_num = divmod(K_pad, L_c)
    pad_keypoints = []
    pad_visibles = []
    pad_se_pairs = []
    if eqdiv_point_num != 0:
        pad_start_id = K_c
        for se_pair in ori_se_pairs:
            parent_start, parent_end = se_pair
            pair_visible = ori_visible[parent_start] & ori_visible[parent_end]
            eqdiv_visible = ori_visible.new(
                nonzero_padding[pad_start_id:(pad_start_id + eqdiv_point_num)]) * pair_visible

            splits = torch.linspace(0, 1, 2 + eqdiv_point_num)[1:-1][:, None]
            eqdiv_point = splits * ori_keypoint[parent_start][None] + (1 - splits) * ori_keypoint[parent_end][None]

            pad_keypoints.append(eqdiv_point)
            pad_visibles.append(eqdiv_visible)
            prev_link_start = parent_start
            eqdiv_se_pairs = []

            if eqdiv_visible.max().item():
                for i, v in enumerate(eqdiv_visible):
                    current_id = pad_start_id + i
                    if v:
                        new_pair = [prev_link_start, current_id]
                        eqdiv_se_pairs.append(new_pair)
                        prev_link_start = current_id

                if prev_link_start != parent_start:
                    eqdiv_se_pairs.append([prev_link_start, parent_end])
            pad_se_pairs += eqdiv_se_pairs
            pad_start_id += eqdiv_point_num

    all_keypoints = torch.cat([ori_keypoint] + pad_keypoints + [ori_keypoint.new_zeros(tail_zero_num, 2)])
    all_visibles = torch.cat([ori_visible] + pad_visibles + [ori_visible.new_zeros(tail_zero_num)])
    all_se_pairs = ori_se_pairs + pad_se_pairs

    return all_keypoints, all_visibles, all_se_pairs


def get_mixup_padding(**kwargs):
    K_c = kwargs['category_keypoint_num']
    K_max = kwargs['K_max']
    ori_keypoint = kwargs['keypoint'][:K_c]
    ori_visible = kwargs['visible'][:K_c]
    ori_se_pairs = kwargs['se_pairs']
    mixup_lam = kwargs['mixup_lam']
    nonzero_padding = kwargs['nonzero_padding']
    # visible_linked_pairs = [pair for pair in ori_se_pairs if (ori_visible[pair[0]] & ori_visible[pair[1]]).item()]
    selected_se_pairs = kwargs['selected_se_pair_idx'][:K_max - K_c]

    pad_keypoints = []
    pad_visibles = []
    pad_se_pairs = []
    pad_start_id = K_c
    for sid in np.unique(selected_se_pairs):
        mixup_point_num = sum(selected_se_pairs == sid)
        parent_start, parent_end = ori_se_pairs[sid]
        pair_visible = ori_visible[parent_start] & ori_visible[parent_end]
        mixup_visible = ori_visible.new(nonzero_padding[pad_start_id:(pad_start_id + mixup_point_num)]) * pair_visible
        lam = mixup_lam[pad_start_id:(pad_start_id + mixup_point_num)]
        lam = ori_keypoint.new(lam).sort()[0][:, None]
        mixup_point = lam * ori_keypoint[parent_start][None] + (1 - lam) * ori_keypoint[parent_end][None]
        pad_keypoints.append(mixup_point)
        pad_visibles.append(mixup_visible)
        prev_link_start = parent_start
        mixup_se_pairs = []
        if mixup_visible.max().item():
            # update links
            for i, v in enumerate(mixup_visible):
                current_id = pad_start_id + i
                if v:
                    new_pair = [prev_link_start, current_id]
                    mixup_se_pairs.append(new_pair)
                    prev_link_start = current_id

            if prev_link_start != parent_start:
                mixup_se_pairs.append([prev_link_start, parent_end])
        pad_se_pairs += mixup_se_pairs
        pad_start_id += mixup_point_num

    all_keypoints = torch.cat([ori_keypoint] + pad_keypoints)
    all_visibles = torch.cat([ori_visible] + pad_visibles)
    all_se_pairs = ori_se_pairs + pad_se_pairs
    assert all_keypoints.size(0) == all_visibles.size(0) == K_max
    return all_keypoints, all_visibles, all_se_pairs
