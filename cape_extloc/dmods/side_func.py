from collections import deque
import torch


def get_side_ref_idx_and_orders(visible, se_pairs, link):
    new_manner = True
    method_type = 3
    if method_type==1:
        link_table = {}
        for point_id, point_visible in enumerate(visible):
            if point_visible:
                link_table[point_id] = []
                for se_pair in se_pairs:
                    if se_pair[1] == point_id:
                        link_table[point_id].append(se_pair[0])
                    if se_pair[0] == point_id:
                        link_table[point_id].append(se_pair[1])
            else:
                link_table[point_id] = []

        K = len(link)
        N_max_pad = 16
        headwise_refpoint_idxs = []
        headwise_refpoint_idxs_order = []
        for k in range(K):
            self_idxs = torch.ones_like(visible).long() * k
            if visible[k].item():
                headwise_refpoint = [self_idxs[:1].item()]
                headwise_refpoint_order = [0]
                recent_add = [self_idxs[:1].item()]
                current_order = 1
                while len(headwise_refpoint) < N_max_pad:
                    next_add = set()
                    for r in recent_add:
                        next_add.update(set(link_table[r]))
                    headwise_refpoint.extend(list(next_add))
                    headwise_refpoint_order.extend([current_order for s in next_add])
                    recent_add = list(next_add)
                    current_order += 1
                    if len(recent_add) == 0:
                        break
                if len(headwise_refpoint) < N_max_pad:
                    headwise_refpoint.extend(self_idxs[:N_max_pad].tolist())
                    headwise_refpoint_order.extend((self_idxs[:N_max_pad] * 0).tolist())

                headwise_refpoint_idxs.append(torch.tensor(headwise_refpoint[:N_max_pad]))
                headwise_refpoint_idxs_order.append(torch.tensor(headwise_refpoint_order[:N_max_pad]))
            else:
                headwise_refpoint_idxs.append(self_idxs[:N_max_pad])
                headwise_refpoint_idxs_order.append(self_idxs[:N_max_pad] * 0)
    elif method_type==2:
        K = len(link)
        N_max_pad = 16
        headwise_refpoint_idxs = []
        for k in range(K):
            self_idxs = torch.ones_like(visible).long() * k
            if visible[k].item():
                headwise_refpoint = [self_idxs[:1]]
                linkpoints = torch.where(link[k].bool())[0][:(N_max_pad - 1)]
                headwise_refpoint.append(linkpoints)
                pad_len = N_max_pad - 1 - len(linkpoints)
                headwise_refpoint.append(self_idxs[:pad_len])
                headwise_refpoint_idxs.append(torch.cat(headwise_refpoint))
            else:
                headwise_refpoint_idxs.append(self_idxs[:N_max_pad])
    elif method_type==3:
        link_table = {}
        for point_id, point_visible in enumerate(visible):
            if point_visible:
                link_table[point_id] = []
                for se_pair in se_pairs:
                    if se_pair[1] == point_id:
                        link_table[point_id].append(se_pair[0])
                    if se_pair[0] == point_id:
                        link_table[point_id].append(se_pair[1])
            else:
                link_table[point_id] = []

        K = len(link)
        N_max_pad = 16
        headwise_refpoint_idxs = []
        headwise_refpoint_idxs_order = []
        for k in range(K):
            self_idxs = torch.ones_like(visible).long() * k
            if visible[k].item():
                visited = set()
                visited.add(self_idxs[:1].item())
                queue = deque([(k, 0)])
                headwise_refpoint = [self_idxs[:1].item()]
                headwise_refpoint_order = [0]

                while queue and len(headwise_refpoint) < N_max_pad:
                    current_node, level = queue.popleft()
                    for node in link_table.get(current_node, []):
                        if node not in visited:
                            queue.append((node, level + 1))
                            visited.add(node)
                            headwise_refpoint.append(node)
                            headwise_refpoint_order.append(level + 1)
                if len(headwise_refpoint) < N_max_pad:
                    headwise_refpoint.extend(self_idxs[:N_max_pad].tolist())
                    headwise_refpoint_order.extend((self_idxs[:N_max_pad] * 0).tolist())
                headwise_refpoint_idxs.append(torch.tensor(headwise_refpoint[:N_max_pad]))
                headwise_refpoint_idxs_order.append(torch.tensor(headwise_refpoint_order[:N_max_pad]))
            else:
                headwise_refpoint_idxs.append(self_idxs[:N_max_pad])
                headwise_refpoint_idxs_order.append(self_idxs[:N_max_pad] * 0)
    headwise_refpoint_idx = torch.stack(headwise_refpoint_idxs)
    headwise_refpoint_idx_order = torch.stack(headwise_refpoint_idxs_order)
    return torch.stack((headwise_refpoint_idx, headwise_refpoint_idx_order))
