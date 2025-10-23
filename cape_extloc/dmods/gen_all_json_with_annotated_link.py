import json
import numpy as np

# split-1
test_cids = [10, 77, 24, 47, 33, 29, 70, 68, 81, 73, 2, 14, 30, 53, 42, 3, 39, 84, 78, 60]
val_cids = [48, 22, 12, 91, 96, 6, 35, 95, 66, 92]
train_cids = [37, 36, 56, 89, 65, 69, 74, 19, 9, 23, 79, 13, 93, 82, 99, 94, 34, 4, 61, 57, 51, 52, 7, 85, 21, 90, 76,
              38, 5, 63, 59, 50, 88, 100, 31, 32, 26, 41, 71, 15, 87, 54, 43, 27, 58, 55, 16, 20, 28, 83, 44, 46, 86,
              98, 17, 49, 45, 40, 8, 64, 80, 62, 11, 18, 1, 97, 75, 67, 25, 72]

all_super_categories = {'Giraffidae', 'hand', 'Castoridae', 'Ursidae', 'Mephitidae', 'Elephantidae', 'Cercopithecidae',
                        'animal', 'Muridae', 'bird', 'clothes', 'Felidae', 'furniture', 'Sciuridae', 'Rhinocerotidae',
                        'Cricetidae', 'Leporidae', 'Hippopotamidae', 'Canidae', 'Mustelidae', 'Bovidae', 'Suidae',
                        'Hominidae', 'vehicle', 'Equidae', 'person', 'Procyonidae', 'Cervidae', 'animal_face'}

if __name__ == '__main__':
    # img_root = '/home/chenjunjie/workspace/datasets/PoseDataset/mp100/mp100_images_cc/'
    with open(f'data/mp100/annotations/mp100_split1_all.json', 'r') as f:
        all_json = json.load(f)
    path = 'data/keypoint_pairs.npy'
    link_anno = np.load(path, allow_pickle=True)
    new_cid_2_link = {}
    for L in link_anno:
        new_cid_2_link[L[-1]] = L[:-1]
    alllink_json = {}
    for k in ['images', 'annotations', 'categories', 'info']:
        alllink_json[k] = all_json[k]
    '''
    if cid in [32, 38, 51] or new_links:
    se_offset = False
    '''
    for i in range(100):
        cdict = alllink_json['categories'][i]
        cid = cdict['id']

        old_links = all_json['categories'][i]['skeleton']
        if len(old_links) == 0:
            new_links = new_cid_2_link[cid]
        else:
            if cid in [84, 39]:
                new_links = old_links
            elif cid in [32, 38, 51]:
                new_links = old_links
            else:
                new_links = []
                for old_link in old_links:
                    s, e = old_link
                    assert s >= 1
                    assert e >= 1
                    new_links.append([s - 1, e - 1])
        alllink_json['categories'][i]['skeleton'] = new_links
        i

    with open(f'data/mp100/annotations/mp100_all_link.json', 'w') as f:
        json.dump(alllink_json, f)
