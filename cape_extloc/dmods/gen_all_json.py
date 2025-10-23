import json

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
    for split_id in [1, 2, 3, 4, 5]:
        # coco_train = COCO(f'data/mp100/annotations/mp100_split{split_id}_train.json')
        # coco_val = COCO(f'data/mp100/annotations/mp100_split{split_id}_val.json')
        # coco_test = COCO(f'data/mp100/annotations/mp100_split{split_id}_test.json')
        # img_root = '/home/chenjunjie/workspace/datasets/PoseDataset/mp100/mp100_images_cc/'

        with open(f'data/mp100/annotations/mp100_split{split_id}_train.json', 'r') as f:
            train_json = json.load(f)
        with open(f'data/mp100/annotations/mp100_split{split_id}_val.json', 'r') as f:
            val_json = json.load(f)
        with open(f'data/mp100/annotations/mp100_split{split_id}_test.json', 'r') as f:
            test_json = json.load(f)

        valtest_json = {}
        for k in ['images', 'annotations', 'categories']:
            valtest_json[k] = val_json[k] + test_json[k]
        valtest_json['info'] = test_json['info']
        # with open(f'data/mp100/annotations/mp100_split{split_id}_valtest.json', 'w') as f:
        #     json.dump(valtest_json, f)

        # all_json = {}
        # for k in ['images', 'annotations', 'categories']:
        #     all_json[k] = train_json[k] + val_json[k] + test_json[k]
        # all_json['info'] = train_json['info']
        #
        # test_cids = [d['id'] for d in test_json['categories']]
        # val_cids = [d['id'] for d in val_json['categories']]
        # train_cids = [d['id'] for d in train_json['categories']]
        # with open(f'data/mp100/annotations/mp100_split{split_id}_all.json', 'w') as f:
        #     json.dump(all_json, f)
        with open(f'data/mp100/annotations/mp100_split{split_id}_all.json', 'r') as f:
            all_json = json.load(f)

        for sc in all_super_categories:
            sckps = [d['keypoints'] for d in valtest_json['categories'] if d['supercategory'] == sc]
            if len(sckps) == 0:
                continue
            print(f'--------------------------------------------------------')
            print(sc)
            print([len(sk) for sk in sckps])
            print(sckps[0])
            sc
        split_id
