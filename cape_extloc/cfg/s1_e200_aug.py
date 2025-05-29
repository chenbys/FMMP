from cape_extloc.cfg.s1 import *

data['samples_per_gpu'] = 4
data['workers_per_gpu'] = 4

data['val']['num_episodes'] = 10

evaluation['interval'] = 10

lr_config['step'] = [160, 180]
total_epochs = 200
checkpoint_config['interval'] = 200

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.4),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    # dict(type='ColorAug', jiggle=(0.3, 0.3, 0.3, 0.3), pj=1.0),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=2),
    dict(type='Collect', keys=['img', 'target', 'target_weight'],
         meta_keys=['image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
                    'rotation', 'bbox_score', 'flip_pairs', 'category_id']),
]
