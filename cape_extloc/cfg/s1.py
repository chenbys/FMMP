from cape_extloc.cfg.info import *

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=200)
evaluation = dict(interval=10,
                  metric=['PCK'],
                  key_indicator='mPCK',
                  gpu_collect=True,
                  res_folder='')

optimizer = dict(type='Adam', lr=1e-4, )

optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=0.001,
                 step=[160, 180])

total_epochs = 200
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), ])

channel_cfg = dict(num_output_channels=1,
                   dataset_joints=1,
                   dataset_channel=[[0, ], ],
                   inference_channel=[0, ], max_kpt_num=70)

model = dict(type='ExtLocDetector',
             pretrained='torchvision://resnet50',
             encoder_config=dict(type='ResNet', depth=50, out_indices=(1, 2, 3,)),
             predictor=dict(res_layer_num=3,
                            side_head=8,
                            use_joint_encoder=False,
                            use_self_att=True,
                            top_down_fuse=False,
                            query_init='cst',
                            embed_init='ref',
                            nhead=8,
                            npoint=4,
                            ref_feat_aware=False,
                            use_deform_encoder=False,
                            use_deform_decoder=True,
                            d_model=256,
                            num_encoder_layers=3,
                            num_decoder_layers=3,
                            dim_feedforward=256,
                            dropout=0.1,
                            activation="relu",
                            normalize_before=False,
                            return_intermediate_dec=True,
                            reuse_layers=False
                            ),
             train_cfg=dict(mix_weight=0.5, viz_interval=1e8),
             test_cfg=dict(flip_test=False,
                           post_process='default',
                           shift_heatmap=True,
                           modulate_kernel=11,
                           viz_interval=1e8,
                           viz_layers=True,
                           viz_keypoints=True, ),
             viz_cfg=dict(epoch_interval=0,
                          sample_interval=5,
                          viz_preds=True,
                          viz_points=False,
                          viz_layers=False,
                          viz_pad_num=0
                          )
             )

data_cfg = dict(image_size=[256, 256],
                heatmap_size=[64, 64],
                num_output_channels=channel_cfg['num_output_channels'],
                num_joints=channel_cfg['dataset_joints'],
                dataset_channel=channel_cfg['dataset_channel'],
                inference_channel=channel_cfg['inference_channel'])

viz_data_cfg = dict(image_size=[512, 512],
                    heatmap_size=[64, 64],
                    num_output_channels=channel_cfg['num_output_channels'],
                    num_joints=channel_cfg['dataset_joints'],
                    dataset_channel=channel_cfg['dataset_channel'],
                    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [dict(type='LoadImageFromFile'),
                  dict(type='TopDownGetRandomScaleRotation', rot_factor=15, scale_factor=0.15),
                  dict(type='TopDownAffineFewShot'),
                  dict(type='ToTensor'),
                  # dict(type='ColorAug', jiggle=(0.3, 0.3, 0.3, 0.3), pj=1.0),
                  dict(type='NormalizeTensor',
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
                  dict(type='TopDownGenerateTargetFewShot', sigma=2),
                  dict(type='Collect',
                       keys=['img', 'target', 'target_weight'],
                       meta_keys=[
                           'image_file', 'joints_3d', 'joints_3d_visible',
                           'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs', 'category_id'
                       ]),
                  ]

valid_pipeline = [dict(type='LoadImageFromFile'),
                  dict(type='TopDownAffineFewShot'),
                  dict(type='ToTensor'),
                  dict(type='NormalizeTensor',
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
                  dict(type='TopDownGenerateTargetFewShot', sigma=2),
                  dict(type='Collect',
                       keys=['img', 'target', 'target_weight'],
                       meta_keys=[
                           'image_file', 'joints_3d', 'joints_3d_visible',
                           'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs', 'category_id'
                       ]),
                  ]

test_pipeline = valid_pipeline

data_root = 'data/mp100'
unified_ann_file = f'{data_root}/annotations/mp100_all_link_0412.json'
unified_img_prefix = f'{data_root}/mp100_images_cc'
num_shots = 1

train_keypoint_padding = dict(type='mixup',
                              num=70, prob=1., alpha=1.)
test_keypoint_padding = dict(type='eqdiv',
                             num=70, prob=1., alpha=1.)

data = dict(samples_per_gpu=16, workers_per_gpu=8,
            train=dict(type='CJCTrainSet',
                       ann_file=unified_ann_file,
                       img_prefix=unified_img_prefix,
                       data_cfg=data_cfg,
                       valid_class_ids=train_cids_split1,
                       max_kpt_num=channel_cfg['max_kpt_num'],
                       num_shots=num_shots,
                       pipeline=train_pipeline,
                       epoch_sample_num=None,
                       keypoint_padding=train_keypoint_padding
                       ),
            val=dict(type='CJCTestSet',
                     ann_file=unified_ann_file,
                     img_prefix=unified_img_prefix,
                     data_cfg=data_cfg,
                     valid_class_ids=val_cids_split1,
                     max_kpt_num=channel_cfg['max_kpt_num'],
                     num_shots=num_shots,
                     num_queries=15,
                     num_episodes=5,
                     pck_threshold_list=[0.05, 0.10, 0.15, 0.2],
                     pipeline=test_pipeline,
                     keypoint_padding=test_keypoint_padding
                     ),
            test=dict(type='CJCTestSet',
                      ann_file=unified_ann_file,
                      img_prefix=unified_img_prefix,
                      data_cfg=data_cfg,
                      valid_class_ids=test_cids_split1,
                      max_kpt_num=channel_cfg['max_kpt_num'],
                      num_shots=num_shots,
                      num_queries=15,
                      num_episodes=200,
                      pck_threshold_list=[0.05, 0.10, 0.15, 0.2],
                      pipeline=test_pipeline,
                      keypoint_padding=test_keypoint_padding
                      ),
            )

shuffle_cfg = dict(interval=1)
