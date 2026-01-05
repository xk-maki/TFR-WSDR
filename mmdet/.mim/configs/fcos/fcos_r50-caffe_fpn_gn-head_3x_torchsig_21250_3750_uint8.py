_base_ = [
    '../_base_/datasets/torchsig_detection_uint8.py',
    '../_base_/schedules/schedule_paper3.py', '../_base_/default_runtime.py'
]
train_dataloader = dict(
    batch_size=8)

val_dataloader = dict(
    batch_size=2)

# model settings
model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[87.43],
        std=[25],
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=1,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        # init_cfg=dict(type='Pretrained', checkpoint='checkpoints/resnet50-19c8e357.pth')
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=6,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# # learning rate
# param_scheduler = [
#     dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=24,
#         by_epoch=True,
#         milestones=[18, 11],
#         gamma=0.1)
# ]

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.001),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))
