_base_ = './retinanet_r50_fpn_8xb8-amp-lsj-200e_torchsig10000.py'

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))
