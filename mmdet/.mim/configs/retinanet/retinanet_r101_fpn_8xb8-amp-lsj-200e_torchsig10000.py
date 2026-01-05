_base_ = './retinanet_r50_fpn_8xb8-amp-lsj-200e_torchsig10000.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.00004))