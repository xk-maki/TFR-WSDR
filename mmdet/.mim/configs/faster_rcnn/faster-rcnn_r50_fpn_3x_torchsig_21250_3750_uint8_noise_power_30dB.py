_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/torchsig_detection_uint8_noise_power.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='FasterRCNN',
    backbone=dict(
        in_channels=1,
    ),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[174.42],
        std=[47.15],
        pad_size_divisor=32),
)

train_dataloader = dict(
    batch_size=8)

val_dataloader = dict(
    batch_size=2)

auto_scale_lr = dict(enable=True, base_batch_size=16)
