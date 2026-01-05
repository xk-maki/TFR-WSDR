_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/torchsig_detection_RGB.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[223.3, 191.8, 30.8],
        std=[46.215, 65.776, 46.3],
        bgr_to_rgb=True,
        pad_size_divisor=32),
)

train_dataloader = dict(
    batch_size=8)

val_dataloader = dict(
    batch_size=2)

