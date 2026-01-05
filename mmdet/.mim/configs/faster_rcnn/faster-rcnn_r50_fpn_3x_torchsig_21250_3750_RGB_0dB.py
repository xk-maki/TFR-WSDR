_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/torchsig_detection_RGB_0dB.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[247.54669351, 116.82226353,   5.99057147],
        std=[25.33268888, 72.41239987 ,24.31408246],
        bgr_to_rgb=True,
        pad_size_divisor=32),
)

train_dataloader = dict(
    batch_size=8)

val_dataloader = dict(
    batch_size=2)

