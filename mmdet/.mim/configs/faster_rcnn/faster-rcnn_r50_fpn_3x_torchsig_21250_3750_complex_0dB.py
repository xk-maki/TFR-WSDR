_base_ = [
    '../_base_/models/faster-rcnn_complex_r50_fpn.py',
    '../_base_/datasets/torchsig_detection_complex_0dB.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]


model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='ComplexDetDataPreprocessor',
        pad_size_divisor=32),
)

train_dataloader = dict(
    batch_size=8)

val_dataloader = dict(
    batch_size=2)

