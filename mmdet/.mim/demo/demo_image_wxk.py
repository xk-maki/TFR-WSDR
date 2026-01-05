from mmdet.apis import init_detector

config_file = '../work_dirs/cornernet_hourglass104_10xb5-crop511-210e-mstest_coco/cornernet_hourglass104_10xb5-crop511-210e-mstest_coco.py'

checkpoint_file = '../work_dirs/cornernet_hourglass104_10xb5-crop511-210e-mstest_coco/epoch_12.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
total = sum([param.nelement() for param in model.parameters()])
print(total)