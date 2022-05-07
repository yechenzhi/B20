find_unused_parameters=True
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'
data_root = '/home/yechenzhi/data/B20/'
classes = classes = ('cat', 'dog', 'pig', 'rabbit', 'parrot', 'snake', 'tiger', 'seal', 'panda', 'mouse', 'hamster', 'motocycle', 'car','sports_car', 'bicycle', 'phone', 'baby', 'Ultraman', 'skateboard', 'high_heels')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 360), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 360),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

trainB=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/yechenzhi/data/B20/C7/sampled_c7v2.json',
        img_prefix='/dataset/COCO/train2017/',
        pipeline=test_pipeline)


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=data_root + 'annotations/trainB20v4.json',
        img_prefix=data_root + 'trainB20/',
        pipeline=train_pipeline),
    val=trainB,
    test=trainB)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=7,
        reg_class_agnostic=True),
    )
)

