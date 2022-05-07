find_unused_parameters=True
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'
data_root = '/home/yechenzhi/data/B20/'
classes = ('cat', 'dog', 'motocycle', 'car', 'bicycle', 'phone', 'skateboard')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 360), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
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
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


trainA=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=data_root + 'annotations/trainB20v2.json',
        img_prefix=data_root + 'trainB20/',
        pipeline=train_pipeline)

trainB=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/yechenzhi/data/B20/C7/sampled_c7v2.json',
        img_prefix='/dataset/COCO/train2017/',
        pipeline=train_pipeline)


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=[trainA,trainB],
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=data_root + 'annotations/valB20v2.json',
        img_prefix=data_root + 'valB20/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=data_root + 'annotations/valB20v2.json',
        img_prefix=data_root + 'valB20/',
        pipeline=test_pipeline))

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=7,
        reg_class_agnostic=True)
    )
)


optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])
# the max_epochs and step in lr_config need specifically tuned for the customized dataset
# runner = dict(max_epochs=8)
log_config = dict(interval=100)

load_from = '/home/yechenzhi/.jupyter/ObjectDetection/mmdetection/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'