from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = '../configs/B20/vild_finetune_c7_infer_c20.py'
checkpoint_file = '../work_dirs/vild_finetune_c7_ensemble_PLUS_WithC7/epoch_12.pth'

# config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = '/home/yechenzhi/.jupyter/ObjectDetection/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_coco/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/home/yechenzhi/data/B20/valB20/BV1ca4y1W7h7_229.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg',score_thr=0.07)
