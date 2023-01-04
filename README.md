# B20

B20数据集来源于B站视频，经过人工标注后，用于评测开放世界视频目标检测任务。

## 数据

图片数据在 data 文件夹下，标注在 data/annotations 文件夹下，标注格式参考[COCO目标检测数据集](https://cocodataset.org/#format-data)给出。

## 代码

代码均在 code 文件夹下，代码采用 mmdetection 框架，具体安装和使用可参考[mmdetection](https://github.com/open-mmlab/mmdetection)，其中主要训练和评估的config文件在 code/configs/B20 文件夹下。

