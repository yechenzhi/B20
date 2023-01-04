# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class ViLDDataset(CocoDataset):
    
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        ################################
        #add fields for kd_loss
        root = '/home/yechenzhi/.jupyter/B20/mmdetection/weights/bbox_embs_from_c7'
        file = img_info['filename'].split('.')[0]+'.npy'
        file = osp.join(root,file)
        try:
            dic = np.load(file,allow_pickle=True).item()
            results['bboxes'] = dic['bboxes']
            results['embs'] = dic['embs']
        except FileNotFoundError: 
            results['embs'] = None
            results['bboxes'] = None
            ################################
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
            # import pdb; pdb.set_trace()
        self.pre_pipeline(results)
        
        return self.pipeline(results)

   