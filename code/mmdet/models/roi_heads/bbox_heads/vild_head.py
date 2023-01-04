# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead

import numpy as np
import torch.nn.functional as F

@HEADS.register_module()
class ViLDHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=4,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 class_weights=None,
                 temperature=100,
                 *args,
                 **kwargs):
        super(ViLDHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.class_weights = torch.tensor(np.load(class_weights),dtype=torch.float32).T.cuda()
        self.class_weights_norm = torch.norm(self.class_weights, p=2, dim=0, keepdim=True)
        self.clip_dim, self.num_classes_to_check = self.class_weights.shape
        assert self.num_classes_to_check==self.num_classes, 'clip_weights shape[0] should be equal to num_classes'
        self.temperature = temperature
        

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        #add x_cls projection 
        self.proj2clip = build_linear_layer(
            self.cls_predictor_cfg,
            in_features=self.shared_out_channels,
            out_features=self.clip_dim)
        
        
        self.relu = nn.ReLU(inplace=True)

        # ---------------- VILD KD HEAD BEGIN---------------- 
        
        # add kd shared convs and fcs
        self.shared_kd_convs, self.shared_kd_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        
        #add x_cls projection 
        self.kd_proj2clip = build_linear_layer(
            self.cls_predictor_cfg,
            in_features=self.shared_out_channels,
            out_features=self.clip_dim)
         
        # ---------------- VILD KD HEAD END----------------    
                
        # fc_cls for bg
        self.fc_cls = build_linear_layer(
            self.cls_predictor_cfg,
            in_features=self.clip_dim,
            out_features=1,
            bias=False)
        nn.init.normal_(self.fc_cls.weight,mean=0.0,std=0.01)

        
        self.reg_last_dim = self.shared_out_channels
        
        out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                        self.num_classes)
        self.fc_reg = build_linear_layer(
            self.reg_predictor_cfg,
            in_features=self.reg_last_dim,
            out_features=out_dim_reg)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    override=[
                        dict(name='shared_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # shared part
        # x:(1024,256,7,7)
        identity = x
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x) #[2048,256,7,7]
        x = x + identity
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        # import pdb; pdb.set_trace()
        x_cls = x #(2048,1024)
        x_reg = x #(2048,1024)

        #Project and Norm
        x_cls = self.proj2clip(x_cls) #2048,512 
        # import pdb; pdb.set_trace()
        x_cls = torch.div(x_cls,torch.norm(x_cls,p=2,dim=-1,keepdim=True)) 

        cls_score = torch.mm(x_cls, self.class_weights)
        cls_score = torch.div(cls_score, self.class_weights_norm)
        bg_score = self.fc_cls(x_cls)
        # import pdb; pdb.set_trace()
        bg_score = torch.div(bg_score,torch.norm(self.fc_cls.weight,p=2,dim=-1,keepdim=True))

        cls_score = torch.cat([cls_score,bg_score],dim=-1)
        cls_score *= self.temperature
        bbox_pred = self.fc_reg(x_reg) 
        return cls_score, bbox_pred, x_cls
    
    def forward_kd(self, x):
        # shared part
        # x:(1024,256,7,7)
        identity = x
        if self.num_shared_convs > 0:
            for conv in self.shared_kd_convs:
                x = conv(x) #[2048,256,7,7]
        x = x + identity
        #TODO: check if we nned new avg_pool layer and relu layer
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_kd_fcs:
                x = self.relu(fc(x))


        #Project and Norm
        x = self.kd_proj2clip(x) #2048,512 
        embs = torch.div(x,torch.norm(x,p=2,dim=-1,keepdim=True)) 

        cls_score = torch.mm(embs, self.class_weights)
        cls_score = torch.div(cls_score, self.class_weights_norm)
        
        bg_score = torch.zeros(cls_score.size(0),1).cuda()
        cls_score = torch.cat([cls_score,bg_score],dim=-1)
        return embs, cls_score

    def infer(self, x):
        # shared part
        # x:(1024,256,7,7)
        # import pdb; pdb.set_trace()
        x_kd = x

        identity = x
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x) #[2048,256,7,7]
        x = x + identity
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        # import pdb; pdb.set_trace()
        x_cls = x #(2048,1024)
        x_reg = x #(2048,1024)

        #Project and Norm
        x_cls = self.proj2clip(x_cls) #2048,512 
        # import pdb; pdb.set_trace()
        x_cls = torch.div(x_cls,torch.norm(x_cls,p=2,dim=-1,keepdim=True)) 

        self.infer_weights =  torch.tensor(np.load('/home/yechenzhi/.jupyter/B20/mmdetection/weights/c21.npy'),dtype=torch.float32)[:,:20].cuda()
        cls_score = torch.mm(x_cls, self.infer_weights)
        self.infer_weights_norm = torch.norm(self.infer_weights, p=2, dim=0, keepdim=True)
        cls_score = torch.div(cls_score, self.infer_weights_norm)
        bg_score = self.fc_cls(x_cls)
        # import pdb; pdb.set_trace()
        bg_score = torch.div(bg_score,torch.norm(self.fc_cls.weight,p=2,dim=-1,keepdim=True))

        cls_score = torch.cat([cls_score,bg_score],dim=-1)
        cls_score *= self.temperature
        cls_score = F.softmax(cls_score,dim=-1)
        bbox_pred = self.fc_reg(x_reg)

        ###FOR KD####
        # import pdb; pdb.set_trace()
        embs, _ = self.forward_kd(x_kd)
        kd_score = torch.mm(embs, self.infer_weights)
        # self.infer_weights_norm = torch.norm(self.infer_weights, p=2, dim=0, keepdim=True)
        kd_score = torch.div(kd_score, self.infer_weights_norm)
        bg_score = self.fc_cls(embs)
        # import pdb; pdb.set_trace()
        bg_score = torch.div(bg_score,torch.norm(self.fc_cls.weight,p=2,dim=-1,keepdim=True))

        bg_score = torch.zeros(kd_score.size(0),1).cuda()#TEST

        kd_score = torch.cat([kd_score,bg_score],dim=-1)
        kd_score *= self.temperature
        kd_score = F.softmax(kd_score,dim=-1)
        
        mask = torch.tensor([1,1,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,0]).cuda()

        txt_score = torch.pow(cls_score,5/6)*torch.pow(kd_score,1/6)*mask
        img_score = torch.pow(cls_score,1/6)*torch.pow(kd_score,5/6)*(1-mask)
        score = txt_score + img_score

        # base_score = 0.9 * cls_score * mask + 0.1 * kd_score * mask
        # novl_score = 0.1 * cls_score * (1-mask) + 0.9 * kd_score * (1-mask)
        # score = base_score+novl_score

        ###END KD###
        return score, bbox_pred, x_cls

    



