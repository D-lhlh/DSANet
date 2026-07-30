from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmengine.runner import CheckpointLoader
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList, ConfigType


class ohem_head(BaseModule):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='GELU'),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=('norm', 'act', 'conv'))
        _, self.norm = build_norm_layer(norm_cfg, num_features=channels)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor, cls_seg: Optional[nn.Module]) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if cls_seg is not None:
            x = cls_seg(x)
        return x


@MODELS.register_module()
class OHEM_HEAD(BaseDecodeHead):
    def __init__(self,
                 in_channels: list,
                 channels: int,
                 num_classes: int,
                 loss_detail_weight=1.0,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='GELU', inplace=True),
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)
        self.act_cfg = act_cfg

        self.loss_detail_weight = loss_detail_weight
        self.ohemhead = ohem_head(self.in_channels, channels, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def init_weights(self):
        """Initialize the weights in backbone.

        """
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `checkpoint` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], map_location='cpu')['state_dict']
            self.prefix = self.init_cfg['prefix']
            backbone_ckpt = {}
            for key, value in ckpt.items():
                if key.startswith(self.prefix):
                    # 移除 前缀
                    new_key = key.replace(self.prefix, '', 1)
                    backbone_ckpt[new_key] = value

            res = self.load_state_dict(backbone_ckpt, strict=False)
            print(res)
        else:
            print('kaiming_normal init in backbone')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def loss(self, backbone_feas, batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        seg_logits = self.forward(backbone_feas)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def forward(self,inputs: Union[Tensor,Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        out = self._transform_inputs(inputs)
        return self.ohemhead(out, self.cls_seg)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        out = seg_logits
        sem_label = self._stack_batch_gt(batch_data_samples)

        out = resize(
            input=out,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        sem_label = sem_label.squeeze(1)  # 分割标签
        loss['loss_ohem'] = self.loss_decode[0](out, sem_label)  # ohem 分割损失
        loss['acc_seg'] = accuracy(
            out, sem_label, ignore_index=self.ignore_index)
        return loss
