from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import os.path as osp

@DATASETS.register_module()
class CamVidDataset(BaseSegDataset):
    CLASSES = ['Sky',
               'Building',
               'Column-Pole',
               'Road',
               'Sidewalk',
               'Tree',
               'Sign-Symbol',
               'Fence',
               'Car',
               'Pedestrain',
               'Bicyclist',
               #'void'
               ]

    PALETTE = [[128, 128, 128],
               [128, 0, 0],
               [192, 192, 128],
               [128, 64, 128],
               [0, 0, 192],
               [128, 128, 0],
               [192, 128, 128],
               [64, 64, 128],
               [64, 0, 128],
               [64, 64, 0],
               [0, 128, 192],
               #[0, 0, 0]
               ]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            metainfo=dict(classes=self.CLASSES, palette=self.PALETTE),
            **kwargs
        )

