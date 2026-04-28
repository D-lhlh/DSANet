import torch
from mmengine import Config
from mmseg.apis import init_model
from mmengine.analysis import get_model_complexity_info


#cfg = Config.fromfile('../../configs/bisenetV2/bisenetv2_fcn_4xb4-ohem-160k_cityscapes-1024x1024.py')
#cfg = Config.fromfile('../../configs/ddrnet/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.py')
#cfg = Config.fromfile('../../configs/bssnet/bssnet-t-b12-120k-1024x1024-cityscapes.py')
#cfg = Config.fromfile('../../configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py')
#cfg = Config.fromfile('../../configs/icnet/icnet_r18-d8-in1k-pre_4xb2-160k_cityscapes-832x832.py')
cfg = Config.fromfile('../../configs/sctnet/sctnet-s_seg50_8x2_160k_cityscapes.py')
#cfg = Config.fromfile('../../configs/dsanet/dsanet_bdd100k_720x1280.py')
model = init_model(cfg, device='cuda')

'''cityscapes'''
input_shape = (3, 512, 1024)
'''camvid'''
#input_shape = (3, 720, 960)
'''bdd100k'''
#input_shape = (3, 720, 1280)
output = get_model_complexity_info(model, input_shape, show_arch=False)

if isinstance(output, dict):
    for key, value in output.items():
        print(f"{key}: {value}")
