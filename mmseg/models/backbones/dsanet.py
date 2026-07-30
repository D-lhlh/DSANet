# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/MichaelFan01/STDC-Seg."""
import torch
from torch import Tensor
from torch import vmap
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Type, Dict
from mmcv.cnn import ConvModule
# from pytorch_wavelets import DWTForward
from mmengine.model import BaseModule, ModuleList, Sequential
# from mmseg.models.utils.shuffleblock import ChannelsSample, SelfThreeDimentionSample
from mmengine.runner import CheckpointLoader
from mmseg.models.utils.ppm import DAPPM, PAPPM
from mmseg.models.utils.se_layer import SELayer
from mmseg.registry import MODELS


# from mmseg.models.utils.HyperACE import C3AH
# from .bisenetv1 import AttentionRefinementModule
from tools.atten_vis import FeatureFusionVisualizer,FlowFeatureVisualizer

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.conv3x3 = ConvModule(in_channels, in_channels, 3, 2, 1, norm_cfg=None, act_cfg=None, bias=False)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv3x3out = ConvModule(out_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg,
                                     bias=False)

    def forward(self, input):
        output = self.conv3x3(input)
        avg_pool = self.avg_pool(input)
        output = torch.cat([output, avg_pool], 1)
        output = self.conv3x3out(output)
        return output


class LargekernelAggregateConvBlock(nn.Module):
    def __init__(self, channels, d=1, kSize=3, dkSize=3, partition=2,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 ):
        super().__init__()
        self.channels = channels
        self.part = partition
        self.aggregate_size = d
        self.mid_channels = channels // self.part
        self.aggregate_conv = nn.ModuleList([])
        for i in range(self.part):
            self.aggregate_conv.append(
                ConvModule(
                    self.mid_channels,
                    self.mid_channels,
                    kernel_size=self.aggregate_size,
                    stride=1,
                    padding=self.aggregate_size // 2,
                    groups=self.mid_channels,
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                    bias=False,
                )
            )
        self.conv3x3 = nn.ModuleList([])
        for i in range(self.part):
            self.conv3x3.append(
                ConvModule(
                    self.mid_channels,
                    self.mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=False
                )
            )

        self.pwconv = ConvModule(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
            bias=False
        )
        if act_cfg.get('type') == "GELU":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

    def forward(self, input):
        # vis = FeatureFusionVisualizer()
        part_list = torch.split(input, self.channels // self.part, 1)
        out_list = []
        for idx in range(self.part):
            part_x = self.aggregate_conv[idx](part_list[idx])
            part_x = self.conv3x3[idx](part_x)
            out_list.append(part_x)
        output = self.pwconv(torch.cat(out_list, 1)) + input
        # vis.visualize_feature_map(output, 'output')
        output = self.act(output)
        # vis.visualize_feature_map(output, 'act(output)')
        return output


class LargekernelAggregateConvBlockV2(nn.Module):
    def __init__(self, channels, agg_size=1, kSize=3, dkSize=3, partition=2,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 ):
        super().__init__()
        self.channels = channels
        self.part = partition
        self.aggregate_size = agg_size
        self.mid_channels = channels // self.part

        self.aggregate_conv = ConvModule(
            channels,
            channels,
            kernel_size=self.aggregate_size,
            stride=1,
            padding=self.aggregate_size // 2,
            groups=channels,
            norm_cfg=norm_cfg,
            act_cfg=None,
            bias=False,
        )

        self.conv3x3 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.part,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False
        )

        self.pwconv = ConvModule(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
            bias=False
        )
        if act_cfg.get('type') == "GELU":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

    def forward(self, input):
        # vis = FeatureFusionVisualizer()
        agg_x = self.aggregate_conv(input)
        agg_x = self.conv3x3(agg_x)

        output = self.pwconv(agg_x) + input
        # vis.visualize_feature_map(output, 'output')
        output = self.act(output)
        # vis.visualize_feature_map(output, 'act(output)')
        return output

class DilationConv(nn.Module):
    def __init__(self, channels, agg_size=1, kSize=3, dkSize=3, partition=2,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 ):
        super().__init__()
        self.channels = channels
        self.part = partition
        self.dilation = agg_size + 2
        self.mid_channels = channels // self.part

        self.Dconv3x3 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            dilation=self.dilation,
            padding=self.dilation,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False
        )

    def forward(self, input):
        # vis = FeatureFusionVisualizer()
        output = self.Dconv3x3(input)
        return output

class SimpleCatDecoder(nn.Module):
    def __init__(self, channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 ):
        super().__init__()

        self.outconv = ConvModule(
            sum(channels),
            channels[0],
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
        )

    def forward(self, inputs):
        # vis = FeatureFusionVisualizer()
        x1, x2, x3 = inputs
        h, w = x1.shape[2:]
        # x_s = self.upconv(x_s)
        x2_up = F.interpolate(x2, size=(h, w), mode='bilinear')
        x3_up = F.interpolate(x3, size=(h, w), mode='bilinear')
        # output = self.HyperACE(torch.cat([x_s, x], 1))
        output = self.outconv(torch.cat([x1, x2_up, x3_up], 1))
        # vis.visualize_feature_map(output, 'output')
        return output



class DAD(nn.Module):
    def __init__(self, c2, c1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 ):
        super().__init__()
        #self.conv1x1_c = Conv(c2, c1, 1, 1, padding=0, bn_acti=True)
        self.conv1x1_c = ConvModule(
            c2, c1, 1, 1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        #self.conv1x1_neg = Conv(c1, c1, 1, 1, padding=0, bn_acti=True)
        self.conv1x1_neg = ConvModule(
            c1, c1, 1, 1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        #self.conv3x3 = Conv(c1, c1, (3, 3), 1, padding=(1, 1), groups=c1, bn_acti=True)
        self.conv3x3 = ConvModule(
            c1, c1, 3, 1,
            padding=1,
            groups=c1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, X, Y):
        X_map = torch.sigmoid(X)
        F_sg = X_map

        Yc = self.conv1x1_c(Y)
        Yc_map = torch.sigmoid(Yc)
        Neg_map = self.conv1x1_neg(-Yc_map)
        F_rg = Neg_map * Yc_map + Yc
        F_rg = F.interpolate(F_rg, F_sg.size()[2:], mode='bilinear', align_corners=False)

        output = F_sg * F_rg
        output = self.conv3x3(output)
        return output

class UMF(nn.Module):
    def __init__(self, c_list,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 ):
        super().__init__()
        self.up_conv1 = ConvModule(
            c_list[2], c_list[0], 3, 1, 1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
        )
        self.up_conv2 = ConvModule(
            c_list[1], c_list[0], 3, 1, 1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
        )
        self.conv_atten = nn.Sequential(
            ConvModule(
                6, 6, 3, 1, 1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False,
            ),
            ConvModule(
                6, 3, 3, 1, 1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=False,
            )
        )

    def forward(self, x, y, z):
        size = x.shape[2:]
        y_up = F.interpolate(self.up_conv2(y), size=size, mode='bilinear')
        z_up = F.interpolate(self.up_conv1(z), size=size, mode='bilinear')

        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        y_mean = torch.mean(y_up, dim=1, keepdim=True)
        y_max = torch.max(y_up, dim=1, keepdim=True)[0]
        z_mean = torch.mean(z_up, dim=1, keepdim=True)
        z_max = torch.max(z_up, dim=1, keepdim=True)[0]
        cat_mm = torch.cat([x_mean,x_max,y_mean,y_max,z_mean,z_max],1)

        w = F.sigmoid(self.conv_atten(cat_mm))
        w1, w2, w3 = torch.split(w, 1, 1)
        output = w1 * x + w2 * y_up + w3 * z_up
        return output

class AlignedModulev2PoolingAtten(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModulev2PoolingAtten, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        h_feature_mean = torch.mean(h_feature, dim=1).unsqueeze(1)
        l_feature_mean = torch.mean(low_feature, dim=1).unsqueeze(1)
        h_feature_max = torch.max(h_feature, dim=1)[0].unsqueeze(1)
        l_feature_max = torch.max(low_feature, dim=1)[0].unsqueeze(1)

        flow_gates = self.flow_gate(torch.cat([h_feature_mean, l_feature_mean, h_feature_max, l_feature_max], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        #grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid.type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class DeatilAlignedModule(nn.Module):
    '''
        GD-SFAM
    '''

    def __init__(self, inplane, outplane, kernel_size=3, predictmethod='normal'):
        super(DeatilAlignedModule, self).__init__()
        self.part_flow = outplane * 2
        self.conv_h = nn.Conv2d(inplane, outplane, 3, 1, 1, groups=inplane, bias=False)
        self.conv_l = nn.Conv2d(inplane, outplane, 3, 1, 1, groups=inplane, bias=False)
        if predictmethod == 'normal':
            self.flow_make_detail = nn.Conv2d(outplane, 2 * outplane * 2, kernel_size=3, padding=1, stride=1,
                                              bias=False)
        if predictmethod == 'groups':
            self.flow_make_detail = nn.Conv2d(outplane, 2 * outplane * 2, kernel_size=3, padding=1, stride=1,
                                              groups=outplane, bias=False)
        if predictmethod == 'extend':
            self.flow_make_detail = nn.Sequential(
                nn.Conv2d(outplane, 2 * outplane * 2, kernel_size=1, bias=False),
                nn.Conv2d(2 * outplane * 2, 2 * outplane * 2, kernel_size=3, padding=1, stride=1, groups=outplane,
                          bias=False)
            )

        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # vis = FeatureFusionVisualizer()
        low_feature, high_feature = x
        # vis.visualize_feature_map(F.interpolate(high_feature, scale_factor=2, mode='bilinear'), 'high_feature_origin_up')
        lb, ln, lh, lw = low_feature.shape
        hb, hn, hh, hw = high_feature.shape
        l_feature = self.conv_l(low_feature)
        h_feature = self.conv_h(high_feature)
        h_feature = F.interpolate(h_feature, size=low_feature.shape[2:], mode='bilinear')
        flow = self.flow_make_detail(l_feature + h_feature)
        flow_h, flow_l = torch.split(flow, self.part_flow, 1)

        high_feature_r = high_feature.reshape(hb * hn, 1, hh, hw)
        flow_h_r = flow_h.reshape(lb * ln, 2, lh, lw)
        high_feature_warp = self.flow_warp(high_feature_r, flow_h_r, (lh, lw))
        high_feature_warp = high_feature_warp.reshape(lb, ln, lh, lw)
        # vis.visualize_feature_map(high_feature_warp, 'high_feature_warp')

        low_feature_r = low_feature.reshape(lb * ln, 1, lh, lw)
        flow_l_r = flow_l.reshape(lb * ln, 2, lh, lw)
        low_feature_warp = self.flow_warp(low_feature_r, flow_l_r, (lh, lw))
        low_feature_warp = low_feature_warp.reshape(lb, ln, lh, lw)

        h_feature_mean = torch.mean(h_feature, dim=1).unsqueeze(1)
        l_feature_mean = torch.mean(l_feature, dim=1).unsqueeze(1)
        h_feature_max = torch.max(h_feature, dim=1)[0].unsqueeze(1)
        l_feature_max = torch.max(l_feature, dim=1)[0].unsqueeze(1)
        flow_gates = self.flow_gate(torch.cat([h_feature_mean, l_feature_mean, h_feature_max, l_feature_max], 1))

        output = high_feature_warp * flow_gates + low_feature_warp * (1 - flow_gates)
        # vis.visualize_feature_map(output, 'output')
        return output

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w
        #flowvis = FlowFeatureVisualizer()
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        #grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid.type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        #flowvis.visualize_features(F.interpolate(input, size=size, mode='bilinear'), output, flow, channel_idx=0)
        return output


class SematicAlignedModule(nn.Module):
    '''
    from SFSEGNet
    '''

    def __init__(self, inplane, outplane, kernel_size=3):
        super(SematicAlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # vis = FeatureFusionVisualizer()
        low_feature, h_feature = x
        # vis.visualize_feature_map(low_feature, 'low_feature_origin')
        # vis.visualize_feature_map(h_feature, 'h_feature_origin')
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        # vis.visualize_feature_map(l_feature, 'l_feature')
        # vis.visualize_feature_map(h_feature, 'h_feature')

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_h, flow_l = torch.split(flow, 2, 1)
        # vis.visualize_feature_map(flow, 'flow')
        h_feature_warp = self.flow_warp(h_feature_orign, flow_h, size=size)
        low_feature_warp = self.flow_warp(low_feature, flow_l, size=size)
        # vis.visualize_feature_map(h_feature_warp, 'h_feature_warp')

        h_feature_mean = torch.mean(h_feature, dim=1).unsqueeze(1)
        l_feature_mean = torch.mean(low_feature, dim=1).unsqueeze(1)
        h_feature_max = torch.max(h_feature, dim=1)[0].unsqueeze(1)
        l_feature_max = torch.max(low_feature, dim=1)[0].unsqueeze(1)

        flow_gates = self.flow_gate(torch.cat([h_feature_mean, l_feature_mean, h_feature_max, l_feature_max], 1))
        # vis.visualize_feature_map(flow_gates, 'flow_gates')

        fuse_feature = h_feature_warp * flow_gates + low_feature_warp * (1 - flow_gates)
        # vis.visualize_feature_map(fuse_feature, 'fuse_feature')

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        #grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid.type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class DetailSematicFlowAligendModule(nn.Module):
    def __init__(self, channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 ):
        super().__init__()
        self.upconv_1 = ConvModule(
            channels[1],
            channels[0],
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
        )
        self.upconv_2 = ConvModule(
            channels[2],
            channels[0],
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
        )
        self.DAM = DeatilAlignedModule(inplane=channels[0], outplane=channels[0], predictmethod='extend')
        self.SAM = SematicAlignedModule(inplane=channels[0], outplane=channels[0])
        self.out_conv = ConvModule(
            channels[0] * 3,
            channels[0],
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
        )

    def forward(self, inputs):
        x1, x2, x3 = inputs  # 1/4  1/8  1/16
        x2 = self.upconv_1(x2)
        x3 = self.upconv_2(x3)
        detail_aligned = self.DAM([x1, x2])
        sematic_aligned = self.SAM([detail_aligned, x3])
        x3_up = F.interpolate(x3, size=sematic_aligned.shape[2:], mode='bilinear')
        output = torch.cat([x1, sematic_aligned, x3_up], 1)
        output = self.out_conv(output)
        return output

@MODELS.register_module()
class DSANet(BaseModule):
    def __init__(self,
                 in_channels,
                 channels,
                 norm_cfg,
                 act_cfg,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.channels = channels
        self.downtimes = len(self.channels) - 1

        self.dilation1 = [3, 5, 7]
        self.dilation2 = [3, 5, 7, 9, 11]
        self.dilation3 = [3, 5, 7, 9, 11, 13, 15]

        # best
        # self.dilation1 = [3, 7, 11]
        # self.dilation2 = [3, 7, 11, 15, 19]
        # self.dilation3 = [3, 7, 11, 15, 19, 23, 27]

        # self.dilation1 = [3, 9, 15]
        # self.dilation2 = [3, 9, 15, 21, 27]
        # self.dilation3 = [3, 9, 15, 21, 27, 33, 39]

        self.layers_num_1 = len(self.dilation1)
        self.layers_num_2 = len(self.dilation2)
        self.layers_num_3 = len(self.dilation3)

        self.cnn_stem = nn.Sequential(
            ConvModule(
                self.in_channels,
                self.channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False
            ),
            ConvModule(
                self.channels[0],
                self.channels[0],
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False),
            ConvModule(
                self.channels[0],
                self.channels[0],
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                bias=False)
        )

        self.downsample_layer = nn.Sequential()
        for i in range(self.downtimes):
            self.downsample_layer.append(
                DownSampleBlock(self.channels[i], self.channels[1 + i],
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
            )

        self.LACB_layers1 = nn.Sequential()
        for idx in range(self.layers_num_1):
            self.LACB_layers1.append(
                LargekernelAggregateConvBlockV2(channels=self.channels[1], agg_size=self.dilation1[idx], partition=2,
                                                norm_cfg=norm_cfg,
                                                act_cfg=act_cfg)
            )
        self.LACB_layers2 = nn.Sequential()
        for idx in range(self.layers_num_2):
            self.LACB_layers2.append(
                LargekernelAggregateConvBlockV2(channels=self.channels[2], agg_size=self.dilation2[idx], partition=2,
                                                norm_cfg=norm_cfg,
                                                act_cfg=act_cfg)
            )
        self.LACB_layers3 = nn.Sequential()
        for idx in range(self.layers_num_3):
            self.LACB_layers3.append(
                LargekernelAggregateConvBlockV2(channels=self.channels[3], agg_size=self.dilation3[idx], partition=2,
                                                norm_cfg=norm_cfg,
                                                act_cfg=act_cfg)
            )

        self.DSFA = DetailSematicFlowAligendModule(channels=self.channels[1:], norm_cfg=norm_cfg, act_cfg=act_cfg)

        #self.scd = SimpleCatDecoder(channels=self.channels[1:], norm_cfg=norm_cfg, act_cfg=act_cfg)
        #self.LCNet_DAD = DAD(self.channels[-1], self.channels[1], norm_cfg=norm_cfg, act_cfg=act_cfg)
        #self.BSSNet_UMF = UMF(self.channels[1:], norm_cfg=norm_cfg, act_cfg=act_cfg)


        # self.fpn = ConvModule(
        #     self.channels[-1],
        #     self.channels[1],
        #     kernel_size=1,
        #     norm_cfg=norm_cfg,
        #     act_cfg=act_cfg,
        #     bias=False
        # )
        # self.GD_FAM = AlignedModulev2PoolingAtten(self.channels[1],self.channels[1])


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

    def forward(self, x):
        x_stem = self.cnn_stem(x)  # 32

        ''' 16 -> 32 '''
        x_ds = self.downsample_layer[0](x_stem)
        x_s1 = self.LACB_layers1(x_ds)

        ''' 32 -> 64 '''
        x_ds = self.downsample_layer[1](x_s1)
        x_s2 = self.LACB_layers2(x_ds)

        ''' 64 -> 128 '''
        x_ds = self.downsample_layer[2](x_s2)
        x_s3 = self.LACB_layers3(x_ds)

        #output = self.scd([x_s1, x_s2, x_s3])

        #output = self.LCNet_DAD(x_s1, x_s3)
        #output = self.BSSNet_UMF(x_s1, x_s2, x_s3)

        # x_s3_ = self.fpn(x_s3)
        # output = self.GD_FAM([x_s1, x_s3_])

        output = self.DSFA([x_s1, x_s2, x_s3])
        return (x_s2, output)
