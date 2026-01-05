import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList
from mmdet.registry import MODELS
from complexNN.nn import cConv2d, cBatchNorm2d, cRelu, cMaxPool2d, ComplexAdaptiveAvgPool2d, cLinear


# 替代 constant_init
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# 替代 kaiming_init
def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'normal':
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ComplexBasicBlock(BaseModule):
    """符合MMDetection接口的复数ResNet基本残差块"""
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(ComplexBasicBlock, self).__init__(init_cfg)
        self.conv1 = cConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, bias=False)
        self.bn1 = cBatchNorm2d(out_channels)
        self.relu = cRelu()
        self.conv2 = cConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = cBatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ComplexBottleneck(BaseModule):
    """符合MMDetection接口的复数ResNet瓶颈残差块"""
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(ComplexBottleneck, self).__init__(init_cfg)
        self.conv1 = cConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = cBatchNorm2d(out_channels)
        self.conv2 = cConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, bias=False)
        self.bn2 = cBatchNorm2d(out_channels)
        self.conv3 = cConv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = cBatchNorm2d(out_channels * self.expansion)
        self.relu = cRelu()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@MODELS.register_module()
class ComplexResNet(BaseModule):
    """符合MMDetection接口的复数ResNet模型"""
    arch_settings = {
        18: (ComplexBasicBlock, (2, 2, 2, 2)),
        34: (ComplexBasicBlock, (3, 4, 6, 3)),
        50: (ComplexBottleneck, (3, 4, 6, 3)),
        101: (ComplexBottleneck, (3, 4, 23, 3)),
        152: (ComplexBottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=1,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super(ComplexResNet, self).__init__(init_cfg)
        self.depth = depth
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.plugins = plugins
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = base_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * (2 ** i)
            res_layer = self._make_layer(
                block=self.block,
                in_channels=self.inplanes,
                out_channels=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * (2 ** (len(self.stage_blocks) - 1))

    def _make_stem_layer(self, in_channels, stem_channels):
        if stem_channels is None:
            stem_channels = self.inplanes
        self.conv1 = cConv2d(
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn1 = cBatchNorm2d(stem_channels)
        self.relu = cRelu()
        self.maxpool = cMaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_layer(self,
                    block,
                    in_channels,
                    out_channels,
                    num_blocks,
                    stride=1,
                    dilation=1,
                    style='pytorch',
                    with_cp=False,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN'),
                    dcn=None,
                    plugins=None):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                cConv2d(
                    in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                cBatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=plugins))
        in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    dilation=dilation,
                    style=style,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn,
                    plugins=plugins))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if i <= len(self.res_layers):
                m = getattr(self, f'layer{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        super(ComplexResNet, self).init_weights()
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, (cConv2d, cLinear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, cBatchNorm2d):
                    # 初始化实部BN层的参数
                    if m.real_bn.affine:
                        nn.init.constant_(m.real_bn.weight, 1)
                        nn.init.constant_(m.real_bn.bias, 0)
                    # 初始化虚部BN层的参数
                    if m.imag_bn.affine:
                        nn.init.constant_(m.imag_bn.weight, 1)
                        nn.init.constant_(m.imag_bn.bias, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, ComplexBottleneck) and m.bn3.real_bn.weight is not None:
                        nn.init.constant_(m.bn3.real_bn.weight, 0)
                        nn.init.constant_(m.bn3.imag_bn.weight, 0)
                    elif isinstance(m, ComplexBasicBlock) and m.bn2.real_bn.weight is not None:
                        nn.init.constant_(m.bn2.real_bn.weight, 0)
                        nn.init.constant_(m.bn2.imag_bn.weight, 0)

    def forward(self, x):
        # 转换实值输入为复数形式
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            # 计算复数特征的模长作为输出
            magnitude = torch.abs(x)

            if i in self.out_indices:
                outs.append(magnitude)

        return tuple(outs)

    def train(self, mode=True):
        super(ComplexResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, cBatchNorm2d):
                    m.eval()
