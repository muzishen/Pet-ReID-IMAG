from math import ceil
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc as container_abcs
from torch.utils import model_zoo
from .build import BACKBONE_REGISTRY
logger = logging.getLogger(__name__)

def _pair(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return (x, x)


class SamePaddingConv2d(nn.Module):
    def __init__(self,
                 in_spatial_shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dilation=1,
                 enforce_in_spatial_shape=False,
                 **kwargs):
        super(SamePaddingConv2d, self).__init__()

        self._in_spatial_shape = _pair(in_spatial_shape)
        # e.g. throw exception if input spatial shape does not match in_spatial_shape
        # when calling self.forward()
        self.enforce_in_spatial_shape = enforce_in_spatial_shape
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)

        in_height, in_width = self._in_spatial_shape
        filter_height, filter_width = kernel_size
        stride_heigth, stride_width = stride
        dilation_height, dilation_width = dilation

        out_height = int(ceil(float(in_height) / float(stride_heigth)))
        out_width = int(ceil(float(in_width) / float(stride_width)))

        pad_along_height = max((out_height - 1) * stride_heigth +
                               filter_height + (filter_height - 1) * (dilation_height - 1) - in_height, 0)
        pad_along_width = max((out_width - 1) * stride_width +
                              filter_width + (filter_width - 1) * (dilation_width - 1) - in_width, 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        paddings = (pad_left, pad_right, pad_top, pad_bottom)
        if any(p > 0 for p in paddings):
            self.zero_pad = nn.ZeroPad2d(paddings)
        else:
            self.zero_pad = None
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              dilation=dilation,
                              **kwargs)

        self._out_spatial_shape = (out_height, out_width)

    @property
    def in_spatial_shape(self):
        return self._in_spatial_shape

    @property
    def out_spatial_shape(self):
        return self._out_spatial_shape

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def check_spatial_shape(self, x):
        if x.size(2) != self.in_spatial_shape[0] or \
                x.size(3) != self.in_spatial_shape[1]:
            raise ValueError(
                "Expected input spatial shape {}, got {} instead".format(self.in_spatial_shape,
                                                                         x.shape[2:]))

    def forward(self, x):
        if self.enforce_in_spatial_shape:
            self.check_spatial_shape(x)
        if self.zero_pad is not None:
            x = self.zero_pad(x)
        x = self.conv(x)
        return x


class ConvBNAct(nn.Module):
    def __init__(self,
                 out_channels,
                 activation=None,
                 bn_epsilon=None,
                 bn_momentum=None,
                 same_padding=False,
                 **kwargs):
        super(ConvBNAct, self).__init__()

        _conv_cls = SamePaddingConv2d if same_padding else nn.Conv2d
        self.conv = _conv_cls(out_channels=out_channels, **kwargs)

        bn_kwargs = {}
        if bn_epsilon is not None:
            bn_kwargs["eps"] = bn_epsilon
        if bn_momentum is not None:
            bn_kwargs["momentum"] = bn_momentum

        self.bn = nn.BatchNorm2d(out_channels, **bn_kwargs)
        self.activation = activation

    @property
    def in_spatial_shape(self):
        if isinstance(self.conv, SamePaddingConv2d):
            return self.conv.in_spatial_shape
        else:
            return None

    @property
    def out_spatial_shape(self):
        if isinstance(self.conv, SamePaddingConv2d):
            return self.conv.out_spatial_shape
        else:
            return None

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Swish(nn.Module):
    def __init__(self,
                 beta=1.0,
                 beta_learnable=False):
        super(Swish, self).__init__()

        if beta == 1.0 and not beta_learnable:
            self._op = self.simple_swish
        else:
            self.beta = nn.Parameter(torch.full([1], beta),
                                     requires_grad=beta_learnable)
            self._op = self.advanced_swish

    def simple_swish(self, x):
        return x * torch.sigmoid(x)

    def advanced_swish(self, x):
        return x * torch.sigmoid(self.beta * x)

    def forward(self, x):
        return self._op(x)


class DropConnect(nn.Module):
    def __init__(self, rate=0.5):
        super(DropConnect, self).__init__()
        self.keep_prob = None
        self.set_rate(rate)

    def set_rate(self, rate):
        if not 0 <= rate < 1:
            raise ValueError("rate must be 0<=rate<1, got {} instead".format(rate))
        self.keep_prob = 1 - rate

    def forward(self, x):
        if self.training:
            random_tensor = self.keep_prob + torch.rand([x.size(0), 1, 1, 1],
                                                        dtype=x.dtype,
                                                        device=x.device)
            binary_tensor = torch.floor(random_tensor)
            return torch.mul(torch.div(x, self.keep_prob), binary_tensor)
        else:
            return x


class SqueezeExcitate(nn.Module):
    def __init__(self,
                 in_channels,
                 se_size,
                 activation=None):
        super(SqueezeExcitate, self).__init__()
        self.dim_reduce = nn.Conv2d(in_channels=in_channels,
                                    out_channels=se_size,
                                    kernel_size=1)
        self.dim_restore = nn.Conv2d(in_channels=se_size,
                                     out_channels=in_channels,
                                     kernel_size=1)
        self.activation = F.relu if activation is None else activation

    def forward(self, x):
        inp = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.dim_reduce(x)
        x = self.activation(x)
        x = self.dim_restore(x)
        x = torch.sigmoid(x)
        return torch.mul(inp, x)


class MBConvBlock(nn.Module):
    def __init__(self,
                 in_spatial_shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expansion_factor,
                 activation,
                 bn_epsilon=None,
                 bn_momentum=None,
                 se_size=None,
                 drop_connect_rate=None,
                 bias=False):
        """
        Initialize new MBConv block
        :param in_spatial_shape: image shape, e.g. tuple [height, width] or int size for [size, size]
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size for depth-wise convolution
        :param stride: stride for depth-wise convolution
        :param expansion_factor: expansion factor
        :param bn_epsilon: batch normalization epsilon
        :param bn_momentum: batch normalization momentum
        :param se_size: number of features in reduction layer of Squeeze-and-Excitate layer
        :param activation: activation function
        :param drop_connect_rate: DropConnect rate
        :param bias: enable bias in convolution operations
        """
        super(MBConvBlock, self).__init__()

        if se_size is not None and se_size < 1:
            raise ValueError("se_size must be >=1, got {} instead".format(se_size))

        if drop_connect_rate is not None and not 0 <= drop_connect_rate < 1:
            raise ValueError("drop_connect_rate must be in range [0,1), got {} instead".format(drop_connect_rate))

        if not (isinstance(expansion_factor, int) and expansion_factor >= 1):
            raise ValueError("expansion factor must be int and >=1, got {} instead".format(expansion_factor))

        exp_channels = in_channels * expansion_factor
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        self.activation = activation

        # expansion convolution
        if expansion_factor != 1:
            self.expand_conv = ConvBNAct(in_channels=in_channels,
                                         out_channels=exp_channels,
                                         kernel_size=(1, 1),
                                         bias=bias,
                                         activation=self.activation,
                                         bn_epsilon=bn_epsilon,
                                         bn_momentum=bn_momentum)
        else:
            self.expand_conv = None

        # depth-wise convolution
        self.dp_conv = ConvBNAct(in_spatial_shape=in_spatial_shape,
                                 in_channels=exp_channels,
                                 out_channels=exp_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 groups=exp_channels,
                                 bias=bias,
                                 activation=self.activation,
                                 same_padding=True,
                                 bn_epsilon=bn_epsilon,
                                 bn_momentum=bn_momentum)

        if se_size is not None:
            self.se = SqueezeExcitate(exp_channels,
                                      se_size,
                                      activation=self.activation)
        else:
            self.se = None

        if drop_connect_rate is not None:
            self.drop_connect = DropConnect(drop_connect_rate)
        else:
            self.drop_connect = None

        if in_channels == out_channels and all(s == 1 for s in stride):
            self.skip_enabled = True
        else:
            self.skip_enabled = False

        # projection convolution
        self.project_conv = ConvBNAct(in_channels=exp_channels,
                                      out_channels=out_channels,
                                      kernel_size=(1, 1),
                                      bias=bias,
                                      activation=None,
                                      bn_epsilon=bn_epsilon,
                                      bn_momentum=bn_momentum)

    @property
    def in_spatial_shape(self):
        return self.dp_conv.in_spatial_shape

    @property
    def out_spatial_shape(self):
        return self.dp_conv.out_spatial_shape

    @property
    def in_channels(self):
        if self.expand_conv is not None:
            return self.expand_conv.in_channels
        else:
            return self.dp_conv.in_channels

    @property
    def out_channels(self):
        return self.project_conv.out_channels

    def forward(self, x):
        inp = x

        if self.expand_conv is not None:
            # expansion convolution applied only if expansion ratio > 1
            x = self.expand_conv(x)

        # depth-wise convolution
        x = self.dp_conv(x)

        # squeeze-and-excitate
        if self.se is not None:
            x = self.se(x)

        # projection convolution
        x = self.project_conv(x)

        if self.skip_enabled:
            # drop-connect applied only if skip connection enabled
            if self.drop_connect is not None:
                x = self.drop_connect(x)
            x = x + inp
        return x


class EnetStage(nn.Module):
    def __init__(self,
                 num_layers,
                 in_spatial_shape,
                 in_channels,
                 out_channels,
                 stride,
                 se_ratio,
                 drop_connect_rates,
                 **kwargs):
        super(EnetStage, self).__init__()

        if not (isinstance(num_layers, int) and num_layers >= 1):
            raise ValueError("num_layers must be int and >=1, got {} instead".format(num_layers))

        if not (isinstance(drop_connect_rates, container_abcs.Iterable) and
                len(drop_connect_rates) == num_layers):
            raise ValueError("drop_connect_rates must be iterable of "
                             "length num_layers ({}), got {} instead".format(num_layers, drop_connect_rates))

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        spatial_shape = in_spatial_shape
        for i in range(self.num_layers):
            se_size = max(1, in_channels // se_ratio)
            layer = MBConvBlock(in_spatial_shape=spatial_shape,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                se_size=se_size,
                                drop_connect_rate=drop_connect_rates[i],
                                **kwargs)
            self.layers.append(layer)
            spatial_shape = layer.out_spatial_shape
            # remaining MBConv blocks have stride 1 and in_channels=out_channels
            stride = 1
            in_channels = out_channels

    @property
    def in_spatial_shape(self):
        return self.layers[0].in_spatial_shape

    @property
    def out_spatial_shape(self):
        return self.layers[-1].out_spatial_shape

    @property
    def in_channels(self):
        return self.layers[0].in_channels

    @property
    def out_channels(self):
        return self.layers[-1].out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def round_filters(filters, width_coefficient, depth_divisor=8):
    """Round number of filters based on depth multiplier."""
    min_depth = depth_divisor

    filters *= width_coefficient
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of filters based on depth multiplier."""
    return int(ceil(depth_coefficient * repeats))


class EfficientNet(nn.Module):
    # (width_coefficient, depth_coefficient, dropout_rate, in_spatial_shape)
    coefficients = [
        (1.0, 1.0, 0.2, 224),
        (1.0, 1.1, 0.2, 240),
        (1.1, 1.2, 0.3, 260),
        (1.2, 1.4, 0.3, 300),
        (1.4, 1.8, 0.4, 380),
        (1.6, 2.2, 0.4, 456),
        (1.8, 2.6, 0.5, 528),
        (2.0, 3.1, 0.5, 600),
    ]

    # block_repeat, kernel_size, stride, expansion_factor, input_channels, output_channels, se_ratio
    stage_args = [
        [1, 3, 1, 1, 32, 16, 4],
        [2, 3, 2, 6, 16, 24, 4],
        [2, 5, 2, 6, 24, 40, 4],
        [3, 3, 2, 6, 40, 80, 4],
        [3, 5, 1, 6, 80, 112, 4],
        [4, 5, 2, 6, 112, 192, 4],
        [1, 3, 1, 6, 192, 320, 4],
    ]

    state_dict_urls = [
        "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmliYV9HaE5PWWVEbXVMd3c/root/content",
        "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmlicV9HaE5PWWVEbXVMd3c/root/content",
        "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmliNl9HaE5PWWVEbXVMd3c/root/content",
        "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmljS19HaE5PWWVEbXVMd3c/root/content",
        "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmljYV9HaE5PWWVEbXVMd3c/root/content",
        "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmljcV9HaE5PWWVEbXVMd3c/root/content",
        "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmljNl9HaE5PWWVEbXVMd3c/root/content",
        "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmlkS19HaE5PWWVEbXVMd3c/root/content",
    ]

    dict_names = [
        'efficientnet-b0-d86f8792.pth',
        'efficientnet-b1-82896633.pth',
        'efficientnet-b2-e4b93854.pth',
        'efficientnet-b3-3b9ca610.pth',
        'efficientnet-b4-24436ca5.pth',
        'efficientnet-b5-d8e577e8.pth',
        'efficientnet-b6-f20845c7.pth',
        'efficientnet-b7-86e8e374.pth'
    ]

    def __init__(self,
                 b,
                 in_channels=3,
                 n_classes=1000,
                 in_spatial_shape=None,
                 activation=Swish(),
                 bias=False,
                 drop_connect_rate=0.2,
                 dropout_rate=None,
                 bn_epsilon=1e-3,
                 bn_momentum=0.01,
                 pretrained=False,
                 progress=False):
        """
        Initialize new EfficientNet model
        :param b: model index, i.e. 0 for EfficientNet-B0
        :param in_channels: number of input channels
        :param n_classes: number of output classes
        :param in_spatial_shape: input image shape
        :param activation: activation function
        :param bias: enable bias in convolution operations
        :param drop_connect_rate: DropConnect rate
        :param dropout_rate: dropout rate, this will override default rate for each model
        :param bn_epsilon: batch normalization epsilon
        :param bn_momentum: batch normalization momentum
        :param pretrained: initialize model with weights pre-trained on ImageNet
        :param progress: show progress when downloading pre-trained weights
        """

        super(EfficientNet, self).__init__()

        # verify all parameters
        EfficientNet.check_init_params(b,
                                       in_channels,
                                       n_classes,
                                       in_spatial_shape,
                                       activation,
                                       bias,
                                       drop_connect_rate,
                                       dropout_rate,
                                       bn_epsilon,
                                       bn_momentum,
                                       pretrained,
                                       progress)

        self.b = b
        self.in_channels = in_channels
        self.activation = activation
        self.drop_connect_rate = drop_connect_rate
        self._override_dropout_rate = dropout_rate

        width_coefficient, _, _, spatial_shape = EfficientNet.coefficients[self.b]

        if in_spatial_shape is not None:
            self.in_spatial_shape = _pair(in_spatial_shape)
        else:
            self.in_spatial_shape = _pair(spatial_shape)

        # initial convolution
        init_conv_out_channels = round_filters(32, width_coefficient)
        self.init_conv = ConvBNAct(in_spatial_shape=self.in_spatial_shape,
                                   in_channels=self.in_channels,
                                   out_channels=init_conv_out_channels,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   bias=bias,
                                   activation=self.activation,
                                   same_padding=True,
                                   bn_epsilon=bn_epsilon,
                                   bn_momentum=bn_momentum)
        spatial_shape = self.init_conv.out_spatial_shape

        self.stages = nn.ModuleList()
        mbconv_idx = 0
        dc_rates = self.get_dc_rates()
        for stage_id in range(self.num_stages):
            kernel_size = self.get_stage_kernel_size(stage_id)
            stride = self.get_stage_stride(stage_id)
            expansion_factor = self.get_stage_expansion_factor(stage_id)
            stage_in_channels = self.get_stage_in_channels(stage_id)
            stage_out_channels = self.get_stage_out_channels(stage_id)
            stage_num_layers = self.get_stage_num_layers(stage_id)
            stage_dc_rates = dc_rates[mbconv_idx:mbconv_idx + stage_num_layers]
            stage_se_ratio = self.get_stage_se_ratio(stage_id)

            stage = EnetStage(num_layers=stage_num_layers,
                              in_spatial_shape=spatial_shape,
                              in_channels=stage_in_channels,
                              out_channels=stage_out_channels,
                              stride=stride,
                              se_ratio=stage_se_ratio,
                              drop_connect_rates=stage_dc_rates,
                              kernel_size=kernel_size,
                              expansion_factor=expansion_factor,
                              activation=self.activation,
                              bn_epsilon=bn_epsilon,
                              bn_momentum=bn_momentum,
                              bias=bias
                              )
            self.stages.append(stage)
            spatial_shape = stage.out_spatial_shape
            mbconv_idx += stage_num_layers

        head_conv_out_channels = round_filters(1280, width_coefficient)
        head_conv_in_channels = self.stages[-1].layers[-1].project_conv.out_channels
        self.head_conv = ConvBNAct(in_channels=head_conv_in_channels,
                                   out_channels=head_conv_out_channels,
                                   kernel_size=(1, 1),
                                   bias=bias,
                                   activation=self.activation,
                                   bn_epsilon=bn_epsilon,
                                   bn_momentum=bn_momentum)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = None

        # self.avpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(head_conv_out_channels, n_classes)

        if pretrained:
            self._load_state(self.b, in_channels, n_classes, progress)

    @property
    def num_stages(self):
        return len(EfficientNet.stage_args)

    @property
    def width_coefficient(self):
        return EfficientNet.coefficients[self.b][0]

    @property
    def depth_coefficient(self):
        return EfficientNet.coefficients[self.b][1]

    @property
    def dropout_rate(self):
        if self._override_dropout_rate is None:
            return EfficientNet.coefficients[self.b][2]
        else:
            return self._override_dropout_rate

    def get_stage_kernel_size(self, stage):
        return EfficientNet.stage_args[stage][1]

    def get_stage_stride(self, stage):
        return EfficientNet.stage_args[stage][2]

    def get_stage_expansion_factor(self, stage):
        return EfficientNet.stage_args[stage][3]

    def get_stage_in_channels(self, stage):
        width_coefficient = self.width_coefficient
        in_channels = EfficientNet.stage_args[stage][4]
        return round_filters(in_channels, width_coefficient)

    def get_stage_out_channels(self, stage):
        width_coefficient = self.width_coefficient
        out_channels = EfficientNet.stage_args[stage][5]
        return round_filters(out_channels, width_coefficient)

    def get_stage_se_ratio(self, stage):
        return EfficientNet.stage_args[stage][6]

    def get_stage_num_layers(self, stage):
        depth_coefficient = self.depth_coefficient
        num_layers = EfficientNet.stage_args[stage][0]
        return round_repeats(num_layers, depth_coefficient)

    def get_num_mbconv_layers(self):
        total = 0
        for i in range(self.num_stages):
            total += self.get_stage_num_layers(i)
        return total

    def get_dc_rates(self):
        total_mbconv_layers = self.get_num_mbconv_layers()
        return [self.drop_connect_rate * i / total_mbconv_layers
                for i in range(total_mbconv_layers)]

    # def _load_state(self, b, in_channels, n_classes, progress):
        # state_dict = model_zoo.load_url(EfficientNet.state_dict_urls[b], progress=progress, file_name=EfficientNet.dict_names[b])
        # strict = True
        # if in_channels != 3:
        #     state_dict.pop('init_conv.conv.conv.weight')
        #     strict = False
        # if n_classes != 1000:
        #     state_dict.pop('fc.weight')
        #     state_dict.pop('fc.bias')
        #     strict = False
        # self.load_state_dict(state_dict, strict=strict)
        # print("Model weights loaded successfully.")
    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
    def check_input(self, x):
        if x.dim() != 4:
            raise ValueError("Input x must be 4 dimensional tensor, got {} instead".format(x.dim()))
        if x.size(1) != self.in_channels:
            raise ValueError("Input must have {} channels, got {} instead".format(self.in_channels,
                                                                                  x.size(1)))

    @staticmethod
    def check_init_params(b,
                          in_channels,
                          n_classes,
                          in_spatial_shape,
                          activation,
                          bias,
                          drop_connect_rate,
                          override_dropout_rate,
                          bn_epsilon,
                          bn_momentum,
                          pretrained,
                          progress):

        if not isinstance(b, int):
            raise ValueError("b must be int, got {} instead".format(type(b)))
        elif not 0 <= b < len(EfficientNet.coefficients):
            raise ValueError("b must be in range 0<=b<=7, got {} instead".format(b))

        if not isinstance(in_channels, int):
            raise ValueError("in_channels must be int, got {} instead".format(type(in_channels)))
        elif not in_channels > 0:
            raise ValueError("in_channels must be > 0, got {} instead".format(in_channels))

        if not isinstance(n_classes, int):
            raise ValueError("n_classes must be int, got {} instead".format(type(n_classes)))
        elif not n_classes > 0:
            raise ValueError("n_classes must be > 0, got {} instead".format(n_classes))

        if not (in_spatial_shape is None or
                isinstance(in_spatial_shape, int) or
                (isinstance(in_spatial_shape, container_abcs.Iterable) and
                 len(in_spatial_shape) == 2 and
                 all(isinstance(s, int) for s in in_spatial_shape))):
            raise ValueError("in_spatial_shape must be either None, int or iterable of ints of length 2"
                             ", got {} instead".format(in_spatial_shape))

        if activation is not None and not callable(activation):
            raise ValueError("activation must be callable but is not")

        if not isinstance(bias, bool):
            raise ValueError("bias must be bool, got {} instead".format(type(bias)))

        if not isinstance(drop_connect_rate, float):
            raise ValueError("drop_connect_rate must be float, got {} instead".format(type(drop_connect_rate)))
        elif not 0 <= drop_connect_rate < 1.0:
            raise ValueError("drop_connect_rate must be within range 0 <= drop_connect_rate < 1.0, "
                             "got {} instead".format(drop_connect_rate))

        if override_dropout_rate is not None:
            if not isinstance(override_dropout_rate, float):
                raise ValueError("dropout_rate must be either None or float, "
                                 "got {} instead".format(type(override_dropout_rate)))
            elif not 0 <= override_dropout_rate < 1.0:
                raise ValueError("dropout_rate must be within range 0 <= dropout_rate < 1.0, "
                                 "got {} instead".format(override_dropout_rate))

        if not isinstance(bn_epsilon, float):
            raise ValueError("bn_epsilon must be float, got {} instead".format(bn_epsilon))

        if not isinstance(bn_momentum, float):
            raise ValueError("bn_momentum must be float, got {} instead".format(bn_momentum))

        if not isinstance(pretrained, bool):
            raise ValueError("pretrained must be bool, got {} instead".format(type(pretrained)))

        if not isinstance(progress, bool):
            raise ValueError("progress must be bool, got {} instead".format(type(progress)))

    def get_features(self, x):

        self.check_input(x)

        x = self.init_conv(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out

    def forward(self, x):

        x = self.get_features(x)[-1]

        x = self.head_conv(x)

        # x = self.avpool(x)
        # x = torch.flatten(x, 1)

        # if self.dropout is not None:
        #     x = self.dropout(x)
        # x = self.fc(x)

        return x

@BACKBONE_REGISTRY.register()
def build_efficient_backbone(cfg):
    """
    Create a ConvNeXt instance from config.
    Returns:
        ResNet: a :class:`ConvNeXt` instance.
    """

    # fmt: off
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    #last_stride = cfg.MODEL.BACKBONE.LAST_STRIDE
    depth = cfg.MODEL.BACKBONE.DEPTH
    # w = cfg.MODEL.BACKBONE.W
    # s = cfg.MODEL.BACKBONE.S
    # fmt: on

    depth = {
        'b7': 7,
        'b6': 6,
        'b5': 5,
        'b4': 4,
        'b3': 3,
    }[depth]

    model = EfficientNet(b=depth)

    if pretrain:
        # Load pretrain path if specifically
        assert pretrain_path is not None, 'pretrain_path must be given'
        model.load_param(pretrain_path)
    return model