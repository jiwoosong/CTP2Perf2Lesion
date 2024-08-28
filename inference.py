"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-preliminary-docker-evaluation | gzip -c > example-algorithm-preliminary-docker-evaluation.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import PIL
import monai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import itertools
import os
import re
from glob import glob
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm


import matplotlib

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None

def window_mapping(arr, src_range=(-1024, 3072), tgt_range=(0,1), clip=True):
    arr = (arr-src_range[0]) / (src_range[1]-src_range[0])
    arr = arr * (tgt_range[1]-tgt_range[0]) + tgt_range[0]
    if clip:
        arr = arr.clip(min(tgt_range), max(tgt_range))
    return arr

def croppad(input_array, z_range=None, y_range=None, x_range=None, pad_mode='constant', pad_val=0):
    # Define the target size based on the ranges provided
    if z_range is None:
        z_range = [0, input_array.shape[-3]]
    if y_range is None:
        y_range = [0, input_array.shape[-2]]
    if x_range is None:
        x_range = [0, input_array.shape[-1]]
    target_size = (z_range[1] - z_range[0], y_range[1] - y_range[0], x_range[1] - x_range[0])

    # Crop the array based on the specified ranges
    cropped_array = input_array[...,
                    max(z_range[0], 0):min(z_range[1], input_array.shape[-3]),
                    max(y_range[0], 0):min(y_range[1], input_array.shape[-2]),
                    max(x_range[0], 0):min(x_range[1], input_array.shape[-1])
                    ]

    # Calculate the padding needed
    pad_z = (max(-z_range[0], 0), max(z_range[1] - input_array.shape[-3], 0))
    pad_y = (max(-y_range[0], 0), max(y_range[1] - input_array.shape[-2], 0))
    pad_x = (max(-x_range[0], 0), max(x_range[1] - input_array.shape[-1], 0))

    pad_width = (pad_x, pad_y, pad_z)

    # Pad the cropped array
    padded_array = torch.nn.functional.pad(cropped_array,
                                           [pad_x[0], pad_x[1],
                                            pad_y[0], pad_y[1],
                                            pad_z[0], pad_z[1]],
                                           mode=pad_mode, value=pad_val)

    return padded_array


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.LeakyReLU(inplace=True)
        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()
        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)

class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.InstanceNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.InstanceNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)

class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.InstanceNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class DenseNet3D_Encoder(nn.Module):
    def __init__(self,
                 input_size=(50, 256, 256),
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):
        super().__init__()

        T, H, W = input_size
        self.out_size = []

        # First convolution
        self.dense_block_list = []
        features = [('conv1', nn.Conv3d(n_input_channels,
                                        num_init_features,
                                        kernel_size=(3, 5, 5),
                                        stride=(1, 1, 1),
                                        padding=(2, 2, 2),
                                        bias=False)),
                    ('norm1', nn.InstanceNorm3d(num_init_features)),
                    ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            features.append(('pool1', nn.MaxPool3d(kernel_size=3, stride=(2, 2, 2), padding=1)))
        features = nn.Sequential(OrderedDict(features))
        self.dense_block_list.append(features)
        H, W = H//2, W//2
        self.out_size.append((num_init_features,H, W))


        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            dense_block = []
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            dense_block.append(('denseblock{}'.format(i + 1), block))
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                dense_block.append(('transition{}'.format(i + 1), trans))
                num_features = num_features // 2
                H, W = H // 2, W // 2
                self.out_size.append((num_features, H, W))
            self.dense_block_list.append(nn.Sequential(OrderedDict(dense_block)))

        # Final batch norm
        self.dense_block_list[-1].add_module('norm5', nn.InstanceNorm3d(num_features))
        self.dense_block_list[-1].add_module('pool5', nn.MaxPool3d(kernel_size=3, stride=(2, 2, 2), padding=1))
        H, W = H//2, W//2
        self.out_size.append((num_features,H, W))
        self.dense_block_list = nn.ModuleList(self.dense_block_list)

        for m in self.dense_block_list.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, return_feat=False):
        B, C, T, H, W = x.shape
        feat_list = []
        for i, layer in enumerate(self.dense_block_list):
            x = layer(x)
            C1, T1, H1, W1 = x.shape[-4:]
            feat = self.avg_pool_1d(x.permute(0,3,4,1,2).reshape(B*H1*W1,C1,T1)).reshape(B,H1,W1,C1).permute(0,3,1,2)
            feat_list.append(feat)
        return feat_list

class Seg_Decoder(nn.Module):
    def __init__(self, out_size=(1, 256, 256), hidden_dims=[], dtype=torch.float16, device='cuda:0'):
        super().__init__()
        self.step_size = hidden_dims[::-1] + [list(out_size)]
        self.module_list = nn.ModuleList()

        # initial conv
        self.module_list.append(nn.Sequential(Conv2dReLU(self.step_size[0][0], self.step_size[0][0], kernel_size=1, stride=1, padding=0, ), ))

        for i in range(len(self.step_size[1:]) - 1):
            cur_size = self.step_size[i]
            next_size = self.step_size[i + 1]

            if cur_size[-2] * 2 != next_size[-2] or cur_size[-1] * 2 != next_size[-1]:
                raise NotImplementedError
            else:


                if i == 0:
                    self.module_list.append(DecoderBlock(cur_size[0], next_size[0], next_size[0], use_batchnorm=False, attention_type='scse',))
                # elif i == len(self.step_size) - 1:
                #     self.module_list.append(DecoderBlock(self.step_size[-2][0], 0, self.step_size[-1][0], use_batchnorm=False, attention_type=None))
                else:
                    self.module_list.append(DecoderBlock(cur_size[0], next_size[0], next_size[0], use_batchnorm=False, attention_type='scse', ))
        # final conv
        self.module_list.append(DecoderBlock(self.step_size[-2][0], 0, self.step_size[-1][0], use_batchnorm=False, attention_type=None ))

    def forward(self, feat_list):

        for i, feat in enumerate(feat_list[::-1]):
            if i == 0:
                result = self.module_list[i](feat)
            else:
                # torch.cat([feat, result], dim=1)
                result = self.module_list[i](result, skip=feat)
        result = self.module_list[-1](result)
        return result

class ctp4D_cnn3D_solver(nn.Module):
    def __init__(self, input_size=(1, 50, 256, 256), output_channel=5, encoder='densenet121'):
        super().__init__()
        C, T, H, W = input_size

        if encoder == 'densenet121':
            self.spatial_3D_encoder = DenseNet3D_Encoder(input_size=(C, H, W),
                                                         num_init_features=32,
                                                         growth_rate=8,
                                                         block_config=(6, 12, 24, 16),
                                                         n_input_channels=1,
                                                         conv1_t_size=3,
                                                         conv1_t_stride=1,
                                                         no_max_pool=False,
                                                         bn_size=4,
                                                         drop_rate=0,
                                                         num_classes=1)
        self.enc_vec_size = self.spatial_3D_encoder.out_size[-1]
        self.hpm_decoder = Seg_Decoder(out_size=(output_channel, H, W), hidden_dims=self.spatial_3D_encoder.out_size)
    def forward(self, x):
        feat_list = self.spatial_3D_encoder(x)
        hpm_out = self.hpm_decoder(feat_list)
        return hpm_out

    def freeze_backbone(self, requires_grad):
        for name, param in self.spatial_3D_encoder.named_parameters():
            param.requires_grad = requires_grad

        for name, param in self.hpm_decoder.named_parameters():
            param.requires_grad = requires_grad


def load_checkpoint(net, weight_param, key='', p=True):
    '''
    :param net: Network
    :param param: Weight params
    :param p: print option
    :return: Network
    '''
    model_dict = net.state_dict()
    for name, param in weight_param.items():
        if name.__contains__(key):
            if key != '':
                name = '.'.join(name.split('.')[1:])
            if name not in model_dict:
                if p:
                    pass
                    print(name + 'Not Loaded')
                continue
            else:
                if model_dict[name].shape == param.shape:
                    model_dict[name].copy_(param)
                    # print('* Loaded ' + name)
                else:
                    print(name + 'Not Loaded (shape difference)')
    return net

def sitk2tensor(sitk_image, dtype=torch.int16, device=torch.device('cuda:0')):
    data = sitk.GetArrayFromImage(sitk_image)
    data = torch.tensor(data, dtype=dtype, device=device)

    if len(data.shape) == 5:
        pass
    elif len(data.shape) == 3:
        data = data[None, None]
    elif len(data.shape) == 4:
        if data.shape[0] == 3 or data.shape[0] == 4:
            data = data[None]
        else:
            data = data[:, None]
    else:
        raise NotImplementedError
    return data

def get_colormap_image_cuda(tensor_image, cmap='jet', vmin=0, vmax=120):
    rgb_img = window_mapping(tensor_image, (vmin, vmax), (0, 1), clip=True)
    colormap = torch.tensor(matplotlib.colormaps.get_cmap(cmap)(np.linspace(0, 1, 512)), dtype=torch.float32,
                            device=tensor_image.device)
    if cmap in ['jet']:
        colormap[0] *= 0
    rgb_img = (colormap[(rgb_img * (colormap.shape[0] - 1)).long()] * 255).int()
    return rgb_img

def predict_infarct(sitk_ctp, hpm_reg_model=None, lesion_seg_model=None, op_dtype=torch.float32,
                    op_device=torch.device('cuda:0')):
    # ncct = window_mapping(-100, 200)

    ctp_arr = sitk.GetArrayFromImage(sitk_ctp).astype(np.int16)
    T, S0, H0, W0 = ctp_arr.shape
    spz, spy, spx = sitk_ctp.GetSpacing()[::-1][-3:]

    if T < 50:
        ts, te = (T - 50, T)
    else:
        ts, te = (0, 50)
    hs, ws = (0, 0)
    he, we = (256, 256)
    H1, W1 = int(H0 / (1.0 / spy)), int(W0 / (1.0 / spx))

    ctp_mean_slice = []
    cbf_pred = []
    tmax_pred = []
    lesion_reg_pred = []
    lesion_seg_pred = []

    with torch.no_grad():
        hpm_reg_model = hpm_reg_model.to(op_dtype).to(op_device)
        hpm_reg_model = hpm_reg_model.eval()

        lesion_seg_model = lesion_seg_model.to(op_dtype).to(op_device)
        lesion_seg_model = lesion_seg_model.eval()
        for s_idx in range(S0):
            ctp_slice = torch.as_tensor(ctp_arr[:, s_idx], dtype=op_dtype, device=op_device)  # THW
            ctp_mean_slice.append(window_mapping(ctp_slice.mean(dim=0), (-100, 200), (0, 1), clip=True))
            ctp_slice = torch.nn.functional.interpolate(ctp_slice[:, None], size=(H1, W1), mode='bilinear').squeeze(
                1)  # TCHW
            ctp_slice = croppad(ctp_slice, z_range=(ts, te), y_range=(0, 256), x_range=(0, 256), pad_mode='constant',
                                pad_val=-1000)
            ctp_slice = window_mapping(ctp_slice, (-100, 1024), (0, 1), clip=False)
            ctp_slice = ctp_slice.squeeze(1)[None, None]  # BCTHW

            hpm_pred = hpm_reg_model(ctp_slice)
            lesion_pred = lesion_seg_model(hpm_pred[:, :3])
            lesion_pred = torch.sigmoid(lesion_pred)

            cbf_pred.append(
                croppad(hpm_pred[:, 0], y_range=(0, H1), x_range=(0, W1), pad_mode='constant', pad_val=0).squeeze(0))
            tmax_pred.append(
                croppad(hpm_pred[:, 1], y_range=(0, H1), x_range=(0, W1), pad_mode='constant', pad_val=0).squeeze(0))
            lesion_reg_pred.append(
                croppad(hpm_pred[:, 2], y_range=(0, H1), x_range=(0, W1), pad_mode='constant', pad_val=0).squeeze(0))
            lesion_seg_pred.append(
                croppad(lesion_pred[:, 0], y_range=(0, H1), x_range=(0, W1), pad_mode='constant', pad_val=0).squeeze(0))

        ctp_mean_slice = torch.nn.functional.interpolate(torch.stack(ctp_mean_slice, dim=0)[:, None], size=(H0, W0),
                                                         mode='bilinear').squeeze(1)
        cbf_pred = torch.nn.functional.interpolate(torch.stack(cbf_pred, dim=0)[:, None], size=(H0, W0),
                                                   mode='bilinear').squeeze(1)
        tmax_pred = torch.nn.functional.interpolate(torch.stack(tmax_pred, dim=0)[:, None], size=(H0, W0),
                                                    mode='bilinear').squeeze(1)
        lesion_reg_pred = torch.nn.functional.interpolate(torch.stack(lesion_reg_pred, dim=0)[:, None], size=(H0, W0),
                                                          mode='bilinear').squeeze(1)
        lesion_seg_pred = torch.nn.functional.interpolate(torch.stack(lesion_seg_pred, dim=0)[:, None], size=(H0, W0),
                                                          mode='bilinear').squeeze(1)

        cbf_pred = (cbf_pred * 120).type(torch.float32).round(decimals=4)
        tmax_pred = (tmax_pred * 12).type(torch.float32).round(decimals=4)
        lesion_reg_pred = lesion_reg_pred.type(torch.float32).round(decimals=4)
        lesion_reg_binarize = (lesion_reg_pred > 0.5).type(torch.int16)
        lesion_seg_pred = lesion_seg_pred.type(torch.float32).round(decimals=4)
        lesion_seg_binarize = (lesion_seg_pred > 0.5).type(torch.int16)

    return ctp_mean_slice, cbf_pred, tmax_pred, lesion_reg_pred, lesion_reg_binarize, lesion_seg_pred, lesion_seg_binarize

def run(save_suppl=False):
    # _show_torch_cuda_info()

    op_dtype = torch.float32
    if torch.cuda.is_available():
        op_device = torch.device('cuda:0')
    else:
        op_device = torch.device('cpu')

    INPUT_PATH = Path("/input")
    OUTPUT_PATH = Path("/output")
    RESOURCE_PATH = Path("resources")

    hpm_reg_model = ctp4D_cnn3D_solver(input_size=(1, 50, 256, 256), output_channel=5, encoder='densenet121')
    hpm_reg_model = load_checkpoint(hpm_reg_model, torch.load(RESOURCE_PATH / 'hpm_reg_model.pth', weights_only=True),
                                    '')

    lesion_seg_model = monai.networks.nets.UNet(spatial_dims=2, in_channels=3, out_channels=1,
                                                channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
    lesion_seg_model = load_checkpoint(lesion_seg_model,
                                       torch.load(RESOURCE_PATH / 'hpm_lesion_seg_model_E35.pth', weights_only=True),
                                       '')

    sitk_ctp = load_image(location=INPUT_PATH / "images/preprocessed-perfusion-ct")

    ctp_img, cbf_pred, tmax_pred, lesion_reg_pred, lesion_reg_binarize, lesion_seg_pred, lesion_seg_binarize = \
        predict_infarct(sitk_ctp, hpm_reg_model, lesion_seg_model, op_dtype, op_device)

    # Save your output
    write_tensor_as_image_file(
        location=OUTPUT_PATH / "images/stroke-lesion-segmentation",
        array=lesion_seg_binarize,
        print_sum=False
    )

    if save_suppl:
        location = OUTPUT_PATH / "images/supplementary"
        location.mkdir(parents=True, exist_ok=True)
        S, H, W = ctp_img.shape
        for s_idx in tqdm(range(S), desc='save supplementary', ncols=100):
            ctp_rgb_slice = get_colormap_image_cuda(ctp_img[s_idx], cmap='gray', vmin=0, vmax=1)
            cbf_rgb_slice = get_colormap_image_cuda(cbf_pred[s_idx], cmap='jet', vmin=0, vmax=120)
            tmax_rgb_slice = get_colormap_image_cuda(tmax_pred[s_idx], cmap='jet', vmin=0, vmax=12)
            lesion_reg_rgb_slice = get_colormap_image_cuda(lesion_reg_pred[s_idx], cmap='jet', vmin=0.01, vmax=1)
            msk = lesion_reg_rgb_slice[..., -1] != 0
            lesion_reg_rgb_slice = ctp_rgb_slice * (~msk[..., None]) + lesion_reg_rgb_slice * msk[..., None]
            lesion_seg_rgb_slice = get_colormap_image_cuda(lesion_seg_pred[s_idx], cmap='jet', vmin=0.01, vmax=1)
            msk = lesion_seg_rgb_slice[..., -1] != 0
            lesion_seg_rgb_slice = ctp_rgb_slice * (~msk[..., None]) + lesion_seg_rgb_slice * msk[..., None]
            img_fig = torch.cat(
                [ctp_rgb_slice, cbf_rgb_slice, tmax_rgb_slice, lesion_reg_rgb_slice, lesion_seg_rgb_slice], dim=1)
            PIL.Image.fromarray(img_fig.clip(0, 255).type(torch.uint8).cpu().numpy()).save(
                location / ("%04d.png" % (s_idx)))
        pass

def load_image(*, location):
    # Use sitk to read a file
    input_files = glob(str(location / "*.mha"))
    result = sitk.ReadImage(input_files[0])
    return result

def write_tensor_as_image_file(*, location, array, print_sum=False):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"
    print(str(location / f"output{suffix}"))
    if print_sum:
        print(array.sum().item())
    image = sitk.GetImageFromArray(array.cpu().numpy())
    sitk.WriteImage(image, location / f"output{suffix}", useCompression=True)

def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

if __name__ == "__main__":
    raise SystemExit(run(save_suppl=False))
