import torch
import torch.nn as nn
from pytorch3dunet.unet3d.buildingblocks import DoubleConv, create_encoders, create_decoders
from pytorch3dunet.unet3d.utils import number_of_features_per_level

# 引入 Mamba 和 Diffusion 组件
from mamba import SpatialMamba
from diffusion import TimeEmbedding


class MambaDiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_dim=256,
                 f_maps=(64, 128, 256), layer_order='gcr',
                 num_groups=8, conv_padding=1, upsample='default', dropout_prob=0.1, is3d=True):
        super(MambaDiffusionUNet, self).__init__()

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1

        if 'g' in layer_order:
            assert num_groups is not None

        # 编码器路径
        self.encoders = create_encoders(in_channels, f_maps, DoubleConv, 3, conv_padding, 2,
                                        dropout_prob, layer_order, num_groups, 2, is3d)

        # 中间 Mamba 模块
        self.context_model = SpatialMamba(feature_dim)

        # 时间步嵌入
        self.time_emb = TimeEmbedding(feature_dim)

        # 解码器路径
        self.decoders = create_decoders(f_maps, DoubleConv, 3, conv_padding,
                                        layer_order, num_groups, upsample, dropout_prob, is3d)

        # 输出卷积层
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        # 非线性激活
        self.final_activation = nn.Sigmoid()

    def forward(self, x, t):
        encoders_features = []

        # Encoder 阶段
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        # 添加时间步信息
        time_emb = self.time_emb(t).view(x.shape[0], -1, 1, 1, 1)
        x = x + time_emb

        # Mamba 上下文建模
        x = self.context_model(x)

        # Decoder 阶段
        for decoder, features in zip(self.decoders, encoders_features):
            x = decoder(features, x)

        x = self.final_conv(x)

        # 推理时加激活
        if not self.training:
            x = self.final_activation(x)

        return x
