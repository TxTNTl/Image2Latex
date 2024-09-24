import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, use_dropout=True):
        super(DenseLayer, self).__init__()
        # extract features, use dropout if necessary
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=True):
        super(TransitionLayer, self).__init__()
        # reduce channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class DenseBlock(nn.Module):
    def __init__(self, n_layers, in_channels, growth_rate, use_dropout=True):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(DenseLayer(in_channels, growth_rate, use_dropout))
            in_channels += growth_rate
        self.dense_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense_block(x)


class DenseNet(nn.Module):
    def __init__(self, growth_rate, block_config=(6, 12, 24, 16), reduction=0.5, use_dropout=True):
        super(DenseNet, self).__init__()

        # Initial image comes in 256 * 256, [batch_size=32, channel=1, height=256, width=256]
        # Initial convolution
        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(1, n_channels, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)

        # [batch_size=32, channel=n_channels, height=128, width=128]
        # DenseBlocks and TransitionLayers
        self.dense1 = self._make_dense_block(n_channels, growth_rate, block_config[0], use_dropout)
        n_channels += block_config[0] * growth_rate
        n_out_channels = int(n_channels * reduction)
        self.trans1 = TransitionLayer(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense2 = self._make_dense_block(n_channels, growth_rate, block_config[1], use_dropout)
        n_channels += block_config[1] * growth_rate
        n_out_channels = int(n_channels * reduction)
        self.trans2 = TransitionLayer(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense3 = self._make_dense_block(n_channels, growth_rate, block_config[2], use_dropout)

    def _make_dense_block(self, in_channels, growth_rate, n_layers, use_dropout):
        return DenseBlock(n_layers, in_channels, growth_rate, use_dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)

        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        # [batch_size, channel=512, height=32, width=32]
        return out
