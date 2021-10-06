"""
Sources:
        https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection
        https://github.com/qiuqiangkong/audioset_tagging_cnn
        https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/211007
        https://www.kaggle.com/c/birdsong-recognition/discussion/183204
        https://www.kaggle.com/c/birdsong-recognition/discussion/183208
        https://arxiv.org/pdf/1912.10211.pdf
        https://arxiv.org/pdf/2102.01243.pdf
        https://arxiv.org/pdf/2104.11587.pdf
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import timm
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from utils.utils import do_mixup

def init_layer(layer):
    """Initializes the layers with the xavier initialization"""

    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvPreWavBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=3, stride=1, dilation=2,
                              padding=2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)


    def forward(self, input, pool_size):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)


    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class AttBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):

        super().__init__()

        # Attention Layer
        self.att = nn.Conv1d(in_channels=in_features,
                                out_channels=out_features,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True
                            )

        # Classification Layer
        self.cla = nn.Conv1d(in_channels=in_features,
                                out_channels=out_features,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True
                            )

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()


    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)


    def forward(self, x):
        # x: (n_samples, n_in, n_time)

        # Tanh is used to allow the attention to be more smooth
        # according with: https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection/comments
        att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(att * cla, dim=2)

        return x, cla


    def nonlinear_transform(self, x):
        return torch.sigmoid(x)


class Cnn_Single_Att(nn.Module):
    def __init__(
            self,
            encoder: str,
            sample_rate: int,
            window_size: int,
            hop_size: int,
            mel_bins: int,
            fmin: int,
            fmax: int,
            encoder_features_num: int,
            embedding_dim: int,
            imagenet_pretrained: bool,
            class_num: int):

        super(Cnn_Single_Att, self).__init__()

        self.encoder = encoder
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.encoder_features_num = encoder_features_num
        self.embedding_dim = embedding_dim
        self.imagenet_pretrained = imagenet_pretrained
        self.class_num = class_num

        self.window = 'hann'
        self.center = True
        self.pad_mode = 'reflect'
        self.ref = 1.0
        self.amin = 1e-10
        self.top_db = None
        self.time_drop_width = 64
        self.time_stripes_num = 2
        self.freq_drop_width = 8
        self.freq_stripes_num = 2

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                    hop_length=hop_size,
                                                    win_length=window_size,
                                                    window=self.window,
                                                    center=self.center,
                                                    pad_mode=self.pad_mode,
                                                    freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                    n_fft=window_size,
                                                    n_mels=mel_bins,
                                                    fmin=fmin, fmax=fmax,
                                                    ref=self.ref,
                                                    amin=self.amin,
                                                    top_db=self.top_db,
                                                    freeze_parameters=True)

        # SpecAugment
        self.spec_augmenter = SpecAugmentation(time_drop_width=self.time_drop_width,
                                                    time_stripes_num=self.time_stripes_num,
                                                    freq_drop_width=self.freq_drop_width,
                                                    freq_stripes_num=self.freq_stripes_num)

        # Model Encoder (Image model backbone)
        self.encoder = timm.create_model(self.encoder, pretrained=self.imagenet_pretrained)

        self.fc1 = nn.Linear(self.encoder_features_num, self.embedding_dim, bias=True)
        self.att_block = AttBlock(self.embedding_dim, self.class_num)
        self.bn0 = nn.BatchNorm2d(self.mel_bins)
        self.init_weight()


    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)


    def forward(self, input, use_specaug=False, mixup_lambda=None) -> Dict:
        """Input : (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x) # Output shape: (batch size, channels = 1, time, frequency)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # SpecAugmentation on spectrogram
        if self.training and use_specaug:
            x = self.spec_augmenter(x)

        # Expand to 3 channels because the EffNet model takes 3 channels as input
        x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3]) # Output shape: (batch size, channels=3, time, frequency)

        # Pass data through encoder
        x = self.encoder.forward_features(x)

        # Colapse the frequency axis
        x = torch.mean(x, dim=3) # Output shape: (batch size, channels, time)

        # Pooling -> max + avg
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        # Fully connected layer
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        clipwise_output, segmentwise_output = self.att_block(x) # clipwise output shape: (batch_size, probs, time)

        segmentwise_output = segmentwise_output.transpose(1, 2) # Output shape: (batch_size, time, probs)

        output_dict = {
            'clipwise_output': clipwise_output,
            'framewise_output': segmentwise_output
        }

        return output_dict

class Wavegram_Logmel_Cnn_Single_Att(nn.Module):
    def __init__(
            self,
            encoder: str,
            sample_rate: int,
            window_size: int,
            hop_size: int,
            mel_bins: int,
            fmin: int,
            fmax: int,
            encoder_features_num: int,
            embedding_dim: int,
            imagenet_pretrained: bool,
            class_num: int):

        super(Wavegram_Logmel_Cnn_Single_Att, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=2, out_channels=64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        # SpecAugment
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        # Model Encoder (Image model backbone)
        self.encoder = timm.create_model(encoder, pretrained=imagenet_pretrained, in_chans=128)

        self.fc1 = nn.Linear(encoder_features_num, embedding_dim, bias=True)
        self.att_block = AttBlock(embedding_dim, class_num)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input, use_specaug=False, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        # print(f"input: {input.shape}")
        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        # print(f"#1: {a1.shape}")
        a1 = self.pre_block1(a1, pool_size=4)
        # print(f"#2: {a1.shape}")
        a1 = self.pre_block2(a1, pool_size=4)
        # print(f"#3: {a1.shape}")
        a1 = self.pre_block3(a1, pool_size=4)
        # print(f"#4: {a1.shape}")
        # print(a1.reshape(a1.shape[0], -1, 64, a1.shape[-1]).shape)
        a1 = a1.reshape((a1.shape[0], -1, 64, a1.shape[-1])).transpose(2, 3)
        # print(f"#After Reshape: {a1.shape}")
        a1 = self.pre_block4(a1, pool_size=(2, 1))
        # print(f"#After polling: {a1.shape}")

        # Log mel spectrogram
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        # print(x.shape)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and use_specaug:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            a1 = do_mixup(a1, mixup_lambda)

        # print(x.shape)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        # print(x.shape)
        # print(a1.shape)

        # Concatenate Wavegram and Log mel spectrogram along the channel dimension
        x = torch.cat((x, a1), dim=1)

        # print(x.shape)

        x = self.encoder.forward_features(x)

        # Colapse the frequency axis
        x = torch.mean(x, dim=3) # Output shape: (batch size, channels, time)

        # Pooling -> max + avg
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        # Fully connected layer
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        clipwise_output, segmentwise_output = self.att_block(x) # clipwise output shape: (batch_size, probs, time)

        segmentwise_output = segmentwise_output.transpose(1, 2) # Output shape: (batch_size, time, probs)

        clipwise_output = clipwise_output.clamp(0,1)
        segmentwise_output = segmentwise_output.clamp(0,1)

        output_dict = {
            'clipwise_output': clipwise_output,
            'framewise_output': segmentwise_output
        }



        return output_dict
