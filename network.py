import torch
import torchvision.models as models
from torch import nn
from torch.nn import functional as F


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }

        return out


class selective_res_block(nn.Module):
    def __init__(self, channels, stride=1, activation=nn.ReLU(inplace=True), norm_type='bn'):
        super(selective_res_block, self).__init__()
        self.conv1 = _make_conv_layer(channels, channels, stride=stride, norm_type=norm_type)
        self.conv2 = _make_conv_layer(channels, channels, stride=stride, norm_type=norm_type)
        self.act = activation
        self.a = nn.Parameter(data=torch.ones(1))
        self.b = nn.Parameter(data=torch.ones(1))

    def forward(self, x):
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = self.act(out.mul(self.a) + identity.mul(self.b)) # weighted sum

        return out


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=nn.ReLU(inplace=True), norm_type='bn'):
        super(encoder, self).__init__()
        self.res_block1 = selective_res_block(in_channels, stride=stride, activation=activation, norm_type=norm_type)
        self.res_block2 = selective_res_block(in_channels, stride=stride, activation=activation, norm_type=norm_type)
        self.downsampler = _make_conv_layer(in_channels, out_channels, stride=2, dilation=1, norm_type=norm_type)
        self.act = activation
        
    def forward(self, x):
        out = self.res_block1(x)
        out = self.res_block2(out)
        out = self.downsampler(out)
        out = self.act(out)

        return out


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=True), norm_type='in'):
        super(decoder, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.res_block1 = selective_res_block(out_channels, activation=activation, norm_type=norm_type)
        self.res_block2 = selective_res_block(out_channels, activation=activation, norm_type=norm_type)

    def forward(self, x, skip):
        out = self.pixel_shuffle(x)
        out = self.res_block1(torch.cat((out, skip), dim=1))
        out = self.res_block2(out)

        return out


class blender(nn.Module):
    def __init__(self, num_scale=4):
        super(blender, self).__init__()
        self.num_scale = num_scale
        self.conv = nn.Conv2d(in_channels=3*num_scale, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        for i, img in enumerate(inputs):
            if i == 0: continue
            assert i < self.num_scale
            inputs[i]= F.interpolate(img, scale_factor=(2**i), mode='bilinear')
        out = torch.cat([*inputs], dim=1)
        
        return self.conv(out)

        
class dehaze_net(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, activation=nn.ReLU(inplace=True)):
        super(dehaze_net, self).__init__()
        
        # Initial convolutional layers for multi-scale inputs
        self.conv10 = nn.Sequential( _make_conv_layer(in_channels, 32, stride=1, norm_type='bn'),
                                    activation,
                                    _make_conv_layer(32, 32, stride=1, norm_type='bn'),
                                    activation
                                )
        self.conv20 = nn.Sequential( _make_conv_layer(in_channels, 64, stride=1, norm_type='bn'),
                                    activation,
                                    _make_conv_layer(64, 64, stride=1, norm_type='bn'),
                                    activation
                                )
        self.conv30 = nn.Sequential( _make_conv_layer(in_channels, 128, stride=1, norm_type='bn'),
                                    activation,
                                    _make_conv_layer(128, 128, stride=1, norm_type='bn'),
                                    activation
                                )
        self.conv40 = nn.Sequential( _make_conv_layer(in_channels, 256, stride=1, norm_type='bn'),
                                    activation,
                                    _make_conv_layer(256, 256, stride=1, norm_type='bn'),
                                    activation
                                )

        # Encoding blocks for the original input
        self.encoder11 = encoder(32, 64, activation=activation)
        self.encoder12 = encoder(64, 128, activation=activation)
        self.encoder13 = encoder(128, 256, activation=activation)

        # Encoding blocks for the (x 1/2) input
        self.encoder21 = encoder(64, 128, activation=activation)
        self.encoder22 = encoder(128, 256, activation=activation)

        # Encoding blocks for the (x 1/4) input
        self.encoder31 = encoder(128, 256, activation=activation)

        # Bottleneck
        self.bottle_neck = nn.Sequential( _make_conv_layer(1024, 512),
                                        activation,
                                        selective_res_block(512, activation=activation)
                                    )
        
        # Decoding blocks
        self.decoder1 = decoder(512, 256, activation=activation)
        self.decoder2 = decoder(256, 128, activation=activation)
        self.decoder3 = decoder(128, 64, activation=activation)

        # Modules for multiscale output
        self.out_conv = nn.Sequential( _make_conv_layer(64, 32),
                                    activation,
                                    _make_conv_layer(32, 3)
                                    )
        self.out_conv_sf_2 = nn.Sequential( _make_conv_layer(128, 64),
                                    activation,
                                    _make_conv_layer(64, 3)
                                    )
        self.out_conv_sf_4 = nn.Sequential( _make_conv_layer(256, 128),
                                    activation,
                                    _make_conv_layer(128, 3)
                                    )
        self.out_conv_sf_8 = nn.Sequential( _make_conv_layer(512, 256),
                                    activation,
                                    _make_conv_layer(256, 3),
                                    )
        
        self.act = activation

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x_original = x
        _, _, h, w = x.size()

        x_down_sf_2 = F.interpolate(x, scale_factor=0.5)
        x_down_sf_4 = F.interpolate(x, scale_factor=0.25)
        x_down_sf_8 = F.interpolate(x, scale_factor=0.125)
        
        x11 = self.conv10(x_original)   # 32, 1/1
        x12 = self.encoder11(x11)       # 64, 1/2
        x13 = self.encoder12(x12)       # 128, 1/4
        x14 = self.encoder13(x13)       # 256, 1/8

        x21 = self.conv20(x_down_sf_2)  # 64, 1/2
        x22 = self.encoder21(x21)       # 128, 1/4
        x23 = self.encoder22(x22)       # 256, 1/8
        
        x31 = self.conv30(x_down_sf_4)  # 128, 1/4
        x32 = self.encoder31(x31)       # 256, 1/8

        x41 = self.conv40(x_down_sf_8)  # 256, 1/8

        out = self.bottle_neck(torch.cat((x14, x23, x32, x41), dim=1)) # 1024 channels
        out_down_sf_8 = self.out_conv_sf_8(out)

        out = self.decoder1(out, (x13 + x22 + x31))
        out_down_sf_4 = self.out_conv_sf_4(out)

        out = self.decoder2(out, (x12 + x21))
        out_down_sf_2 = self.out_conv_sf_2(out)

        out = self.decoder3(out, x11)
        out = self.out_conv(out)
        
        return out, out_down_sf_2, out_down_sf_4, out_down_sf_8


def _make_conv_layer(in_channels, out_channels, stride=1, dilation=1, norm_type='bn'):
        conv_layer = [
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=0, dilation=dilation)
        ]

        if norm_type == 'bn':
            conv_layer.append(nn.BatchNorm2d(out_channels))
        elif norm_type == 'in':
            conv_layer.append(nn.InstanceNorm2d(out_channels))
        
        return nn.Sequential(*conv_layer)
