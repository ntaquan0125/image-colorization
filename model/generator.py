import torch
import torch.nn as nn
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_leaky, use_dropout):
        super(DownBlock, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,\
                        4, 2, 1, bias=False, padding_mode="reflect"),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(.2) if is_leaky else nn.ReLU(),
                )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = self.conv(x)

        return self.dropout(x) if self.use_dropout else x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_leaky, use_dropout):
        super(UpBlock, self).__init__()

        self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels,\
                        4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(.2) if is_leaky else nn.ReLU(),
                )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = self.conv(x)

        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, feature=64):
        super(Generator, self).__init__()

        self.initial_down = nn.Sequential(
                nn.Conv2d(in_channels, feature, 4, 2, 1, padding_mode="reflect"),
                nn.LeakyReLU(.2),
                )
        self.down1 = DownBlock(feature, feature * 2, True, False)
        self.down2 = DownBlock(feature * 2, feature * 4, True, False)
        self.down3 = DownBlock(feature * 4, feature * 8, True, False)
        self.down4 = DownBlock(feature * 8, feature * 8, True, False)
        self.down5 = DownBlock(feature * 8, feature * 8, True, False)
        self.down6 = DownBlock(feature * 8, feature * 8, True, False)
        self.bottleneck = nn.Sequential(
                nn.Conv2d(feature * 8, feature * 8, 4, 2, 1),
                nn.ReLU(),
                )

        self.up1 = UpBlock(feature * 8, feature * 8, False, True)
        self.up2 = UpBlock(feature * 8 * 2, feature * 8, False, True)
        self.up3 = UpBlock(feature * 8 * 2, feature * 8, False, True)
        self.up4 = UpBlock(feature * 8 * 2, feature * 8, False, False)
        self.up5 = UpBlock(feature * 8 * 2, feature * 4, False, False)
        self.up6 = UpBlock(feature * 4 * 2, feature * 2, False, False)
        self.up7 = UpBlock(feature * 2 * 2, feature, False, False)
        self.final_up = nn.Sequential(
                nn.ConvTranspose2d(feature * 2, out_channels, 4, 2, 1),
                nn.Tanh(),
                )

    def forward(self, x):
        init = self.initial_down(x)
        d1 = self.down1(init)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        bottleneck = self.bottleneck(d6)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d6], 1))
        up3 = self.up3(torch.cat([up2, d5], 1))
        up4 = self.up4(torch.cat([up3, d4], 1))
        up5 = self.up5(torch.cat([up4, d3], 1))
        up6 = self.up6(torch.cat([up5, d2], 1))
        up7 = self.up7(torch.cat([up6, d1], 1))

        return self.final_up(torch.cat([up7, init], 1))

def pretrained_generator(size, in_channels, out_channels, backbone, cut):
    body = create_body(backbone, in_channels, True, cut)

    return DynamicUnet(body, out_channels, (size, size))

def test():
    x = torch.rand(32, 1, 256, 256)
    m = Generator(1, 2)

    print(m)
    print(m(x).shape)

    from torchvision.models.resnet import resnet18
    g = pretrained_generator(256, 1, 2, resnet18, -2)
    # g.load_state_dict(torch.load("path"))
    print(g)

if __name__ == "__main__":
    test()
