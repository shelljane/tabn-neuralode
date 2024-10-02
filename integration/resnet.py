import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ConvLayer(nn.Sequential): 
    def __init__(self, chIn, chOut, kernel, stride, padding, relu=True, init=None): 
        super().__init__(
            nn.Conv2d(chIn, chOut, kernel_size=kernel, stride=stride, padding=padding), 
            nn.BatchNorm2d(chOut), 
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )
        if not init is None: 
            layer = self[0]
            torch.nn.init.normal_(layer.weight, mean=0.0, std=init)
            torch.nn.init.normal_(layer.bias, mean=0.0, std=init)

class DeConvLayer(nn.Sequential): 
    def __init__(self, chIn, chOut, kernel, stride, padding, output_padding, relu=True, init=None): 
        super().__init__(
            nn.ConvTranspose2d(chIn, chOut, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding), 
            nn.BatchNorm2d(chOut), 
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )
        if not init is None: 
            layer = self[0]
            torch.nn.init.normal_(layer.weight, mean=0.0, std=init)
            torch.nn.init.normal_(layer.bias, mean=0.0, std=init)

class LinearLayer(nn.Sequential): 
    def __init__(self, chIn, chOut): 
        super().__init__(
            nn.Conv2d(chIn, chOut), 
            nn.BatchNorm2d(chOut), 
            nn.ReLU(inplace=True)
        )

class ResV1Layer(nn.Module): 
    def __init__(self, chIn, chOut, kernel, stride, padding): 
        super().__init__()
        self.conv1 = ConvLayer(chIn, chOut, kernel, stride, padding)
        self.conv2 = ConvLayer(chOut, chOut, kernel, 1, padding)
        self.skip = nn.Conv2d(chIn, chOut, kernel_size=1, stride=stride, padding=0) if chIn != chOut or stride > 1 else nn.Identity()

    def forward(self, x0): 
        x = self.conv1(x0)
        x = self.conv2(x)
        return x + self.skip(x0)

class ResV2Layer(nn.Module): 
    def __init__(self, chIn, chOut, kernel, stride, padding): 
        super().__init__()
        self.norm0 = nn.BatchNorm2d(chIn)
        self.relu0 = nn.ReLU()
        self.conv1 = ConvLayer(chIn, chOut, kernel, stride, padding)
        self.conv2 = nn.Conv2d(chOut, chOut, kernel, 1, padding)
        self.skip = nn.Conv2d(chIn, chOut, kernel_size=1, stride=stride, padding=0) if chIn != chOut or stride > 1 else nn.Identity()

    def forward(self, x0): 
        x = self.norm0(x0) 
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + self.skip(x0)


class ConvNetTiny(nn.Sequential): 
    def __init__(self): 
        super().__init__(
            ConvLayer(3, 64, kernel=3, stride=1, padding=1), 
            nn.MaxPool2d(3, stride=2, padding=1), 
            ConvLayer(64, 64, kernel=3, stride=1, padding=1),
            ConvLayer(64, 64, kernel=3, stride=1, padding=1),
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(16384, 10)
        )


class ResNetTiny(nn.Sequential): 
    def __init__(self): 
        super().__init__(
            ConvLayer(3, 64, kernel=3, stride=1, padding=1), 
            ResV2Layer(64, 64, kernel=3, stride=2, padding=1), 
            ResV2Layer(64, 64, kernel=3, stride=2, padding=1), 
            ResV2Layer(64, 64, kernel=3, stride=1, padding=1), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(4096, 10)
        )
        

class ResNetSlim(nn.Sequential): 
    def __init__(self): 
        super().__init__(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(3, stride=2, padding=1), 
            ResV2Layer(64, 64, kernel=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1), 
            ResV2Layer(64, 128, kernel=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1), 
            ResV2Layer(128, 256, kernel=3, stride=1, padding=1),
            ResV2Layer(256, 512, kernel=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Linear(512, 10)
        )


class ResNet18V2(nn.Sequential): 
    def __init__(self): 
        super().__init__(
            ConvLayer(3, 64, kernel=3, stride=1, padding=1), 
            nn.MaxPool2d(3, stride=2, padding=1), 
            ResV2Layer(64, 64, kernel=3, stride=1, padding=1), 
            ResV2Layer(64, 64, kernel=3, stride=1, padding=1), 
            ResV2Layer(64, 128, kernel=3, stride=2, padding=1), 
            ResV2Layer(128, 128, kernel=3, stride=1, padding=1), 
            ResV2Layer(128, 256, kernel=3, stride=2, padding=1), 
            ResV2Layer(256, 256, kernel=3, stride=1, padding=1), 
            ResV2Layer(256, 512, kernel=3, stride=1, padding=1), 
            ResV2Layer(512, 512, kernel=3, stride=1, padding=1), 
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Linear(512, 10)
        )
        

class ResNet50V2(nn.Sequential): 
    def __init__(self): 
        super().__init__(
            ConvLayer(3, 64, kernel=3, stride=1, padding=1), 
            nn.MaxPool2d(3, stride=2, padding=1), 
            ResV2Layer(64, 64, kernel=3, stride=1, padding=1), 
            ResV2Layer(64, 64, kernel=3, stride=1, padding=1), 
            ResV2Layer(64, 64, kernel=3, stride=1, padding=1), 
            ResV2Layer(64, 128, kernel=3, stride=2, padding=1), 
            ResV2Layer(128, 128, kernel=3, stride=1, padding=1), 
            ResV2Layer(128, 128, kernel=3, stride=1, padding=1), 
            ResV2Layer(128, 128, kernel=3, stride=1, padding=1), 
            ResV2Layer(128, 256, kernel=3, stride=2, padding=1), 
            ResV2Layer(256, 256, kernel=3, stride=1, padding=1), 
            ResV2Layer(256, 256, kernel=3, stride=1, padding=1), 
            ResV2Layer(256, 256, kernel=3, stride=1, padding=1), 
            ResV2Layer(256, 256, kernel=3, stride=1, padding=1), 
            ResV2Layer(256, 256, kernel=3, stride=1, padding=1), 
            ResV2Layer(256, 512, kernel=3, stride=1, padding=1), 
            ResV2Layer(512, 512, kernel=3, stride=1, padding=1), 
            ResV2Layer(512, 512, kernel=3, stride=1, padding=1), 
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Linear(512, 10)
        )

        

class Simple1Net(nn.Sequential): 
    def __init__(self): 
        super().__init__(
            ConvLayer(3, 32, 3, 1, 1), 
            ConvLayer(32, 64, 3, 2, 1), 
            ConvLayer(64, 128, 3, 2, 1), 
            DeConvLayer(128, 64, 3, 2, 1, 1), 
            DeConvLayer(64, 32, 3, 2, 1, 1), 
            ConvLayer(32, 3, 3, 1, 1), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(3*32*32, 10)
        )
    

class UNetConv2Dx2(nn.Module): 
    def __init__(self, chIn, chOut): 
        super().__init__()
        self.conv1 = ConvLayer(chIn, chOut, 3, 1, 1)
        self.conv2 = ConvLayer(chOut, chOut, 3, 1, 1)
    def forward(self, x): 
        return self.conv2(self.conv1(x))


def UNetFinal(chIn, chOut): 
    return ConvLayer(chIn, chOut, 3, 1, 1, relu=False)


def UNetDeConv2D(chIn, chOut): 
    return DeConvLayer(chIn, chOut, 3, 2, 1, 1)

class UNet(nn.Module): 
    def __init__(self, chIn, chOut):
        super().__init__()

        self.dconv_down0 = UNetConv2Dx2(chIn, 32)
        self.dconv_down1 = UNetConv2Dx2(32, 64)
        self.dconv_down2 = UNetConv2Dx2(64, 128)
        self.dconv_down3 = UNetConv2Dx2(128, 256)

        self.maxpool = nn.MaxPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = UNetDeConv2D(256, 128)
        self.upsample1 = UNetDeConv2D(128, 64)
        self.upsample0 = UNetDeConv2D(64, 32)

        self.dconv_up2 = UNetConv2Dx2(128 + 128, 128)
        self.dconv_up1 = UNetConv2Dx2(64 + 64, 64)
        self.dconv_up0 = UNetConv2Dx2(32 + 32, 32)

        self.conv_last = UNetFinal(32, chOut)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(3*32*32, 10)

    def forward(self, x):
        # encode
        conv0 = self.dconv_down0(x) 
        x = self.maxpool(conv0)  
        conv1 = self.dconv_down1(x) 
        x = self.maxpool(conv1) 
        conv2 = self.dconv_down2(x) 
        x = self.maxpool(conv2) 
        x = self.dconv_down3(x)

        # decode
        x = self.upsample2(x) 
        x = torch.cat([x, conv2], dim=1) 
        x = self.dconv_up2(x) 
        x = self.upsample1(x) 
        x = torch.cat([x, conv1], dim=1) 
        x = self.dconv_up1(x) 
        x = self.upsample0(x) 
        x = torch.cat([x, conv0], dim=1) 
        x = self.dconv_up0(x) 
        x = self.conv_last(x)
        x = self.flatten(x)
        x = self.dropout(x)
        out = self.output(x)

        return out
