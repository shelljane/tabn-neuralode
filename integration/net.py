from node import *
from norm import *


class Conv2d4NODE(nn.Module): 
    def __init__(self, chIn, chOut, kernel, stride, padding, relu=True, ntype="batchnorm"): 
        super().__init__()
        self.conv = nn.Conv2d(chIn+1, chOut, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = normlayer(chOut, ntype)
        self.act = nn.ReLU(inplace=True) if relu else nn.Identity()
    
    def forward(self, t, x): 
        x = self.conv(torch.cat([x, t*torch.ones_like(x[:, :1, :, :])], dim=1))
        x = self.bn(t, x)
        x = self.act(x)
        return x


class DeConv2d4NODE(nn.Module): 
    def __init__(self, chIn, chOut, kernel, stride, padding, output_padding, relu=True, ntype="batchnorm"): 
        super().__init__()
        self.conv = nn.ConvTranspose2d(chIn+1, chOut, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding)
        self.bn = normlayer(chOut, ntype)
        self.act = nn.ReLU(inplace=True) if relu else nn.Identity()
    
    def forward(self, t, x): 
        x = self.conv(torch.cat([x, t*torch.ones_like(x[:, :1, :, :])], dim=1))
        x = self.bn(t, x)
        x = self.act(x)
        return x
    

class UNetConv2Dx2(nn.Module): 
    def __init__(self, chIn, chOut, ntype="batchnorm"): 
        super().__init__()
        self.conv1 = Conv2d4NODE(chIn, chOut, 3, 1, 1, ntype=ntype)
        self.conv2 = Conv2d4NODE(chOut, chOut, 3, 1, 1, ntype=ntype)
    def forward(self, t, x): 
        return self.conv2(t, self.conv1(t, x))


def UNetFinal(chIn, chOut, ntype="batchnorm"): 
    return Conv2d4NODE(chIn, chOut, 3, 1, 1, relu=False, ntype=ntype)


def UNetDeConv2D(chIn, chOut, ntype="batchnorm"): 
    return DeConv2d4NODE(chIn, chOut, 3, 2, 1, 1, ntype=ntype)


class UNet(nn.Module): 
    def __init__(self, chIn, chOut, ntype="batchnorm"):
        super().__init__()

        self.dconv_down0 = UNetConv2Dx2(chIn, 32, ntype=ntype)
        self.dconv_down1 = UNetConv2Dx2(32, 64, ntype=ntype)
        self.dconv_down2 = UNetConv2Dx2(64, 128, ntype=ntype)
        self.dconv_down3 = UNetConv2Dx2(128, 256, ntype=ntype)

        self.maxpool = nn.MaxPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = UNetDeConv2D(256, 128, ntype=ntype)
        self.upsample1 = UNetDeConv2D(128, 64, ntype=ntype)
        self.upsample0 = UNetDeConv2D(64, 32, ntype=ntype)

        self.dconv_up2 = UNetConv2Dx2(128 + 128, 128, ntype=ntype)
        self.dconv_up1 = UNetConv2Dx2(64 + 64, 64, ntype=ntype)
        self.dconv_up0 = UNetConv2Dx2(32 + 32, 32, ntype=ntype)

        self.conv_last = UNetFinal(32, chOut, ntype=ntype)

    def forward(self, t, x):
        # encode
        conv0 = self.dconv_down0(t, x) 
        x = self.maxpool(conv0)  
        conv1 = self.dconv_down1(t, x) 
        x = self.maxpool(conv1) 
        conv2 = self.dconv_down2(t, x) 
        x = self.maxpool(conv2) 
        x = self.dconv_down3(t, x)

        # decode
        x = self.upsample2(t, x) 
        x = torch.cat([x, conv2], dim=1) 
        x = self.dconv_up2(t, x) 
        x = self.upsample1(t, x) 
        x = torch.cat([x, conv1], dim=1) 
        x = self.dconv_up1(t, x) 
        x = self.upsample0(t, x) 
        x = torch.cat([x, conv0], dim=1) 
        x = self.dconv_up0(t, x) 
        out = self.conv_last(t, x) 

        return out


class UNetsmall(nn.Module): 
    def __init__(self, chIn, chOut, ntype="batchnorm"):
        super().__init__()

        self.dconv_down0 = UNetConv2Dx2(chIn, 32, ntype=ntype)
        self.dconv_down1 = UNetConv2Dx2(32, 64, ntype=ntype)
        self.dconv_down2 = UNetConv2Dx2(64, 128, ntype=ntype)
        # self.dconv_down3 = UNetConv2Dx2(128, 256, ntype=ntype)

        self.maxpool = nn.MaxPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = UNetDeConv2D(128, 128, ntype=ntype)
        self.upsample1 = UNetDeConv2D(128, 64, ntype=ntype)
        self.upsample0 = UNetDeConv2D(64, 32, ntype=ntype)

        self.dconv_up2 = UNetConv2Dx2(128 + 128, 128, ntype=ntype)
        self.dconv_up1 = UNetConv2Dx2(64 + 64, 64, ntype=ntype)
        self.dconv_up0 = UNetConv2Dx2(32 + 32, 32, ntype=ntype)

        self.conv_last = UNetFinal(32, chOut, ntype=ntype)

    def forward(self, t, x):
        # encode
        conv0 = self.dconv_down0(t, x) 
        x = self.maxpool(conv0)  
        conv1 = self.dconv_down1(t, x) 
        x = self.maxpool(conv1) 
        conv2 = self.dconv_down2(t, x) 
        x = self.maxpool(conv2) 
        # x = self.dconv_down3(t, x)

        # decode
        x = self.upsample2(t, x) 
        x = torch.cat([x, conv2], dim=1) 
        x = self.dconv_up2(t, x) 
        x = self.upsample1(t, x) 
        x = torch.cat([x, conv1], dim=1) 
        x = self.dconv_up1(t, x) 
        x = self.upsample0(t, x) 
        x = torch.cat([x, conv0], dim=1) 
        x = self.dconv_up0(t, x) 
        out = self.conv_last(t, x) 

        return out


class NODEblock0(nn.Module): 
    def __init__(self, chIn, chOut, kernel, stride, padding, ntype="batchnorm"): 
        super().__init__()
        assert chIn == chOut
        self.dim = chOut
        self.conv1 = Conv2d4NODE(chIn, chOut, kernel, stride, padding, ntype=ntype)
        self.conv2 = Conv2d4NODE(chOut, chOut, kernel, 1, padding, relu=False, ntype=ntype)

    def forward(self, t, x0): 
        x = rearrange(x0, 'b d c x y -> b (d c) x y')
        x = self.conv1(t, x)
        x = self.conv2(t, x)
        out = rearrange(x, 'b c x y -> b 1 c x y')
        return out


class NODEblock1(nn.Module): 
    def __init__(self, chIn, chOut, kernel, stride, padding, ntype="batchnorm"): 
        super().__init__()
        assert chIn == chOut
        self.dim = chOut
        basedim = 16 if chIn == 1 else 32
        self.conv1 = Conv2d4NODE(chIn, basedim, kernel, stride, padding, ntype=ntype)
        self.conv2 = Conv2d4NODE(basedim, basedim*2, kernel=3, stride=2, padding=1, ntype=ntype)
        self.conv3 = Conv2d4NODE(basedim*2, basedim*4, kernel=3, stride=2, padding=1, ntype=ntype)
        self.conv4 = DeConv2d4NODE(basedim*4, basedim*2, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)
        self.conv5 = DeConv2d4NODE(basedim*2, basedim, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)
        self.conv6 = Conv2d4NODE(basedim, chOut, kernel, stride, padding, ntype=ntype)

    def forward(self, t, x0): 
        x = rearrange(x0, 'b d c x y -> b (d c) x y')
        x = self.conv1(t, x)
        x = self.conv2(t, x)
        x = self.conv3(t, x)
        x = self.conv4(t, x)
        x = self.conv5(t, x)
        x = self.conv6(t, x)
        out = rearrange(x, 'b c x y -> b 1 c x y')
        return out


class NODEblock2(nn.Module): 
    def __init__(self, size, ntype="batchnorm"): 
        super().__init__()
        self.dim = size
        self.fc1 = nn.Linear(size+1, size)
        self.fc2 = nn.Linear(size+1, size)
        self.bn1 = normlayer(size, ntype if ntype!="batchnorm" else "batchnorm1d")
        self.bn2 = normlayer(size, ntype if ntype!="batchnorm" else "batchnorm1d")
        self.relu = nn.ReLU()

    def forward(self, t, x0): 
        x = torch.flatten(x0, start_dim=1)
        x = self.relu(self.bn1(t, self.fc1(torch.cat([x, t*torch.ones_like(x[:, :1])], dim=1))))
        x = self.relu(self.bn2(t, self.fc2(torch.cat([x, t*torch.ones_like(x[:, :1])], dim=1))))
        out = x.view(x0.shape)
        return out


class NODEblock3(nn.Module): 
    def __init__(self, chIn, chOut, kernel, ntype="batchnorm"): 
        super().__init__()
        assert chIn == chOut
        self.dim = chOut
        basedim = 8 if chIn == 1 else 16
        self.conv1 = Conv2d4NODE(chIn, basedim, kernel, stride=2, padding=1, ntype=ntype)
        self.conv2 = Conv2d4NODE(basedim, basedim, kernel=3, stride=1, padding=1, ntype=ntype)
        self.conv3 = DeConv2d4NODE(basedim, chOut, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)

    def forward(self, t, x0): 
        x = rearrange(x0, 'b d c x y -> b (d c) x y')
        x = self.conv1(t, x)
        x = self.conv2(t, x)
        x = self.conv3(t, x)
        out = rearrange(x, 'b c x y -> b 1 c x y')
        return out


class NODEblock4(nn.Module): 
    def __init__(self, chIn, chOut, kernel, ntype="batchnorm"): 
        super().__init__()
        assert chIn == chOut
        self.dim = chOut
        basedim = 16 if chIn == 1 else 32
        self.conv1 = Conv2d4NODE(chIn, basedim, kernel, stride=2, padding=1, ntype=ntype)
        self.conv2 = Conv2d4NODE(basedim, basedim*2, kernel=3, stride=2, padding=1, ntype=ntype)
        self.conv3 = DeConv2d4NODE(basedim*2, basedim, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)
        self.conv4 = DeConv2d4NODE(basedim, chOut, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)

    def forward(self, t, x0): 
        x = rearrange(x0, 'b d c x y -> b (d c) x y')
        x = self.conv1(t, x)
        x = self.conv2(t, x)
        x = self.conv3(t, x)
        x = self.conv4(t, x)
        out = rearrange(x, 'b c x y -> b 1 c x y')
        return out


class NODEblock5(nn.Module): 
    def __init__(self, chIn, chOut, kernel, ntype="batchnorm"): 
        super().__init__()
        assert chIn == chOut
        self.dim = chOut
        basedim = 16 if chIn == 1 else 32
        self.conv1 = Conv2d4NODE(chIn, basedim, kernel, stride=1, padding=1, ntype=ntype)
        self.conv2 = Conv2d4NODE(basedim, basedim*2, kernel=3, stride=2, padding=1, ntype=ntype)
        self.conv3 = Conv2d4NODE(basedim*2, basedim*4, kernel=3, stride=2, padding=1, ntype=ntype)
        self.conv4 = Conv2d4NODE(basedim*4, basedim*8, kernel=3, stride=2, padding=1, ntype=ntype)
        self.conv5 = DeConv2d4NODE(basedim*8, basedim*4, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)
        self.conv6 = DeConv2d4NODE(basedim*4, basedim*2, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)
        self.conv7 = DeConv2d4NODE(basedim*2, basedim, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)
        self.conv8 = Conv2d4NODE(basedim, chOut, kernel=3, stride=1, padding=1, ntype=ntype)

    def forward(self, t, x0): 
        x = rearrange(x0, 'b d c x y -> b (d c) x y')
        x = self.conv1(t, x)
        x = self.conv2(t, x)
        x = self.conv3(t, x)
        x = self.conv4(t, x)
        x = self.conv5(t, x)
        x = self.conv6(t, x)
        x = self.conv7(t, x)
        x = self.conv8(t, x)
        out = rearrange(x, 'b c x y -> b 1 c x y')
        return out


class NODEblock6(nn.Module): 
    def __init__(self, chIn, chOut, kernel, ntype="batchnorm"): 
        super().__init__()
        assert chIn == chOut
        self.dim = chOut
        basedim = 16 if chIn == 1 else 32
        self.conv1 = Conv2d4NODE(chIn, basedim, kernel, stride=1, padding=1, ntype=ntype)
        self.conv2 = Conv2d4NODE(basedim, basedim*2, kernel=3, stride=2, padding=1, ntype=ntype)
        self.conv3 = Conv2d4NODE(basedim*2, basedim*4, kernel=3, stride=2, padding=1, ntype=ntype)
        self.conv4 = Conv2d4NODE(basedim*4, basedim*8, kernel=3, stride=2, padding=1, ntype=ntype)
        self.convm1 = Conv2d4NODE(basedim*8, basedim*8, kernel=3, stride=1, padding=1, ntype=ntype)
        self.convm2 = Conv2d4NODE(basedim*8, basedim*8, kernel=3, stride=1, padding=1, ntype=ntype)
        self.conv5 = DeConv2d4NODE(basedim*8, basedim*4, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)
        self.conv6 = DeConv2d4NODE(basedim*4, basedim*2, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)
        self.conv7 = DeConv2d4NODE(basedim*2, basedim, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)
        self.conv8 = Conv2d4NODE(basedim, chOut, kernel=3, stride=1, padding=1, ntype=ntype)

    def forward(self, t, x0): 
        x = rearrange(x0, 'b d c x y -> b (d c) x y')
        x = self.conv1(t, x)
        x = self.conv2(t, x)
        x = self.conv3(t, x)
        x = self.conv4(t, x)
        x = self.convm1(t, x)
        x = self.convm2(t, x)
        x = self.conv5(t, x)
        x = self.conv6(t, x)
        x = self.conv7(t, x)
        x = self.conv8(t, x)
        out = rearrange(x, 'b c x y -> b 1 c x y')
        return out


class NODEblock7(nn.Module): 
    def __init__(self, channel=3, ntype="batchnorm"): 
        super().__init__()
        self.dim = channel
        self.conv = UNetsmall(channel, channel, ntype=ntype)
    
    def forward(self, t, x0): 
        x = rearrange(x0, 'b d c x y -> b (d c) x y')
        x = self.conv(t, x)
        out = rearrange(x, 'b c x y -> b 1 c x y')
        return out


class NODEblock8(nn.Module): 
    def __init__(self, chIn, chOut, kernel, ntype="batchnorm"): 
        super().__init__()
        assert chIn == chOut
        self.dim = chOut
        basedim = 16 if chIn == 1 else 32
        self.conv1 = Conv2d4NODE(chIn, basedim, kernel, stride=1, padding=1, ntype=ntype)
        self.conv2 = Conv2d4NODE(basedim, basedim*2, kernel=3, stride=2, padding=1, ntype=ntype)
        self.conv3 = Conv2d4NODE(basedim*2, basedim*4, kernel=3, stride=2, padding=1, ntype=ntype)
        self.conv4 = Conv2d4NODE(basedim*4, basedim*8, kernel=3, stride=2, padding=1, ntype=ntype)
        self.convm1 = Conv2d4NODE(basedim*8, basedim*8, kernel=3, stride=1, padding=1, ntype=ntype)
        self.convm2 = Conv2d4NODE(basedim*8, basedim*8, kernel=3, stride=1, padding=1, ntype=ntype)
        self.convm3 = Conv2d4NODE(basedim*8, basedim*8, kernel=3, stride=1, padding=1, ntype=ntype)
        self.convm4 = Conv2d4NODE(basedim*8, basedim*8, kernel=3, stride=1, padding=1, ntype=ntype)
        self.conv5 = DeConv2d4NODE(basedim*8, basedim*4, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)
        self.conv6 = DeConv2d4NODE(basedim*4, basedim*2, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)
        self.conv7 = DeConv2d4NODE(basedim*2, basedim, kernel=3, stride=2, padding=1, output_padding=1, ntype=ntype)
        self.conv8 = Conv2d4NODE(basedim, chOut, kernel=3, stride=1, padding=1, ntype=ntype)

    def forward(self, t, x0): 
        x = rearrange(x0, 'b d c x y -> b (d c) x y')
        x = self.conv1(t, x)
        x = self.conv2(t, x)
        x = self.conv3(t, x)
        x = self.conv4(t, x)
        x = self.convm1(t, x)
        x = self.convm2(t, x)
        x = self.convm3(t, x)
        x = self.convm4(t, x)
        x = self.conv5(t, x)
        x = self.conv6(t, x)
        x = self.conv7(t, x)
        x = self.conv8(t, x)
        out = rearrange(x, 'b c x y -> b 1 c x y')
        return out


class NODEunet0(nn.Module): 
    def __init__(self, channel=3, ntype="batchnorm"): 
        super().__init__()
        self.dim = channel
        self.conv = UNet(channel, channel, ntype=ntype)
    
    def forward(self, t, x0): 
        x = rearrange(x0, 'b d c x y -> b (d c) x y')
        x = self.conv(t, x)
        out = rearrange(x, 'b c x y -> b 1 c x y')
        return out


class NODEbig0(nn.Sequential): #Params: 2.17M, 2170560
    def __init__(self, channel=3, size=3*32*32, classes=10, ntype="batchnorm"): 
        super().__init__(
            NODEinit(channel, aug=1), 
            NODElayer(NODE(NODEunet0(channel=channel, ntype=ntype))),
            NODEbatchnorm(channel), 
            NODEreadout(), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(size, classes)
        )


class NODEsimple0(nn.Sequential): #Params: 91.84k, 91840
    def __init__(self, channel=3, size=3*32*32, classes=10, ntype="batchnorm"): 
        super().__init__(
            ConvLayer(channel, 64, kernel=3, stride=2, padding=1), 
            nn.BatchNorm2d(64), 
            NODEinit(64, aug=1), 
            NODElayer(NODE(NODEblock0(64, 64, kernel=3, stride=1, padding=1, ntype=ntype))),
            NODEbatchnorm(64), 
            NODEreadout(), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(round(size/channel*16), classes)
        )


class NODEsimple1(nn.Sequential): #Params: 216k, 216768
    def __init__(self, channel=3, size=3*32*32, classes=10, ntype="batchnorm"): 
        super().__init__(
            NODEinit(channel, aug=1), 
            NODElayer(NODE(NODEblock1(channel, channel, kernel=3, stride=1, padding=1, ntype=ntype))),
            NODEbatchnorm(channel), 
            NODEreadout(), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(size, classes)
        )


class NODEsimple2(nn.Sequential): #Params: 36.9k, 36864
    def __init__(self, channel=3, size=3*32*32, classes=10, ntype="batchnorm"): 
        super().__init__(
            NODEinit(channel, aug=1), 
            NODElayer(NODE(NODEblock2(size, ntype=ntype))),
            NODEreadout(), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(size, classes)
        )


class NODEsimple3(nn.Sequential): #Params: 33.9k, 33888
    def __init__(self, channel=3, size=3*32*32, classes=10, ntype="batchnorm"): 
        super().__init__(
            NODEinit(channel, aug=1), 
            NODElayer(NODE(NODEblock3(channel, channel, kernel=3, ntype=ntype))),
            NODEreadout(), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(size, classes)
        )


class NODEsimple4(nn.Sequential): #Params: 69.3k, 69312
    def __init__(self, channel=3, size=3*32*32, classes=10, ntype="batchnorm"): 
        super().__init__(
            NODEinit(channel, aug=1), 
            NODElayer(NODE(NODEblock4(channel, channel, kernel=3, ntype=ntype))),
            NODEbatchnorm(channel), 
            NODEreadout(), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(size, classes)
        )


class NODEsimple5(nn.Sequential):
    def __init__(self, channel=3, size=3*32*32, classes=10, ntype="batchnorm"): 
        super().__init__(
            NODEinit(channel, aug=1), 
            NODElayer(NODE(NODEblock5(channel, channel, kernel=3, ntype=ntype))),
            NODEbatchnorm(channel), 
            NODEreadout(), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(size, classes)
        )


class NODEsimple6(nn.Sequential):
    def __init__(self, channel=3, size=3*32*32, classes=10, ntype="batchnorm"): 
        super().__init__(
            NODEinit(channel, aug=1), 
            NODElayer(NODE(NODEblock6(channel, channel, kernel=3, ntype=ntype))),
            NODEbatchnorm(channel), 
            NODEreadout(), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(size, classes)
        )


class NODEsimple7(nn.Sequential):
    def __init__(self, channel=3, size=3*32*32, classes=10, ntype="batchnorm"): 
        super().__init__(
            NODEinit(channel, aug=1), 
            NODElayer(NODE(NODEblock7(channel=channel, ntype=ntype))),
            NODEbatchnorm(channel), 
            NODEreadout(), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(size, classes)
        )


class NODEsimple8(nn.Sequential):
    def __init__(self, channel=3, size=3*32*32, classes=10, ntype="batchnorm"): 
        super().__init__(
            NODEinit(channel, aug=1), 
            NODElayer(NODE(NODEblock8(channel, channel, kernel=3, ntype=ntype))),
            NODEbatchnorm(channel), 
            NODEreadout(), 
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(size, classes)
        )


def nodenet(name="simple", channel=3, size=3*32*32, classes=10, ntype="batchnorm"): 
    result = None
    if name == "simple" or name == "simple1": 
        result = NODEsimple1(channel=channel, size=size, classes=classes, ntype=ntype)
    elif name == "simple0": 
        result = NODEsimple0(channel=channel, size=size, classes=classes, ntype=ntype)
    elif name == "linear" or name == "simple2": 
        result = NODEsimple2(channel=channel, size=size, classes=classes, ntype=ntype)
    elif name == "simple3": 
        result = NODEsimple3(channel=channel, size=size, classes=classes, ntype=ntype)
    elif name == "simple4": 
        result = NODEsimple4(channel=channel, size=size, classes=classes, ntype=ntype)
    elif name == "simple5": 
        result = NODEsimple5(channel=channel, size=size, classes=classes, ntype=ntype)
    elif name == "simple6": 
        result = NODEsimple6(channel=channel, size=size, classes=classes, ntype=ntype)
    elif name == "simple7": 
        result = NODEsimple7(channel=channel, size=size, classes=classes, ntype=ntype)
    elif name == "simple8": 
        result = NODEsimple8(channel=channel, size=size, classes=classes, ntype=ntype)
    elif name == "unet":
        result = NODEbig0(channel=channel, size=size, classes=classes, ntype=ntype)
    return result



