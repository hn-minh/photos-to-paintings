import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, upsample=None):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest') if upsample else None
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.norm(out)
        if self.relu:
            out = self.relu(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer(c, c, 3, 1, relu=True),
            ConvLayer(c, c, 3, 1, relu=False)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            ConvLayer(64, 128, kernel_size=3, stride=2)
        )
        self.residual = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.decoder = nn.Sequential(
            ConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            ConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            ConvLayer(32, 3, kernel_size=9, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residual(x)
        x = self.decoder(x)
        return (x + 1) / 2