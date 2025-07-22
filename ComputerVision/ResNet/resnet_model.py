import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
        def __init__(self, input_channels, output_channel, stride = 1, downsample = None):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channel, kernel_size = 3, stride = stride, padding = 1), nn.BatchNorm2d(output_channel), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(output_channel, output_channel, kernel_size = 3, stride = 1, padding = 1), nn.BatchNorm2d(output_channel))
            self.downsample = downsample
            self.relu = nn.ReLU()
            self.output_channel = output_channel

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample:
                residual = self.downsample(x)
            out = out + residual
            out = self.relu(out)
            return out

class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes = 10):
            super(ResNet, self).__init__()
            self.inplanes = 64
            self.conv1 = nn.Sequential( nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            self.layer0 = self.layer(block, 64, layers[0], stride = 1)
            self.layer1 = self.layer(block, 128, layers[1], stride = 2)
            self.layer2 = self.layer(block, 256, layers[2], stride = 2)
            self.layer3 = self.layer(block, 512, layers[3], stride = 2)
            self.fc = nn.Linear(512, num_classes)

        def layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride), nn.BatchNorm2d(planes))
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x