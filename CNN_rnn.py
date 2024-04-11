import torch.nn as nn
import torch
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.models import vgg16
from PIL import Image
import os

class Vgg16_net(nn.Module):
    def __init__(self):
        super(Vgg16_net, self).__init__()
 
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # 224 * 224 * 64
            nn.BatchNorm2d(64), # Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，一方面使得数据分布一致，另一方面避免梯度消失。
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 224 * 224 * 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 * 112 * 64
        )
 
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 112 * 112 * 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 112 * 112 * 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(2, 2)  # 56 * 56 * 128
        )
 
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(2, 2)  # 28 * 28 * 256
        )
 
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), # 28 * 28 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 28 * 28 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 28 * 28 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(2, 2)  # 14 * 14 * 512
        )
 
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(2, 2)  # 7 * 7 * 512
        )
 
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )
 
        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
 
            nn.Linear(256, 14) # 十四分类问题
        )
 
    def forward(self, x):
        x = self.conv(x)
        # 这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成7*7*512列
        # 那不确定的地方就可以写成-1
        # 如果出现x.size(0)表示的是batchsize的值
        # x=x.view(x.size(0),-1)
        x = x.view(-1, 7*7*512)
        x = self.fc(x)
        return x
 
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
# 定义ResNet18/34的残差结构，为2个3x3的卷积
class BasicBlock(nn.Module):
    # 判断残差结构中，主分支的卷积核个数是否发生变化，不变则为1
    expansion = 1
 
    # init()：进行初始化，申明模型中各层的定义
    # downsample=None对应实线残差结构，否则为虚线残差结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # 使用批量归一化
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 使用ReLU作为激活函数
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
 
        return out
 

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
 
        width = int(out_channel * (width_per_group / 64.)) * groups
 
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        # 使用批量归一化
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        # 使用ReLU作为激活函数
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
 
    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 残差块保留原始输入
        identity = x
        # 如果是虚线残差结构，则进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
 
        return out
 
 
# 定义ResNet类
class ResNet(nn.Module):
    # 初始化函数
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=3,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        # maxpool的输出通道数为64，残差结构输入通道数为64
        self.in_channel = 64
 
        self.groups = groups
        self.width_per_group = width_per_group
 
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 浅层的stride=1，深层的stride=2
        # block：定义的两种残差模块
        # block_num：模块中残差块的个数
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.ca1 = ChannelAttention(512)
        self.sa1 = SpatialAttention()
        if self.include_top:
            # 自适应平均池化，指定输出（H，W），通道数不变
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # 全连接层
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 遍历网络中的每一层
        # 继承nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有modules
        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # kaiming正态分布初始化，使得Conv2d卷积层反向传播的输出的方差都为1
                # fan_in：权重是通过线性层（卷积或全连接）隐性确定
                # fan_out：通过创建随机矩阵显式创建权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
 
    # 定义残差模块，由若干个残差块组成
    # block：定义的两种残差模块，channel：该模块中所有卷积层的基准通道数。block_num：模块中残差块的个数
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 如果满足条件，则是虚线残差结构
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
 
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
 
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        # Sequential：自定义顺序连接成模型，生成网络结构
        return nn.Sequential(*layers)
 
    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 无论哪种ResNet，都需要的静态层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        x = self.maxpool(x)
        # 动态层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

 
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
 
        return x
 
# ResNet()中block参数对应的位置是BasicBlock或Bottleneck
# ResNet()中blocks_num[0-3]对应[3, 4, 6, 3]，表示残差模块中的残差数
# 34层的resnet
def resnet34(num_classes=14, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
 
 
# 50层的resnet
def resnet50(num_classes=14, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
 
 
# 101层的resnet
def resnet101(num_classes=14, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
# 定义VGG16模型
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.features = vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x