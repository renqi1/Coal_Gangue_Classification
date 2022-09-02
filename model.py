import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.covn1 = nn.Sequential(       # 原始图片为（1，224，224）
            nn.Conv2d(1, 8, 5, 2),        # 卷积，卷积核5×5，步长2， (8,110,110)
            nn.ReLU(),                    # ReLU激活函数
            nn.MaxPool2d(2),              # 最大池化，池化核2×2，步长2， (8,55,55)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1),        # (16,53,53)
            nn.ReLU(),
            nn.MaxPool2d(2),               # (16,26,26)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1),       # (32,24,24)
            nn.ReLU(),
            nn.MaxPool2d(2),               # (32,12，12）
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),       # (64,10,10)
            nn.ReLU(),
            nn.MaxPool2d(2),               # (64,5,5)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1),       # (64,1,1)
        )
        self.layer1 = nn.Linear(64*1*1, 2)  # 全连接层将它展平  2类
        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
#前向传播
    def forward(self, x):
        x = self.covn1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        output = self.layer1(x)
        return output



