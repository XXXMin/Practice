import torch
from torch import nn
from torchvision.models import resnet18

class MyModel(nn.Module):
    def __init__(self)->None:#定义网络
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=1024, out_features=64)
        self.linear2=nn.Linear(in_features=64, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.conv3(x)
        x=self.maxpool3(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.softmax(x)
        return x


class ImprovedResNet18(nn.Module):
    def __init__(self):
        super(ImprovedResNet18, self).__init__()
        # 加载ResNet18并修改适配32x32图像
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # 去掉最大池化
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    x=torch.randn(1,3,32,32)
    MyModel=MyModel()
    out = MyModel(x)
    print(out)#属于每一个类别的概率是多少

