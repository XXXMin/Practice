import torchvision
import torchvision.transforms as transforms

# 简单的转换（必需）
transform = transforms.Compose([
    transforms.ToTensor()
])

# 下载训练集
trainset = torchvision.datasets.CIFAR10(
    root='C:/Users/翛/Desktop/XIAO/Lou_data',
    train=True,
    download=True,
    transform=transform  # 必须要有transform参数
)

# 下载测试集（可选）
testset = torchvision.datasets.CIFAR10(
    root='C:/Users/翛/Desktop/XIAO/Lou_data',
    train=False,#测试数据集，false
    download=True,
    transform=transform
)

print("下载完成！")
print(f"训练集路径: C:/Users/翛/Desktop/XIAO/Lou_data")