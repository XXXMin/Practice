import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# 1. 设置设备（优先 MPS）
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("✅ 使用 MPS (Apple Silicon GPU) 加速")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ 使用 CUDA (NVIDIA GPU) 加速")
else:
    device = torch.device('cpu')
    print("⚠️ 使用 CPU，训练会很慢")

# ========== 1. 准备数据 ==========
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# 加载数据集（同时加载测试集，用于评估）
trainset = torchvision.datasets.CIFAR10(
    root='C:/Users/翛/Desktop/XIAO/Python_project/CIFAR10/CIFAR10', train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root='C:/Users/翛/Desktop/XIAO/Python_project/CIFAR10/CIFAR10', train=False, download=True, transform=transform_test
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# ========== 2. 加载教师模型（预训练 ResNet）==========
teacher = torchvision.models.resnet18(pretrained=True)
teacher.fc = nn.Linear(512, 10)
teacher = teacher.to(device)
teacher.eval()

# 设置环境变量，使用国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TIMM_USE_HF_HUB"] = "False"

import timm
from transformers import AutoModel, AutoTokenizer
# 将 HF 域名指向国内镜像站
# ========== 3. 加载学生模型（DeiT）==========
student = timm.create_model('deit_small_distilled_patch16_224', pretrained=False)
student.head = nn.Linear(student.head.in_features, 10)
student.head_dist = nn.Linear(student.head_dist.in_features, 10)
student = student.to(device)


# ========== 4. 蒸馏损失函数 ==========
def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=4.0, alpha=0.7):
    hard_loss = F.cross_entropy(student_logits, labels)

    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    soft_loss = soft_loss * (temperature ** 2)

    total_loss = (1 - alpha) * hard_loss + alpha * soft_loss
    return total_loss, hard_loss, soft_loss


# ========== 5. 计算准确率的函数 ==========
def compute_accuracy(loader):
    student.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    student.train()
    return 100 * correct / total


# ========== 6. 训练学生 ==========
optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

best_test_acc = 0

print("开始蒸馏训练...")
for epoch in range(10):
    student.train()
    running_hard_loss = 0.0
    running_soft_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher(images)

        student_logits = student(images)

        loss, hard_loss, soft_loss = distillation_loss(
            student_logits, teacher_logits, labels,
            temperature=4.0, alpha=0.7
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_hard_loss += hard_loss.item()
        running_soft_loss += soft_loss.item()

    # 每个 epoch 结束后计算准确率
    train_acc = compute_accuracy(trainloader)
    test_acc = compute_accuracy(testloader)

    print(f"Epoch {epoch + 1}/10")
    print(f"  Hard Loss: {running_hard_loss / len(trainloader):.4f}")
    print(f"  Soft Loss: {running_soft_loss / len(trainloader):.4f}")
    print(f"  Train Acc: {train_acc:.2f}%")
    print(f"  Test Acc:  {test_acc:.2f}%")

    # 🎯 保存准确率最高的模型
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(student.state_dict(), 'lou_model/best_deit_distilled_cifar10.pth')
        print(f"  ✅ 保存最佳模型！测试准确率: {best_test_acc:.2f}%")
    print()

print(f"\n蒸馏训练完成！")
print(f"最佳测试准确率: {best_test_acc:.2f}%")
print(f"模型已保存为: best_deit_distilled_cifar10.pth")
