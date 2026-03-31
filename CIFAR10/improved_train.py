import torch
import torchvision
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from net import ImprovedResNet18
from torch.utils.data import DataLoader

writer=SummaryWriter(log_dir='logs')

transform_train=transforms.Compose([#数据预处理
    transforms.RandomHorizontalFlip(),#垂直翻转
    transforms.RandomCrop(32, 4),#填充4个像素
    transforms.ToTensor(),#转Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))#标准化处理((图像通道均值),(每个通道的标准差))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_data_set = torchvision.datasets.CIFAR10(
    root='C:/Users/翛/Desktop/XIAO/Python_project/CIFAR10/CIFAR10',
    train=True,
    download=False,  # 不下载
    transform=transform_train
)

test_data_set = torchvision.datasets.CIFAR10(
    root='C:/Users/翛/Desktop/XIAO/Python_project/CIFAR10/CIFAR10',
    train=False,
    download=False,
    transform=transform_test
)

#数据集的大小
train_data_size=len(train_data_set)
test_data_size=len(test_data_set)

#加载数据集
train_data_loader=DataLoader(train_data_set,batch_size=128,shuffle=True)#shuffle=True:进行打乱
test_data_loader=DataLoader(test_data_set,batch_size=128,shuffle=True)

#定义网络
myModel=ImprovedResNet18()

#判断是否使用GPU
use_gpu=torch.cuda.is_available()
if use_gpu:
    print("use gpu")
    myModel=myModel.cuda()
#训练轮数
epochs=150
#损失函数
lossFn=torch.nn.CrossEntropyLoss()#交叉熵：处理分类问题
#优化器(改进学习率，用带动量的SGD+权重衰减)
optimizer=SGD(myModel.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)#学习速率
#添加学习率调节器（余弦退火）
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

for epoch in range(epochs):
    print(f"epoch:{epoch}/{epochs}")
    #损失变量
    myModel.train()
    train_total_loss=0
    test_total_loss=0
    #准确率
    train_total_acc=0
    test_total_acc=0
    #开始训练
    for data in train_data_loader:
        inputs,labels=data
        if use_gpu:
            inputs=inputs.cuda()
            labels=labels.cuda()

        optimizer.zero_grad()#每一批训练数据梯度要清空，不能延续上一次计算得到的梯度

        outputs=myModel(inputs)
        #计算实际输出与真实类别之间的差距（损失）
        loss=lossFn(outputs,labels)
        #反向传播给网络
        loss.backward()
        #更新每层网络的参数
        optimizer.step()

        #计算精度
        _,index=torch.max(outputs,1)#得到预测值最大的值和下标
        acc=torch.sum(index==labels).item()

        train_total_loss+=loss.item()#tensor数据用item取到里面的值
        train_total_acc+=acc

    #测试：不需要计算梯度
    with torch.no_grad():
        for data in test_data_loader:
            inputs,labels=data
            if use_gpu:
                inputs=inputs.cuda()
                labels=labels.cuda()
            outputs=myModel(inputs)
            # 计算实际输出与真实类别之间的差距（损失）
            loss = lossFn(outputs, labels)

            # 计算精度
            _,index = torch.max(outputs, 1)  # 得到预测值最大的值和下标
            acc = torch.sum(index == labels).item()

            test_total_loss += loss.item()  # tensor数据用item取到里面的值
            test_total_acc += acc

    scheduler.step()

    # 打印结果
    train_acc_percent = 100. * train_total_acc / train_data_size
    test_acc_percent = 100. * test_total_acc / test_data_size
    print(f"train loss:{train_total_loss:.4f}, train acc:{train_acc_percent:.2f}%")
    print(f"test loss:{test_total_loss:.4f}, test acc:{test_acc_percent:.2f}%")

    #每轮训练完添加到日志文件
    writer.add_scalar('loss/train',train_total_loss,epoch+1)
    writer.add_scalar('acc/train',train_acc_percent,epoch+1)
    writer.add_scalar('loss/test',test_total_loss,epoch+1)
    writer.add_scalar('acc/test',test_acc_percent,epoch+1)

    if((epoch+1)%50==0):
        # 训练完保存模型(每50轮)
        torch.save(myModel.state_dict(), f"model/improved_model_{epoch+1}.pth")

writer.close()