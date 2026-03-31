import torch
import torchvision
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from net import MyModel
from torch.utils.data import DataLoader

writer=SummaryWriter(log_dir='logs')

transform=transforms.Compose([#数据预处理
    transforms.RandomHorizontalFlip(),#垂直翻转
    transforms.RandomCrop(32, 4),#填充4个像素
    transforms.ToTensor(),#转Tensor
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#标准化处理((图像通道均值),(每个通道的标准差))
])

train_data_set = torchvision.datasets.CIFAR10(
    root='C:/Users/翛/Desktop/XIAO/Python_project/CIFAR10/CIFAR10',
    train=True,
    download=False,  # 不下载
    transform=transform
)

test_data_set = torchvision.datasets.CIFAR10(
    root='C:/Users/翛/Desktop/XIAO/Python_project/CIFAR10/CIFAR10',
    train=False,
    download=False,
    transform=transform
)

#数据集的大小
train_data_size=len(train_data_set)
test_data_size=len(test_data_set)

#加载数据集
train_data_loader=DataLoader(train_data_set,batch_size=64,shuffle=True)#shuffle=True:进行打乱
test_data_loader=DataLoader(test_data_set,batch_size=64,shuffle=True)

#定义网络
myModel=MyModel()

#判断是否使用GPU
use_gpu=torch.cuda.is_available()
if use_gpu:
    print("use gpu")
    myModel=myModel.cuda()
#训练轮数
epochs=300
#损失函数
lossFn=torch.nn.CrossEntropyLoss()#交叉熵：处理分类问题
#优化器（梯度下降）
optimizer=SGD(myModel.parameters(),lr=0.01)#学习速率
for epoch in range(epochs):
    print(f"epoch:{epoch}/{epochs}")
    #损失变量
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

    print(f"train loss:{train_total_loss},acc:{train_total_acc/train_data_size},test loss:{test_total_loss},test acc:{test_total_acc/test_data_size}")

    #每轮训练完添加到日志文件
    writer.add_scalar('loss/train',train_total_loss,epoch+1)
    writer.add_scalar('acc/train',train_total_acc/train_data_size,epoch+1)
    writer.add_scalar('loss/test',test_total_loss,epoch+1)
    writer.add_scalar('acc/test',test_total_acc/test_data_size,epoch+1)

    if((epoch+1)%50==0):
        # 训练完保存模型(每50轮)
        torch.save(myModel, f"model/model_{epoch+1}.pth")