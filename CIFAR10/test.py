import os
import cv2
import torch
from torchvision.transforms import transforms
from PIL import Image
from net import ImprovedResNet18

classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

use_gpu=torch.cuda.is_available()
model = ImprovedResNet18()
state_dict=torch.load(
    "C:/Users/翛/Desktop/XIAO/Python_project/CIFAR10/model/improved_model_150.pth",
    map_location=torch.device('cuda' if use_gpu else 'cpu'),
    weights_only=False
)

model.load_state_dict(state_dict)#后面改了这个用state_dict的方式保存才需要用的
model.eval()#切换到评估模式

#移动到GPU/CPU
if use_gpu:
    model=model.cuda()

transform=transforms.Compose([#数据预处理（转tensor）
    transforms.Resize((32, 32)),
    transforms.ToTensor(),#转Tensor
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#标准化处理((图像通道均值),(每个通道的标准差))
])

#指定文件夹
folder_path='C:/Users/翛/Desktop/XIAO/Python_project/CIFAR10/testimages'
files=os.listdir(folder_path)

#得到每个文件的地址
images_files=[os.path.join(folder_path,f) for f in files]

with torch.no_grad():
    for img in images_files:
        print(img)
        image=Image.open(img).convert('RGB')#转成RGB图像
        # image.show()
        image = transform(image)
        image=image.unsqueeze(0)
        image=image.to('cuda' if use_gpu else 'cpu')

        output=model(image)

        #计算输出真实类别
        probabilities = torch.softmax(output, dim=1)  # 转换为概率，范围 [0, 1]
        true_prob, index = torch.max(probabilities, 1)
        pre_val=classes[index]

        print(f"预测概率：{true_prob.item()}，预测下标：{index.item()}，预测结果：{pre_val}")

        # #等该用户按键，才执行
        # cv2.waitKey(0)