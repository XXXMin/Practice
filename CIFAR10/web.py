import streamlit as st
import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from net import ImprovedResNet18

# 设置页面配置
st.set_page_config(
    page_title="CIFAR-10 图像分类器",
    page_icon="🤖",
    layout="centered"
)

# 定义类别
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# 模型加载（使用缓存避免重复加载）
@st.cache_resource
def load_model():
    """加载训练好的模型"""
    use_gpu = torch.cuda.is_available()

    # 初始化模型
    model = ImprovedResNet18()

    # 加载模型权重
    model_path = "C:/Users/翛/Desktop/XIAO/Python_project/CIFAR10/model/improved_model_150.pth"
    state_dict = torch.load(
        model_path,
        map_location=torch.device('cuda' if use_gpu else 'cpu'),
        weights_only=False
    )

    model.load_state_dict(state_dict)
    model.eval()  # 切换到评估模式

    # 移动到GPU（如果可用）
    if use_gpu:
        model = model.cuda()
        st.sidebar.info("✅ 使用 GPU 进行推理")
    else:
        st.sidebar.info("ℹ️ 使用 CPU 进行推理")

    return model, use_gpu


# 数据预处理
def get_transform():
    """定义图像预处理转换"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整大小到32x32
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    return transform


# 预测函数
def predict_image(model, image, use_gpu, transform):
    """对上传的图片进行预测"""
    # 确保图片是RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 应用预处理
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度

    # 移动到GPU（如果可用）
    if use_gpu:
        image_tensor = image_tensor.cuda()

    # 进行预测
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        true_prob, index = torch.max(probabilities, 1)

        # 获取所有类别的概率
        all_probs = probabilities.cpu().numpy()[0]

    return classes[index.item()], true_prob.item(), all_probs


# 主程序
def main():
    # 标题和说明
    st.title("🎨 CIFAR-10 图像分类器")
    st.markdown("""
    上传一张图片，模型将预测它属于以下10个类别中的哪一个：
    - ✈️ airplane (飞机)
    - 🚗 automobile (汽车)  
    - 🐦 bird (鸟)
    - 🐱 cat (猫)
    - 🦌 deer (鹿)
    - 🐶 dog (狗)
    - 🐸 frog (青蛙)
    - 🐴 horse (马)
    - 🚢 ship (船)
    - 🚚 truck (卡车)
    """)

    st.markdown("---")

    # 加载模型
    with st.spinner("正在加载模型，请稍候..."):
        try:
            model, use_gpu = load_model()
            transform = get_transform()
            st.success("✅ 模型加载成功！")
        except Exception as e:
            st.error(f"❌ 模型加载失败: {str(e)}")
            st.stop()

    # 文件上传组件
    uploaded_file = st.file_uploader(
        "📤 请选择一张图片上传",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="支持 JPG, PNG, BMP, TIFF 格式"
    )

    # 当有文件上传时
    if uploaded_file is not None:
        # 显示上传的图片
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📷 上传的图片")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        # 预测按钮
        if st.button("🔍 开始预测", type="primary"):
            with st.spinner("正在分析图片..."):
                try:
                    # 进行预测
                    predicted_class, confidence, all_probs = predict_image(
                        model, image, use_gpu, transform
                    )

                    # 显示预测结果
                    with col2:
                        st.subheader("🎯 预测结果")

                        # 显示主要预测结果
                        st.markdown(f"""
                        ### **{predicted_class}**
                        """)

                        # 显示置信度
                        st.metric(
                            label="置信度",
                            value=f"{confidence:.2%}",
                            delta=None
                        )

                        # 根据置信度显示不同颜色
                        if confidence > 0.8:
                            st.success("✅ 高置信度预测")
                        elif confidence > 0.6:
                            st.warning("⚠️ 中等置信度预测")
                        else:
                            st.error("❌ 低置信度预测")

                    # 显示所有类别的概率分布
                    st.markdown("---")
                    st.subheader("📊 各类别概率分布")

                    # 创建概率条形图
                    chart_data = {classes[i]: all_probs[i] for i in range(len(classes))}

                    # 使用st.bar_chart显示条形图
                    import pandas as pd
                    df = pd.DataFrame([chart_data])
                    st.bar_chart(df.T)

                    # 详细概率表格
                    with st.expander("查看详细概率"):
                        for i, (cls, prob) in enumerate(chart_data.items()):
                            st.write(f"{cls}: {prob:.2%}")

                except Exception as e:
                    st.error(f"预测过程中出现错误: {str(e)}")

    else:
        # 未上传文件时的提示
        st.info("👆 请在上侧上传图片开始预测")

    # 侧边栏信息
    with st.sidebar:
        st.markdown("## 📋 使用说明")
        st.markdown("""
        1. 点击"Browse files"上传图片
        2. 等待图片预览显示
        3. 点击"开始预测"按钮
        4. 查看预测结果和置信度
        """)

        st.markdown("## ⚙️ 模型信息")
        st.markdown(f"""
        - **模型类型**: improved_model_150(resnet18)
        - **数据集**: CIFAR-10
        - **类别数量**: 10
        - **输入尺寸**: 32x32
        - **推理设备**: {'GPU' if use_gpu else 'CPU'}
        """)

        st.markdown("## 💡 提示")
        st.markdown("""
        - 图片会自动缩放到32x32像素
        - 建议上传清晰、主体明确的图片
        - 模型在CIFAR-10数据集上训练，主要识别小型物体
        """)


if __name__ == "__main__":
    main()