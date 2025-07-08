import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import os

# 配置页面
st.set_page_config(
    page_title="DBPs预测模型",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Halsey-ux/dbps_prediction_test1/issues',
        'Report a bug': 'https://github.com/Halsey-ux/dbps_prediction_test1/issues',
        'About': '# DBPs预测模型\n 此应用使用机器学习预测饮用水中的消毒副产物(DBPs)含量。'
    }
)

# 设置缓存目录
if not os.path.exists('.cache'):
    os.makedirs('.cache')

# 添加页面标题和样式
st.title("消毒副产物(DBPs)预测模型")
st.markdown("""
<style>
.main {
    padding: 20px;
}
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
}
.stButton>button:hover {
    background-color: #45a049;
}
.reportview-container {
    background: #fafafa;
}
.sidebar .sidebar-content {
    background: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# 添加访问统计
if 'visitor_count' not in st.session_state:
    st.session_state.visitor_count = 0
st.session_state.visitor_count += 1

st.markdown("""
### 欢迎使用DBPs预测系统
此应用用于预测饮用水中的消毒副产物(DBPs)含量。
* **开发者:** 化学机器学习实验室
* **数据来源:** 实验室测试数据
* **模型类型:** 随机森林回归
* **访问次数:** {}
""".format(st.session_state.visitor_count))

# 加载模型和标准化器
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None, None

# 侧边栏配置
with st.sidebar:
    st.header("输入参数")
    st.info("请调整以下参数进行预测")
    
    # 添加输入参数
    ph = st.slider("pH值", 6.0, 9.0, 7.0, 0.1, 
                  help="pH值范围：6.0-9.0")
    temperature = st.slider("温度 (°C)", 15.0, 35.0, 25.0, 0.5,
                          help="水温范围：15-35°C")
    cl2_dose = st.slider("氯投加量 (mg/L)", 0.0, 10.0, 5.0, 0.1,
                       help="氯投加量范围：0-10 mg/L")
    doc = st.slider("溶解性有机碳 (mg/L)", 0.0, 20.0, 5.0, 0.1,
                   help="DOC范围：0-20 mg/L")
    bromide = st.slider("溴离子浓度 (μg/L)", 0.0, 1000.0, 100.0, 10.0,
                      help="溴离子浓度范围：0-1000 μg/L")
    contact_time = st.slider("接触时间 (h)", 0.0, 168.0, 24.0, 1.0,
                           help="接触时间范围：0-168小时")

# 创建特征数据框
@st.cache_data
def create_features(ph, temperature, cl2_dose, doc, bromide, contact_time):
    return pd.DataFrame({
        'pH': [ph],
        'Temperature': [temperature],
        'Cl2_dose': [cl2_dose],
        'DOC': [doc],
        'Bromide': [bromide],
        'Contact_time': [contact_time]
    })

# 主要内容区域
col1, col2 = st.columns([2, 1])

with col1:
    # 预测按钮
    if st.button('进行预测', key='predict'):
        try:
            with st.spinner('正在计算中...'):
                model, scaler = load_model()
                if model is not None and scaler is not None:
                    features = create_features(ph, temperature, cl2_dose, doc, bromide, contact_time)
                    
                    # 显示输入参数
                    st.write("#### 输入参数:")
                    st.write(features)
                    
                    # 数据标准化和预测
                    features_scaled = scaler.transform(features)
                    prediction = model.predict(features_scaled)[0]
                    
                    # 显示预测结果
                    st.write("#### 预测的DBPs浓度:")
                    st.success(f"{prediction:.2f} μg/L")
                    
                    # 添加置信区间
                    predictions = []
                    for estimator in model.estimators_:
                        predictions.append(estimator.predict(features_scaled)[0])
                    
                    confidence_interval = np.percentile(predictions, [2.5, 97.5])
                    st.write(f"95%置信区间: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}] μg/L")
        except Exception as e:
            st.error(f"预测过程中出现错误: {str(e)}")

with col2:
    # 添加特征重要性图
    try:
        st.markdown("### 特征重要性分析")
        image = Image.open('feature_importance.png')
        st.image(image, caption='模型特征重要性分析')
    except Exception as e:
        st.warning("特征重要性图加载失败，请稍后再试。")

# 添加说明文档
with st.expander("使用说明", expanded=False):
    st.markdown("""
    ### 如何使用本系统
    1. 在左侧边栏调整输入参数
    2. 点击"进行预测"按钮获取预测结果
    3. 查看预测结果和数据可视化

    ### 参数说明
    - **pH值**: 水样的酸碱度
    - **温度**: 水样温度
    - **氯投加量**: 消毒剂投加量
    - **DOC**: 溶解性有机碳含量
    - **溴离子**: 水中溴离子浓度
    - **接触时间**: 消毒剂与水样接触时间

    ### 注意事项
    - 请确保输入参数在合理范围内
    - 模型预测结果仅供参考
    - 如有问题请联系技术支持
    """)

# 添加页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>© 2024 化学机器学习实验室. All rights reserved.</p>
    <p>联系邮箱：2489762201@qq.com</p>
</div>
""", unsafe_allow_html=True) 