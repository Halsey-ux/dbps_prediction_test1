# DBPs预测模型

这是一个基于机器学习的消毒副产物(DBPs)预测Web应用程序。该应用程序使用随机森林回归模型来预测给定水质参数下的DBPs浓度。

## 在线访问

公开访问地址：https://dbps-prediction.streamlit.app

## 本地部署

### 安装要求

1. Python 3.8+
2. 依赖包安装：
```bash
pip install -r requirements.txt
```

### 使用说明

1. 首先训练模型：
```bash
python train_model.py
```

2. 启动Web应用：
```bash
streamlit run app.py
```

3. 在浏览器中访问应用：
   - 本地访问：http://localhost:8501
   - 局域网访问：http://192.168.1.116:8501
   - 公网访问：http://207.174.7.167:8501

## Streamlit Cloud部署

1. Fork这个项目到你的GitHub账号
2. 访问 https://share.streamlit.io/
3. 使用GitHub账号登录
4. 点击 "New app" 并选择你fork的仓库
5. 选择main分支和app.py文件
6. 点击 "Deploy"

部署完成后，你会得到一个类似 https://your-app-name.streamlit.app 的永久访问地址。

## 输入参数说明

- pH值：6.0-9.0
- 温度：15.0-35.0 °C
- 氯投加量：0.0-10.0 mg/L
- 溶解性有机碳：0.0-20.0 mg/L
- 溴离子浓度：0.0-1000.0 μg/L
- 接触时间：0.0-168.0 h

## 模型说明

- 模型类型：随机森林回归
- 特征工程：标准化处理
- 评估指标：MSE、R²、交叉验证得分

## 项目结构

```
dbps-prediction/
├── app.py              # Streamlit Web应用
├── train_model.py      # 模型训练脚本
├── requirements.txt    # 项目依赖
├── model.pkl          # 训练好的模型
├── scaler.pkl        # 特征标准化器
└── feature_importance.png  # 特征重要性图
```

## 注意事项

- 请确保输入参数在合理范围内
- 模型预测结果仅供参考
- 建议定期使用新数据更新模型

## 部署说明

### 本地部署
适用于个人使用或开发测试。

### Streamlit Cloud部署（推荐）
- 免费托管
- 自动更新
- 永久域名
- SSL加密
- 无需服务器维护

### 自托管服务器部署
如果需要自定义域名或特殊配置，可以在自己的服务器上部署。

## 联系方式

如有问题或建议，请联系：
- 邮箱：your.email@example.com
- 项目主页：https://github.com/your-username/dbps-prediction
- 在线演示：https://dbps-prediction.streamlit.app

## 许可证

MIT License 