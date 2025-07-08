# DBPs (消毒副产物) 预测系统

这是一个基于机器学习的消毒副产物(DBPs)预测系统，使用Flask框架开发的Web应用程序。该系统能够预测水处理过程中形成的消毒副产物的浓度。

## 功能特点

- 基于机器学习模型的DBPs浓度预测
- 响应式Web界面设计
- 实时预测结果显示
- 特征重要性可视化
- 数据验证和错误处理

## 技术栈

- Python 3.8+
- Flask
- scikit-learn
- pandas
- numpy
- Bootstrap 5
- Vercel (部署平台)

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/Halsey-ux/dbps_prediction_test1.git
cd dbps_prediction_test1
```

2. 创建并激活虚拟环境：
```bash
conda env create -f environment.yml
conda activate dbps_env
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 本地运行

1. 启动Flask应用：
```bash
python app.py
```

2. 在浏览器中访问：
```
http://localhost:5000
```

## 在线访问

您可以通过以下链接访问已部署的应用：
[DBPs预测系统](https://dbps-prediction-test1.vercel.app)

## 项目结构

```
├── app.py                 # Flask应用主文件
├── requirements.txt       # Python依赖
├── model.pkl             # 训练好的机器学习模型
├── scaler.pkl            # 数据标准化模型
├── feature_importance.png # 特征重要性可视化
├── train_model.py        # 模型训练脚本
├── static/              # 静态资源文件
├── templates/           # HTML模板
└── vercel.json         # Vercel部署配置
```

## 使用说明

1. 在网页界面输入所需的水质参数
2. 点击"预测"按钮
3. 系统将显示预测的DBPs浓度结果
4. 可以查看特征重要性图表，了解各参数的影响程度

## 开发者

- GitHub: [Halsey-ux](https://github.com/Halsey-ux)
- Email: 2489762201@qq.com

## 许可证

MIT License 