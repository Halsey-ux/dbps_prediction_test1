from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 加载模型和标准化器
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
        
        logger.info(f"尝试加载模型文件: {model_path}")
        logger.info(f"尝试加载标准化器文件: {scaler_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            logger.info("模型加载成功")
            
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            logger.info("标准化器加载成功")
            
        return model, scaler
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        return None, None

model, scaler = load_model()

@app.route('/')
def home():
    logger.info("访问主页")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            raise ValueError("模型或标准化器未正确加载")
            
        data = request.get_json()
        logger.info(f"收到预测请求数据: {data}")
        
        if not data:
            raise ValueError("没有接收到数据")
            
        required_fields = ['ph', 'temperature', 'cl2_dose', 'doc', 'bromide', 'contact_time']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"缺少必要的字段: {field}")
        
        features = pd.DataFrame({
            'pH': [float(data['ph'])],
            'Temperature': [float(data['temperature'])],
            'Cl2_dose': [float(data['cl2_dose'])],
            'DOC': [float(data['doc'])],
            'Bromide': [float(data['bromide'])],
            'Contact_time': [float(data['contact_time'])]
        })
        
        logger.info("开始进行预测")
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        logger.info(f"预测完成: {prediction}")
        
        return jsonify({'prediction': float(prediction)})
    except ValueError as e:
        logger.error(f"输入数据错误: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        return jsonify({'error': '预测过程中发生错误'}), 500

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(host='127.0.0.1', port=5000, debug=True) 