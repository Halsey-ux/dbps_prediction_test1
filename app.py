try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    import pandas as pd
    import numpy as np
    import pickle
    import os
    import logging
    import sys
    from datetime import datetime

    # 配置日志
    log_file = 'output.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # 记录启动信息
    logger.info(f"应用启动时间: {datetime.now()}")
    logger.info(f"当前工作目录: {os.getcwd()}")
    logger.info(f"Python版本: {sys.version}")

    app = Flask(__name__)
    app.config['DEBUG'] = False  # 生产环境禁用调试模式
    app.config['PROPAGATE_EXCEPTIONS'] = True  # 确保异常被正确传播

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
        try:
            logger.info("访问主页")
            return render_template('index.html')
        except Exception as e:
            logger.error(f"渲染主页时出错: {str(e)}")
            return jsonify({'error': '服务器内部错误'}), 500

    @app.route('/static/<path:path>')
    def send_static(path):
        try:
            logger.info(f"请求静态文件: {path}")
            return send_from_directory('static', path)
        except Exception as e:
            logger.error(f"访问静态文件时出错: {str(e)}")
            return jsonify({'error': '文件不存在'}), 404

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

    @app.route('/health')
    def health_check():
        """健康检查端点"""
        try:
            status = {
                'status': 'healthy',
                'model_loaded': model is not None,
                'scaler_loaded': scaler is not None,
                'timestamp': datetime.now().isoformat(),
                'python_version': sys.version,
                'working_directory': os.getcwd()
            }
            logger.info(f"健康检查: {status}")
            return jsonify(status)
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return jsonify({'error': '健康检查失败'}), 500

    # 添加错误处理器
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({'error': '页面不存在'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': '服务器内部错误'}), 500

    # Vercel需要这个应用实例
    application = app

except Exception as e:
    print(f"应用初始化错误: {str(e)}")
    raise

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"启动服务器，端口: {port}")
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        raise 