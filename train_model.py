import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，避免显示相关错误
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_sample_data(n_samples=1000):
    """生成示例数据"""
    try:
        np.random.seed(42)
        data = {
            'pH': np.random.uniform(6.0, 9.0, n_samples),
            'Temperature': np.random.uniform(15.0, 35.0, n_samples),
            'Cl2_dose': np.random.uniform(0.0, 10.0, n_samples),
            'DOC': np.random.uniform(0.0, 20.0, n_samples),
            'Bromide': np.random.uniform(0.0, 1000.0, n_samples),
            'Contact_time': np.random.uniform(0.0, 168.0, n_samples)
        }
        
        # 模拟DBPs生成的简单关系
        df = pd.DataFrame(data)
        df['DBPs'] = (0.5 * df['Cl2_dose'] + 
                     0.3 * df['DOC'] + 
                     0.2 * df['Temperature'] + 
                     0.1 * df['pH'] + 
                     0.1 * df['Bromide'] / 100 + 
                     0.2 * np.sqrt(df['Contact_time']) +
                     np.random.normal(0, 0.1, n_samples))
        return df
    except Exception as e:
        print(f"生成数据时出错: {str(e)}")
        return None

def save_model_and_scaler(model, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
    """保存模型和标准化器"""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"模型和标准化器已保存到 {model_path} 和 {scaler_path}")
        return True
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
        return False

def create_feature_importance_plot(model, feature_names, save_path='feature_importance.png'):
    """创建并保存特征重要性图"""
    try:
        plt.figure(figsize=(10, 6))
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('特征重要性分析')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"特征重要性图已保存到 {save_path}")
        return True
    except Exception as e:
        print(f"创建特征重要性图时出错: {str(e)}")
        return False

def train_model():
    """训练模型的主函数"""
    try:
        print("开始训练模型...")
        
        # 生成或加载数据
        df = generate_sample_data()
        if df is None:
            return False
        
        # 分离特征和目标变量
        X = df.drop('DBPs', axis=1)
        y = df['DBPs']
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # 使用所有可用的CPU核心
        )
        model.fit(X_train_scaled, y_train)
        
        # 模型评估
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n模型评估结果:")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"决定系数 (R²): {r2:.4f}")
        
        # 交叉验证
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=5, scoring='r2'
        )
        print(f"交叉验证得分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 保存模型和标准化器
        if not save_model_and_scaler(model, scaler):
            return False
        
        # 创建特征重要性图
        if not create_feature_importance_plot(model, X.columns):
            return False
        
        print("\n模型训练和评估完成！")
        return True
        
    except Exception as e:
        print(f"训练模型时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 确保输出目录存在
    ensure_dir('output')
    
    # 训练模型
    success = train_model()
    
    if success:
        print("\n所有任务已完成！您现在可以运行 streamlit run app.py 来启动预测应用。")
    else:
        print("\n训练过程中出现错误，请检查上述错误信息。") 