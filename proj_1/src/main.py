import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from feature_extractor import NGramExtractor
from model import SentimentClassifier, SGD, MBGD, BGD, train_model, evaluate_model
import os

def main():
    # 设置随机种子，确保结果可重复
    np.random.seed(42)
    
    # 显示当前工作目录和文件路径
    print(f"当前工作目录: {os.getcwd()}")
    train_path = '../data/train.tsv'
    test_path = '../data/test.tsv'
    print(f"训练数据路径: {train_path}, 存在: {os.path.exists(train_path)}")
    print(f"测试数据路径: {test_path}, 存在: {os.path.exists(test_path)}")
    
    # 加载数据
    print("加载数据...")
    data_loader = DataLoader()
    data_loader.load_data(train_path=train_path, test_path=test_path)
    data_loader.preprocess_data()
    train_texts, train_labels, val_texts, val_labels = data_loader.split_train_val(val_ratio=0.2)
    print(f"训练数据: {len(train_texts)}个样本, 验证数据: {len(val_texts)}个样本")
    
    # 特征提取
    print("提取特征...")
    feature_extractor = NGramExtractor(dimension=2)  # 使用1-gram和2-gram特征
    train_features = feature_extractor.fit_transform(train_texts)
    val_features = feature_extractor.transform(val_texts)
    print(f"特征维度: {feature_extractor.vocab_size}")
    
    # 配置学习率列表
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    # 记录每种优化器在不同学习率下的训练准确率
    sgd_accuracies = []
    mbgd_accuracies = []
    bgd_accuracies = []
    
    # 对每个学习率进行实验
    for lr in learning_rates:
        print(f"\n学习率: {lr}")
        
        # SGD优化器
        print("使用SGD优化器训练...")
        sgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=100, output_dim=5)
        sgd_optimizer = SGD(learning_rate=lr)
        sgd_history = train_model(sgd_model, sgd_optimizer, train_features, train_labels, val_features, val_labels, epochs=5)
        sgd_train_accuracy = evaluate_model(sgd_model, train_features, train_labels)
        sgd_accuracies.append(sgd_train_accuracy)
        print(f"SGD训练准确率: {sgd_train_accuracy:.4f}")
        
        # MBGD优化器
        print("使用MBGD优化器训练...")
        mbgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=100, output_dim=5)
        mbgd_optimizer = MBGD(learning_rate=lr, batch_size=32)
        mbgd_history = train_model(mbgd_model, mbgd_optimizer, train_features, train_labels, val_features, val_labels, epochs=5)
        mbgd_train_accuracy = evaluate_model(mbgd_model, train_features, train_labels)
        mbgd_accuracies.append(mbgd_train_accuracy)
        print(f"MBGD训练准确率: {mbgd_train_accuracy:.4f}")
        
        # BGD优化器
        print("使用BGD优化器训练...")
        bgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=100, output_dim=5)
        bgd_optimizer = BGD(learning_rate=lr)
        bgd_history = train_model(bgd_model, bgd_optimizer, train_features, train_labels, val_features, val_labels, epochs=5)
        bgd_train_accuracy = evaluate_model(bgd_model, train_features, train_labels)
        bgd_accuracies.append(bgd_train_accuracy)
        print(f"BGD训练准确率: {bgd_train_accuracy:.4f}")
    
    # 输出结果数据
    print("\n学习率与训练准确率数据:")
    for i, lr in enumerate(learning_rates):
        print(f"学习率: {lr}, SGD: {sgd_accuracies[i]:.4f}, MBGD: {mbgd_accuracies[i]:.4f}, BGD: {bgd_accuracies[i]:.4f}")
    
    # 绘制学习率与训练准确率的关系图
    print("\n绘制图表...")
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, sgd_accuracies, 'o-', label='SGD')
    plt.plot(learning_rates, mbgd_accuracies, 's-', label='MBGD')
    plt.plot(learning_rates, bgd_accuracies, '^-', label='BGD')
    plt.xscale('log')  # 使用对数刻度，便于观察不同量级的学习率
    plt.xlabel('学习率')
    plt.ylabel('训练准确率')
    plt.title('不同优化器在各学习率下的训练准确率')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    save_path = 'optimizer_comparison.png'
    plt.savefig(save_path)
    full_save_path = os.path.abspath(save_path)
    print(f"图片已保存至: {full_save_path}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"显示图片时出现错误: {e}")
    
    print("实验完成")

if __name__ == "__main__":
    main() 