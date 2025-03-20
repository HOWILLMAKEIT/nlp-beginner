import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import DataLoader
from src.feature_extractor import NGramExtractor
from src.model import SentimentClassifier, SGD, MBGD, BGD, train_model, evaluate_model
from matplotlib.font_manager import FontProperties 
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  

# 用于观察大概多少epoch收敛
def experinment0():
    np.random.seed(20)
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_data()
    train_texts, train_labels, val_texts, val_labels = data_loader.split_train_val(val_ratio=0.2)
    print(f"训练数据: {len(train_texts)}个样本, 验证数据: {len(val_texts)}个样本")

    # 设置最小词频和最大特征数，减少内存占用
    feature_extractor = NGramExtractor(dimension=2, min_freq=10, max_features=5000)  
    train_features = feature_extractor.fit_transform(train_texts)
    val_features = feature_extractor.transform(val_texts)
    print(f"特征维度: {feature_extractor.vocab_size}")
    
    # MBGD优化器
    print("训练单个MBGD模型测试...")
    mbgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=None, output_dim=5)
    mbgd_optimizer = MBGD(learning_rate=0.1, batch_size=32)
    mbgd_history = train_model(mbgd_model, mbgd_optimizer, train_features, train_labels, val_features, val_labels, epochs=50, compute_history=True)
    mbgd_train_accuracy = evaluate_model(mbgd_model, train_features, train_labels)
    mbgd_val_accuracy = evaluate_model(mbgd_model, val_features, val_labels)
    print(f"MBGD训练准确率: {mbgd_train_accuracy:.4f}, 验证准确率: {mbgd_val_accuracy:.4f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(mbgd_history['train_loss'], 'r-', linewidth=2, label='训练损失')
    plt.plot(mbgd_history['val_loss'], 'b-', linewidth=2, label='验证损失')
    plt.xlabel('训练轮数', fontproperties=font)
    plt.ylabel('损失', fontproperties=font)
    plt.title('MBGD优化器训练过程中的损失变化', fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True)
    plt.savefig('experiment0.png')
    plt.show()

def experiment1():
    np.random.seed(20)
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_data()
    train_texts, train_labels, val_texts, val_labels = data_loader.split_train_val(val_ratio=0.2)
    print(f"训练数据: {len(train_texts)}个样本, 验证数据: {len(val_texts)}个样本")

    # 设置最小词频和最大特征数，减少内存占用
    feature_extractor = NGramExtractor(dimension=2, min_freq=10, max_features=5000)  
    train_features = feature_extractor.fit_transform(train_texts)
    val_features = feature_extractor.transform(val_texts)
    print(f"特征维度: {feature_extractor.vocab_size}")
    

    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0,10.0,100.0,1000.0]
    
    # 记录每种优化器在不同学习率下的训练准确率和验证准确率
    sgd_train_accuracies = []
    sgd_val_accuracies = []
    mbgd_train_accuracies = []
    mbgd_val_accuracies = []
    bgd_train_accuracies = []
    bgd_val_accuracies = []
    
    # 对每个学习率进行实验
    for lr in learning_rates:
        print(f"\n学习率: {lr}")
        
        # SGD优化器
        print(f"使用SGD优化器训练，学习率: {lr}...")
        sgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=None, output_dim=5)
        sgd_optimizer = SGD(learning_rate=lr)
        sgd_history = train_model(sgd_model, sgd_optimizer, train_features, train_labels, val_features, val_labels, epochs=30, compute_history=False)
        sgd_train_accuracy = evaluate_model(sgd_model, train_features, train_labels)
        sgd_val_accuracy = evaluate_model(sgd_model, val_features, val_labels)
        sgd_train_accuracies.append(sgd_train_accuracy)
        sgd_val_accuracies.append(sgd_val_accuracy)
        print(f"SGD训练准确率: {sgd_train_accuracy:.4f}, 验证准确率: {sgd_val_accuracy:.4f}")
        
        # MBGD优化器
        print(f"使用MBGD优化器训练，学习率: {lr}...")
        mbgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=None, output_dim=5)
        mbgd_optimizer = MBGD(learning_rate=lr, batch_size=32)
        mbgd_history = train_model(mbgd_model, mbgd_optimizer, train_features, train_labels, val_features, val_labels, epochs=30, compute_history=False)
        mbgd_train_accuracy = evaluate_model(mbgd_model, train_features, train_labels)
        mbgd_val_accuracy = evaluate_model(mbgd_model, val_features, val_labels)
        mbgd_train_accuracies.append(mbgd_train_accuracy)
        mbgd_val_accuracies.append(mbgd_val_accuracy)
        print(f"MBGD训练准确率: {mbgd_train_accuracy:.4f}, 验证准确率: {mbgd_val_accuracy:.4f}")
        
        # BGD优化器
        print(f"使用BGD优化器训练，学习率: {lr}...")
        bgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=None, output_dim=5)
        bgd_optimizer = BGD(learning_rate=lr)
        bgd_history = train_model(bgd_model, bgd_optimizer, train_features, train_labels, val_features, val_labels, epochs=30, compute_history=False)
        bgd_train_accuracy = evaluate_model(bgd_model, train_features, train_labels)
        bgd_val_accuracy = evaluate_model(bgd_model, val_features, val_labels)
        bgd_train_accuracies.append(bgd_train_accuracy)
        bgd_val_accuracies.append(bgd_val_accuracy)
        print(f"BGD训练准确率: {bgd_train_accuracy:.4f}, 验证准确率: {bgd_val_accuracy:.4f}")
    
    # 输出结果数据
    print("\n学习率与训练准确率数据:")
    for i, lr in enumerate(learning_rates):
        print(f"学习率: {lr}, SGD: {sgd_train_accuracies[i]:.4f}, MBGD: {mbgd_train_accuracies[i]:.4f}, BGD: {bgd_train_accuracies[i]:.4f}")
    
    print("\n学习率与验证准确率数据:")
    for i, lr in enumerate(learning_rates):
        print(f"学习率: {lr}, SGD: {sgd_val_accuracies[i]:.4f}, MBGD: {mbgd_val_accuracies[i]:.4f}, BGD: {bgd_val_accuracies[i]:.4f}")
    
    # 绘制学习率与训练准确率的关系图
    plt.figure(figsize=(12, 8))
    plt.plot(learning_rates, sgd_train_accuracies, 'ro-', linewidth=2, markersize=8, label='SGD')  # 红色
    plt.plot(learning_rates, mbgd_train_accuracies, 'bs-', linewidth=2, markersize=8, label='MBGD')  # 蓝色
    plt.plot(learning_rates, bgd_train_accuracies, 'g^-', linewidth=2, markersize=8, label='BGD')  # 绿色
    plt.xscale('log')  # 使用对数刻度，便于观察不同量级的学习率
    plt.xlabel('学习率', fontproperties=font)
    plt.ylabel('训练准确率', fontproperties=font)
    plt.title('不同优化器在各学习率下的训练准确率', fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True)
    plt.savefig('experiment1_1.png')
    
    # 绘制学习率与验证准确率的关系图
    plt.figure(figsize=(12, 8))
    plt.plot(learning_rates, sgd_val_accuracies, 'ro-', linewidth=2, markersize=8, label='SGD')  # 红色
    plt.plot(learning_rates, mbgd_val_accuracies, 'bs-', linewidth=2, markersize=8, label='MBGD')  # 蓝色
    plt.plot(learning_rates, bgd_val_accuracies, 'g^-', linewidth=2, markersize=8, label='BGD')  # 绿色
    plt.xscale('log')  # 使用对数刻度，便于观察不同量级的学习率
    plt.xlabel('学习率', fontproperties=font)
    plt.ylabel('验证准确率', fontproperties=font)
    plt.title('不同优化器在各学习率下的验证准确率', fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True)
    plt.savefig('experiment1_2.png')
    plt.show()

if __name__ == "__main__":
    experiment1()