import numpy as np
from src.data_loader import DataLoader
from src.feature_extractor import NGramExtractor
from src.model import SentimentClassifier, MBGD, train_model, evaluate_model
import os
import time
import wandb

def train_with_staged_lr_batched(model, feature_extractor, train_texts, train_labels, val_texts, val_labels, 
                                batch_size=32, train_batch_size=5000, epochs_per_stage=None, learning_rates=None):
    """
    使用分阶段学习率和分批次训练模型
    
    Args:
        model: 模型实例
        feature_extractor: 特征提取器
        train_texts: 训练文本
        train_labels: 训练标签
        val_texts: 验证文本
        val_labels: 验证标签
        batch_size: 模型训练批次大小
        train_batch_size: 大批次大小，用于分批生成特征
        epochs_per_stage: 每个阶段的训练轮数列表
        learning_rates: 每个阶段的学习率列表
        
    Returns:
        训练好的模型
    """
    if epochs_per_stage is None:
        epochs_per_stage = [20, 20, 10]  # 默认三个阶段：20轮+20轮+10轮
        
    if learning_rates is None:
        learning_rates = [0.5, 0.2, 0.05]  # 默认学习率策略：先大后小
        
    # 生成验证集特征
    print("生成验证集特征...")
    val_features = feature_extractor.transform(val_texts)
    
    print("开始分阶段学习率训练...")
    
    global_epoch = 0  # 全局epoch计数，用于wandb记录
    
    for stage, (epochs, lr) in enumerate(zip(epochs_per_stage, learning_rates)):
        print(f"\n第{stage+1}阶段 - 学习率: {lr}, 训练轮数: {epochs}")
        wandb.log({"stage": stage+1, "learning_rate": lr})
        
        optimizer = MBGD(learning_rate=lr, batch_size=batch_size)
        
        for epoch in range(epochs):
            # 分批加载训练数据，避免内存溢出
            train_loss = 0
            num_batches = 0
            
            # 打乱训练数据索引
            indices = np.random.permutation(len(train_texts))
            
            for batch_start in range(0, len(indices), train_batch_size):
                batch_end = min(batch_start + train_batch_size, len(indices))
                batch_indices = indices[batch_start:batch_end]
                
                # 提取当前批次的文本和标签
                batch_texts = [train_texts[i] for i in batch_indices]
                batch_labels = np.array([train_labels[i] for i in batch_indices])
                
                # 转换为特征
                batch_features = feature_extractor.transform(batch_texts)
                
                # 更新模型
                batch_loss = optimizer.update(model, batch_features, batch_labels)
                train_loss += batch_loss
                num_batches += 1
            
            # 计算平均训练损失
            avg_train_loss = train_loss / num_batches
            
            # 每个epoch都评估，记录wandb指标
            # 分批评估验证集
            val_correct = 0
            val_loss = 0
            eval_batch_size = 1000
            
            for i in range(0, len(val_features), eval_batch_size):
                end = min(i + eval_batch_size, len(val_features))
                X_val_batch = val_features[i:end]
                y_val_batch = np.array([val_labels[j] for j in range(i, end)])
                
                val_preds = model.forward(X_val_batch)
                val_loss += model.cross_entropy_loss(val_preds, model.to_one_hot(y_val_batch, model.output_dim))
                val_correct += np.sum(model.predict(X_val_batch) == y_val_batch)
            
            val_acc = val_correct / len(val_labels)
            val_loss /= (len(val_features) // eval_batch_size + (1 if len(val_features) % eval_batch_size > 0 else 0))
            
            # 评估训练集准确率（使用样本）
            sample_size = min(5000, len(train_texts))  # 使用一部分样本来快速评估
            sample_indices = np.random.choice(len(train_texts), sample_size, replace=False)
            sample_texts = [train_texts[i] for i in sample_indices]
            sample_labels = np.array([train_labels[i] for i in sample_indices])
            sample_features = feature_extractor.transform(sample_texts)
            train_preds = model.predict(sample_features)
            train_acc = np.mean(train_preds == sample_labels)
            
            # 记录到wandb
            global_epoch += 1
            wandb.log({
                "epoch": global_epoch,
                "stage": stage+1,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": lr
            })
            
            # 每10个epoch或最后一个epoch打印信息
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # 每个阶段结束后评估模型
        print(f"第{stage+1}阶段结束 - 评估中...")
        
        # 分批评估训练集
        train_correct = 0
        eval_batch_size = 1000
        
        for batch_start in range(0, len(train_texts), train_batch_size):
            batch_end = min(batch_start + train_batch_size, len(train_texts))
            batch_texts = train_texts[batch_start:batch_end]
            batch_labels = np.array([train_labels[i] for i in range(batch_start, batch_end)])
            batch_features = feature_extractor.transform(batch_texts)
            
            for i in range(0, len(batch_features), eval_batch_size):
                end = min(i + eval_batch_size, len(batch_features))
                X_batch = batch_features[i:end]
                y_batch = batch_labels[i:end]
                train_correct += np.sum(model.predict(X_batch) == y_batch)
        
        train_acc = train_correct / len(train_labels)
        
        # 评估验证集
        val_correct = 0
        for i in range(0, len(val_features), eval_batch_size):
            end = min(i + eval_batch_size, len(val_features))
            X_val_batch = val_features[i:end]
            y_val_batch = np.array([val_labels[j] for j in range(i, end)])
            val_correct += np.sum(model.predict(X_val_batch) == y_val_batch)
        
        val_acc = val_correct / len(val_labels)
        
        # 记录阶段结束的完整评估结果
        wandb.log({
            "stage_end": stage+1,
            "stage_train_acc": train_acc,
            "stage_val_acc": val_acc
        })
        
        print(f"第{stage+1}阶段结束 - 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")
    
    return model

def predict_test_data():
    # 设置随机种子，确保结果可复现
    np.random.seed(42)
    
    # 创建结果目录(如果不存在)
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # 记录开始时间
    start_time = time.time()
    
    # 初始化wandb
    wandb.init(
        project="sentiment-analysis-project", 
        name=f"staged_lr_{int(time.time())}",
        config={
            "model_type": "logistic_regression",
            "feature_type": "ngram",
            "ngram_dimension": 3,
            "binary_features": True,
            "use_cumulative": True,
            "max_features": 10000,
            "training_strategy": "staged_learning_rate"
        }
    )
    
    print("加载数据...")
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_data()
    
    # 获取全部训练数据(不需要验证集划分)
    train_data = data_loader.train_data
    train_texts = train_data['processed_text'].values
    train_labels = train_data['Sentiment'].values
    
    # 从训练数据中划分一小部分作为验证集，用于监控训练进度
    train_indices = list(range(len(train_texts)))
    np.random.shuffle(train_indices)
    val_size = int(0.1 * len(train_indices))  # 使用10%的数据作为验证集
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]
    
    # 划分训练集和验证集
    X_train_texts = [train_texts[i] for i in train_indices]
    y_train = np.array([train_labels[i] for i in train_indices])
    X_val_texts = [train_texts[i] for i in val_indices]
    y_val = np.array([train_labels[i] for i in val_indices])
    
    # 获取测试数据
    test_texts, test_phrase_ids = data_loader.get_test_data()
    print(f"训练数据: {len(X_train_texts)}个样本, 验证数据: {len(X_val_texts)}个样本, 测试数据: {len(test_texts)}个样本")
    
    # 记录数据集信息
    wandb.log({
        "train_samples": len(X_train_texts),
        "val_samples": len(X_val_texts),
        "test_samples": len(test_texts)
    })
    
    # 使用与experiment5相同的参数进行特征提取，但减少特征数量减少内存占用
    print("提取特征中，这可能需要一些时间...")
    feature_extractor = NGramExtractor(
        dimension=3,
        min_freq=10, 
        max_features=20000,  
        binary=True, 
        use_cumulative=True
    )
    
    # 分批拟合特征提取器
    print("使用全部训练数据拟合特征提取器...")
    
    # 使用全部训练数据来拟合特征提取器，不再随机采样
    feature_extractor.fit(X_train_texts)
    print(f"特征维度: {feature_extractor.vocab_size}")
    
    # 记录特征提取信息
    wandb.log({"feature_dim": feature_extractor.vocab_size})
    
    print("使用分阶段学习率策略训练模型...")
    # 创建模型
    model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=None, output_dim=5)
    
    # 定义分阶段学习率策略
    epochs_per_stage = [15, 20, 15]  # 总共50轮，分三个阶段
    learning_rates = [0.5, 0.3, 0.1]  # 根据实验1结果，MBGD在较大学习率下表现较好
    
    # 记录训练策略
    wandb.config.update({
        "epochs_per_stage": epochs_per_stage,
        "learning_rates": learning_rates,
        "total_epochs": sum(epochs_per_stage)
    })
    
    # 使用分阶段学习率和分批次训练
    train_with_staged_lr_batched(
        model, feature_extractor,
        X_train_texts, y_train, X_val_texts, y_val, 
        batch_size=32, train_batch_size=5000,
        epochs_per_stage=epochs_per_stage, 
        learning_rates=learning_rates
    )
    
    print("在测试集上进行预测...")
    # 分批预测测试数据
    batch_size = 1000
    test_predictions = []
    
    for i in range(0, len(test_texts), batch_size):
        end = min(i + batch_size, len(test_texts))
        X_batch_texts = test_texts[i:end]
        X_batch = feature_extractor.transform(X_batch_texts)
        batch_predictions = model.predict(X_batch)
        test_predictions.extend(batch_predictions)
    
    # 创建提交文件
    submission_file = f'results/submission_staged_lr_{int(time.time())}.csv'
    data_loader.create_submission(test_phrase_ids, test_predictions, output_path=submission_file)
    
    # 计算运行时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"总运行时间: {total_time:.2f}秒")
    print(f"预测完成！提交文件已保存到: {submission_file}")
    
    # 记录最终结果
    wandb.log({
        "total_runtime": total_time,
        "submission_file": submission_file
    })
    
    # 结束wandb会话
    wandb.finish()

if __name__ == "__main__":
    predict_test_data() 