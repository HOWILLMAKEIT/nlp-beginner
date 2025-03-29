import numpy as np
from src.model import train_sentiment_model
from src.utils import set_seed, get_device
import torch
import traceback
import os

def experiment_lstm_single():
    """单向LSTM模型实验（使用预训练词向量）"""
    print("\n====== 单向LSTM模型（预训练词向量）======")
    try:
        # 创建checkpoint目录
        checkpoint_dir = os.path.join('checkpoints', 'lstm_single')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model, history = train_sentiment_model(
            data_dir='data',
            model_type='lstm',
            bidirectional=False,  # 单向LSTM
            batch_size=64,
            max_length=200,
            embed_dim=300,
            hidden_dim=128,
            num_layers=2,
            dropout=0.5,
            learning_rate=0.001,
            num_epochs=20,
            use_pretrained=True,  # 使用预训练词向量
            use_wandb=False,      # 不使用wandb记录
            generate_submission=True,  # 生成测试集提交文件
            checkpoint_dir=checkpoint_dir,  # 指定checkpoint目录
            seed=42
        )
        
        print(f"单向LSTM（预训练词向量）- 最终训练准确率: {history['train_acc'][-1]:.2f}%, 验证准确率: {history['val_acc'][-1]:.2f}%")
        return model, history
    except Exception as e:
        print(f"单向LSTM实验出错: {e}")
        traceback.print_exc()
        return None, None

def experiment_lstm_bidirectional():
    """双向LSTM模型实验（使用预训练词向量 + 注意力机制）"""
    print("\n====== 双向LSTM模型（预训练词向量 + 注意力机制）======")
    try:
        # 创建checkpoint目录
        checkpoint_dir = os.path.join('checkpoints', 'lstm_bidirectional')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model, history = train_sentiment_model(
            data_dir='data',
            model_type='lstm',
            bidirectional=True,  # 双向LSTM
            batch_size=64,
            max_length=200,
            embed_dim=300,
            hidden_dim=256,
            num_layers=2,
            dropout=0.5,
            learning_rate=0.001,
            num_epochs=10,
            use_pretrained=True,  # 使用预训练词向量
            use_wandb=True,      # 使用wandb记录
            generate_submission=True,  # 生成测试集提交文件
            checkpoint_dir=checkpoint_dir,  # 指定checkpoint目录
            use_attention=True,   # 启用注意力机制
            attention_heads=1,
            seed=42
        )
        
        print(f"双向LSTM（预训练词向量 + 注意力机制）- 最终训练准确率: {history['train_acc'][-1]:.2f}%, 验证准确率: {history['val_acc'][-1]:.2f}%")
        return model, history
    except Exception as e:
        print(f"双向LSTM实验出错: {e}")
        traceback.print_exc()
        return None, None

def experiment_cnn():
    """CNN模型实验（使用预训练词向量）"""
    print("\n====== CNN模型（预训练词向量）======")
    model, history = train_sentiment_model(
        data_dir='data',
        model_type='cnn',
        batch_size=64,
        max_length=200,
        embed_dim=300,
        filter_sizes=[3, 4, 5],  # CNN特定参数
        num_filters=100,         # CNN特定参数
        dropout=0.5,
        learning_rate=0.001,
        num_epochs=20,           # 增加训练轮数
        use_pretrained=True,     # 使用预训练词向量
        use_wandb=True,         # 使用wandb记录
        generate_submission=True,  # 生成测试集提交文件
        seed=42
    )
    
    print(f"CNN（预训练词向量）- 最终训练准确率: {history['train_acc'][-1]:.2f}%, 验证准确率: {history['val_acc'][-1]:.2f}%")
    return model, history


def main():
    """主函数：运行所有实验并比较结果"""
    # 设置随机种子确保实验可重现性
    set_seed(42)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"CUDA可用! 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA不可用，使用CPU训练")
    
    print("开始情感分析模型实验...")
    # experiment_lstm_single()
    experiment_lstm_bidirectional()
    # experiment_cnn()
    

if __name__ == "__main__":
    main()