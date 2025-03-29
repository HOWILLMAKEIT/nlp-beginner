import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
import wandb
from tqdm import tqdm
import numpy as np
from src.utils import get_device, download_glove_embeddings, load_glove_embeddings
import torch.optim as optim
import os
from datetime import datetime
import pandas as pd
import math

from src.data_loader import load_data, create_test_loader
from src.utils import set_seed, plot_confusion_matrix

class SentimentClassifierBase(nn.Module):
    """情感分类模型基类"""
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 300,
                 num_classes: int = 5,
                 dropout: float = 0.5,
                 pretrained_embedding: Optional[torch.Tensor] = None):
        """初始化模型
        
        Args:
            vocab_size: 词表大小
            embed_dim: 词嵌入维度
            num_classes: 类别数量
            dropout: Dropout比率
            pretrained_embedding: 预训练词嵌入矩阵（可选）
        """
        super().__init__()
        
        # 词嵌入层
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding,
                padding_idx=1,
                freeze=False
            )
            # 更新维度以匹配预训练嵌入
            vocab_size, embed_dim = pretrained_embedding.shape
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        
        # 保存关键参数
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，shape为(batch_size, sequence_length)
        
        Returns:
            torch.Tensor: 输出张量，shape为(batch_size, num_classes)
        """
        raise NotImplementedError("子类必须实现forward方法")


class CNNSentimentClassifier(SentimentClassifierBase):
    """CNN情感分类模型"""
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 300,
                 num_classes: int = 5,
                 filter_sizes: List[int] = [3, 4, 5],
                 num_filters: int = 100,
                 dropout: float = 0.5,
                 pretrained_embedding: Optional[torch.Tensor] = None):
        """初始化CNN模型
        
        Args:
            vocab_size: 词表大小
            embed_dim: 词嵌入维度
            num_classes: 类别数量
            filter_sizes: 卷积核大小列表
            num_filters: 每种卷积核的数量
            dropout: Dropout比率
            pretrained_embedding: 预训练词嵌入矩阵（可选）
        """
        super().__init__(vocab_size, embed_dim, num_classes, dropout, pretrained_embedding)
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # 输出层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        # 保存参数
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，shape为(batch_size, sequence_length)
        
        Returns:
            torch.Tensor: 输出张量，shape为(batch_size, num_classes)
        """
        # 词嵌入，shape: (batch_size, sequence_length, embed_dim)
        x = self.embedding(x)
        
        # 转换维度顺序，以适应Conv1d
        # 从(batch_size, sequence_length, embed_dim) 到 (batch_size, embed_dim, sequence_length)
        x = x.permute(0, 2, 1)
        
        # 应用卷积层
        # 每个卷积输出形状为 (batch_size, num_filters, seq_len - filter_size + 1)
        conved = [F.relu(conv(x)) for conv in self.convs]
        
        # 池化，每个输出形状为 (batch_size, num_filters, 1)
        pooled = [F.max_pool1d(conv, conv.shape[2]) for conv in conved]
        
        # 连接所有池化结果，形状为 (batch_size, len(filter_sizes) * num_filters)
        cat = torch.cat([pool.squeeze(2) for pool in pooled], dim=1)
        
        # Dropout
        cat = self.dropout(cat)
        
        # 全连接层
        return self.fc(cat)




class LSTMSentimentClassifier(SentimentClassifierBase):
    """基于LSTM的情感分类模型，使用KQV注意力机制"""
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 300,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 num_classes: int = 5,
                 dropout: float = 0.5,
                 use_attention: bool = False,
                 attention_heads: int = 4,  # 注意力头数量
                 pretrained_embedding: Optional[torch.Tensor] = None):
        """初始化LSTM模型"""
        # 调用基类初始化，处理词嵌入
        super().__init__(vocab_size, embed_dim, num_classes, dropout, pretrained_embedding)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 保存参数
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        
        # 输出层维度
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 多头注意力机制 (KQV形式)
        if use_attention:
            self.d_k = self.output_dim // attention_heads  # 每个头的维度
            
            # 定义Q、K、V的线性变换
            self.q_linear = nn.Linear(self.output_dim, self.output_dim)
            self.k_linear = nn.Linear(self.output_dim, self.output_dim)
            self.v_linear = nn.Linear(self.output_dim, self.output_dim)
            
            # 输出线性层
            self.out_linear = nn.Linear(self.output_dim, self.output_dim)
        
        # 分类层
        self.fc = nn.Linear(self.output_dim, num_classes)
    
    def attention(self, q, k, v, mask=None):
        """实现缩放点积注意力机制"""
        batch_size = q.size(0)
        
        # 分割为多头 shape: (batch_size, heads, seq_len, d_k)
        q = q.view(batch_size, -1, self.attention_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.attention_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.attention_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数 (batch_size, heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码(如果有)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重得到输出
        output = torch.matmul(attn_weights, v)
        
        # 拼接多头的结果
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.output_dim)
        
        return output, attn_weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 词嵌入 shape: (batch_size, seq_len, embed_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM处理 output shape: (batch_size, seq_len, hidden_dim*num_directions)
        output, (hidden, _) = self.lstm(embedded)
        
        if self.use_attention:
            # 生成Q、K、V
            q = self.q_linear(output)
            k = self.k_linear(output)
            v = self.v_linear(output)
            
            # 应用多头注意力
            attn_output, _ = self.attention(q, k, v)
            
            # 最终线性变换
            context_vector = self.out_linear(attn_output[:, -1, :])  # 使用最后一个时间步的输出
            
            # Dropout
            features = self.dropout(context_vector)
        else:
            # 不使用注意力时，直接使用LSTM的最终隐藏状态
            if self.bidirectional:
                hidden_final = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden_final = hidden[-1]
            features = self.dropout(hidden_final)
        
        # 全连接层分类
        return self.fc(features)


class ResidualLSTMSentimentClassifier(SentimentClassifierBase):
    """带残差连接的LSTM情感分类模型"""
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 300,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 num_classes: int = 5,
                 dropout: float = 0.5,
                 use_attention: bool = False,
                 pretrained_embedding: Optional[torch.Tensor] = None):
        """初始化带残差连接的LSTM模型"""
        super().__init__(vocab_size, embed_dim, num_classes, dropout, pretrained_embedding)
        
        # 第一层LSTM
        self.lstm1 = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 计算第二层及以后的LSTM输入维度
        lstm_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 创建其余的LSTM层（带残差连接）
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    bidirectional=bidirectional,
                    batch_first=True
                )
            )
        
        # 其他参数保存
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # 注意力机制
        if use_attention:
            attn_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.attention = nn.Sequential(
                nn.Linear(attn_dim, attn_dim // 2),
                nn.Tanh(),
                nn.Linear(attn_dim // 2, 1)
            )
        
        # 分类层
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 词嵌入
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        
        # 第一层LSTM
        output, (hidden, cell) = self.lstm1(embedded)
        
        # 后续LSTM层（带残差连接）
        for i, lstm_layer in enumerate(self.lstm_layers):
            new_output, (new_hidden, new_cell) = lstm_layer(output)
            # 添加残差连接
            output = output + new_output
        
        # 处理最终输出
        if self.use_attention:
            # 注意力机制
            attention_weights = self.attention(output).squeeze(-1)
            attention_weights = F.softmax(attention_weights, dim=1)
            attention_weights = attention_weights.unsqueeze(1)
            context_vector = torch.bmm(attention_weights, output)
            context_vector = context_vector.squeeze(1)
            features = self.dropout(context_vector)
        else:
            # 获取最终隐藏状态
            if self.bidirectional:
                hidden_final = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden_final = hidden[-1]
            features = self.dropout(hidden_final)
        
        # 分类
        return self.fc(features)


class ModelFactory:
    """模型工厂类，用于创建不同类型的情感分类模型"""
    
    @staticmethod
    def create_model(model_type: str, model_params: Dict[str, Any]) -> nn.Module:
        """创建模型
        
        Args:
            model_type: 模型类型，'cnn'、'lstm'或'residual_lstm'
            model_params: 模型参数字典
        
        Returns:
            nn.Module: 创建的模型
        """
        if model_type.lower() == 'cnn':
            return CNNSentimentClassifier(**model_params)
        elif model_type.lower() == 'lstm':
            return LSTMSentimentClassifier(**model_params)
        elif model_type.lower() == 'residual_lstm':
            return ResidualLSTMSentimentClassifier(**model_params)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}，支持的类型有: cnn, lstm, residual_lstm")


def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                num_epochs: int,
                device: torch.device,
                use_wandb: bool = False,
                checkpoint_dir: str = 'checkpoints') -> Dict[str, List[float]]:
    """训练模型
    
    Args:
        model: PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        criterion: 损失函数
        num_epochs: 训练轮数
        device: 设备（CPU/GPU）
        use_wandb: 是否使用wandb记录训练过程
        checkpoint_dir: 模型保存路径
    
    Returns:
        dict: 包含训练历史的字典和最佳模型路径
    """
    # 如果需要，初始化wandb
    if use_wandb:
        try:
            wandb.init(project="sentiment-analysis", name="sentiment-model-training")
        except Exception as e:
            print(f"Wandb初始化失败: {e}，将不使用wandb记录训练过程")
            use_wandb = False
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 创建checkpoints目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 记录最佳验证准确率和对应的模型路径
    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (texts, labels) in enumerate(train_bar):
            # 将数据移到指定设备
            texts, labels = texts.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            train_bar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # 计算训练指标
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validation')
            for batch_idx, (texts, labels) in enumerate(val_bar):
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{val_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # 计算验证指标
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印当前轮次的结果
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 如果当前验证准确率是最佳的，保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, best_model_path)
            print(f"保存最佳模型于 {best_model_path}，验证准确率: {val_acc:.2f}%")
        
        # 如果使用wandb，记录训练信息
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc
            })
    
    # 如果使用wandb，训练结束时关闭
    if use_wandb:
        wandb.finish()
    
    print(f"训练完成，最佳验证准确率: {best_val_acc:.2f}%")
    return history, best_model_path

def predict(model: nn.Module,
           test_loader: torch.utils.data.DataLoader,
           device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """模型预测
    
    Args:
        model: PyTorch模型
        test_loader: 测试数据加载器
        device: 设备（CPU/GPU）
    
    Returns:
        tuple: (预测标签, 预测概率)
    """
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for texts, _ in tqdm(test_loader, desc='Predicting'):
            texts = texts.to(device)
            outputs = model(texts)
            probs = F.softmax(outputs, dim=1)
            
            predictions.extend(outputs.argmax(1).cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)

def train_sentiment_model(
    data_dir='data', 
    model_type='lstm',
    batch_size=64, 
    max_length=200,
    embed_dim=300,
    hidden_dim=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.5,
    learning_rate=0.001,
    num_epochs=10,
    # CNN特定参数
    filter_sizes=[3, 4, 5],
    num_filters=100,
    # LSTM特定参数
    use_attention=False,  # 是否使用注意力机制
    attention_heads=4,  # 添加注意力头数量参数
    # 其他参数
    use_pretrained=False,
    use_wandb=False,
    generate_submission=True,  # 是否生成提交文件
    checkpoint_dir='checkpoints',  # 模型保存路径
    seed=42
):
    """显式训练情感分析模型的函数，使用ModelFactory创建模型
    
    Args:
        data_dir: 数据目录
        model_type: 模型类型 ('lstm' 或 'cnn')
        batch_size: 批次大小
        max_length: 最大文本长度
        embed_dim: 词嵌入维度
        hidden_dim: 隐藏层维度
        num_layers: LSTM层数 (仅适用于LSTM模型)
        bidirectional: 是否使用双向LSTM (仅适用于LSTM模型)
        dropout: Dropout比率
        learning_rate: 学习率
        num_epochs: 训练轮数
        use_pretrained: 是否使用预训练词嵌入
        use_wandb: 是否使用wandb记录训练过程
        generate_submission: 是否生成提交文件
        checkpoint_dir: 模型保存路径
        seed: 随机种子
        
        # CNN特定参数
        filter_sizes: 卷积核尺寸列表（仅用于CNN模型）
        num_filters: 每种尺寸的卷积核数量（仅用于CNN模型）
        
        # LSTM特定参数
        use_attention: 是否使用注意力机制（仅用于LSTM模型）
        attention_heads: 注意力头数量（仅用于LSTM模型）
        
    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    # 设置随机种子
    set_seed(seed)
    
    # 获取设备
    device = get_device()
    print(f'使用设备: {device}')
    
    # 加载数据
    print('加载数据...')
    train_loader, val_loader, vocab, vocab_size = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        max_length=max_length,
        random_seed=seed
    )
    
    # 准备预训练词嵌入（如果需要）
    pretrained_embedding = None
    if use_pretrained:
        print('使用预训练词向量...')
        glove_dir = os.path.join(data_dir, 'glove')
        glove_file = download_glove_embeddings(glove_dir, f'glove.6B.{embed_dim}d')
        pretrained_embedding = load_glove_embeddings(glove_file, vocab, embed_dim)
        # 将预训练词向量移动到设备上
        pretrained_embedding = pretrained_embedding.to(device)
        print(f'预训练词向量已加载，形状: {pretrained_embedding.shape}')
    
    # 准备模型参数
    model_params = {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'num_classes': 5,
        'dropout': dropout,
        'pretrained_embedding': pretrained_embedding
    }
    
    # 根据模型类型添加特定参数
    if model_type == 'cnn':
        model_params['filter_sizes'] = filter_sizes
        model_params['num_filters'] = num_filters
    elif model_type == 'lstm':
        model_params['hidden_dim'] = hidden_dim
        model_params['num_layers'] = num_layers
        model_params['bidirectional'] = bidirectional
        model_params['use_attention'] = use_attention
        model_params['attention_heads'] = attention_heads  # 添加注意力头参数
    elif model_type == 'residual_lstm':
        model_params['hidden_dim'] = hidden_dim
        model_params['num_layers'] = num_layers
        model_params['bidirectional'] = bidirectional
        model_params['use_attention'] = use_attention
    
    # 使用工厂函数创建模型
    print(f'创建{model_type}模型...')
    if (model_type == 'lstm' or model_type == 'residual_lstm') and use_attention:
        print('启用注意力机制')
    model = ModelFactory.create_model(model_type, model_params).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 使用标准交叉熵损失，不使用权重
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print('开始训练...')
    history, best_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device,
        use_wandb=use_wandb,
        checkpoint_dir=checkpoint_dir
    )
    
    # 在验证集上进行预测并绘制混淆矩阵
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 加载最佳模型
    print(f'加载最佳模型: {best_model_path}')
    best_model_state = torch.load(best_model_path)
    model.load_state_dict(best_model_state['model_state_dict'])
    best_val_acc = best_model_state['val_acc']
    print(f'最佳验证准确率: {best_val_acc:.2f}%，来自Epoch {best_model_state["epoch"]+1}')
    
    # 使用最佳模型在验证集上预测
    predictions, _ = predict(model, val_loader, device)
    
    # 获取验证集标签
    val_labels = []
    for _, labels in val_loader:
        val_labels.extend(labels.tolist())
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        val_labels,
        predictions,
        labels=['极负面', '负面', '中性', '正面', '极正面'],
        save_path=f'results/{model_type}_confusion_matrix_{timestamp}.png'
    )

    # 训练完成后生成测试集预测结果
    if generate_submission:
        print('使用最佳模型生成测试集预测结果...')
        test_file = os.path.join(data_dir, 'test.tsv')
        
        # 检查测试文件是否存在
        if not os.path.exists(test_file):
            print(f"警告: 测试文件不存在: {test_file}")
            return model, history
            
        try:
            # 加载测试集
            test_loader = create_test_loader(
                test_file=test_file,
                vocab=vocab,
                batch_size=batch_size,
                max_length=max_length
            )
            
            # 预测测试集
            print('预测测试集...')
            test_predictions, test_probabilities = predict(model, test_loader, device)
            
            # 生成提交文件
            submission_file = f'results/{model_type}_submission_{timestamp}.csv'
            
            # 加载测试ID
            test_data = pd.read_csv(test_file, sep='\t')
            
            # 尝试获取ID列，如果不存在则使用序列号
            if 'PhraseId' in test_data.columns:
                test_ids = test_data['PhraseId'].values
                print(f"使用'PhraseId'列作为测试样本ID")
            else:
                # 如果没有PhraseId列，使用序列号
                print(f"未找到'PhraseId'列，使用序列号作为测试样本ID")
                test_ids = np.arange(len(test_predictions))
                
            # 确保ID列和预测结果长度一致
            if len(test_ids) != len(test_predictions):
                print(f"警告: ID列长度({len(test_ids)})与预测结果长度({len(test_predictions)})不一致")
                # 使用最小长度
                min_len = min(len(test_ids), len(test_predictions))
                test_ids = test_ids[:min_len]
                test_predictions = test_predictions[:min_len]
            
            # 创建提交DataFrame
            submission = pd.DataFrame({
                'PhraseId': test_ids,
                'Sentiment': test_predictions
            })
            
            # 保存提交文件
            submission.to_csv(submission_file, index=False)
            print(f"提交文件已保存至: {submission_file}")
            
        except Exception as e:
            print(f"生成提交文件时出错: {e}")
            import traceback
            traceback.print_exc()
            print("继续执行，但提交文件未生成")

    return model, history

