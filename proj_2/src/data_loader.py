import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
import re
import os
from collections import Counter

class TextDataset(Dataset):
    """文本数据集类"""
    
    def __init__(self, texts, labels, vocab=None, max_length=200):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = self.tokenize
        self.vocab = vocab if vocab else self._build_vocab()
    
    def tokenize(self, text):
        """自定义分词函数，替代 torchtext 的 get_tokenizer
        
        简单按空格和标点符号分割文本
        """
        # 将文本转换为小写并移除多余空格
        text = text.lower().strip()
        # 在标点符号前后添加空格以便分割
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        # 按空格分割为tokens
        tokens = text.split()
        return tokens
    
    def _build_vocab(self):
        counter = Counter()
        for text in self.texts:
            tokens = self.tokenizer(self.preprocess_text(text))
            counter.update(tokens)
        
        vocab = {'<unk>': 0, '<pad>': 1}
        for i, (word, _) in enumerate(counter.most_common(), start=2):
            vocab[word] = i
        
        return vocab
    
    def preprocess_text(self, text):
        # 确保文本是字符串类型
        if not isinstance(text, str):
            # 如果是NaN或其他数值类型，转为字符串
            text = str(text)
            print(f"警告: 发现非字符串输入: {text[:20]}..., 已转换为字符串")
            
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        text = re.sub(r'\d+', '[NUMBER]', text)
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def text_to_tensor(self, text):
        tokens = self.tokenizer(self.preprocess_text(text))
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend(['<pad>'] * (self.max_length - len(tokens)))
        
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.text_to_tensor(self.texts[idx]), self.labels[idx]


def load_data(data_dir, batch_size=32, max_length=200, val_ratio=0.2, random_seed=42):
    """加载数据并创建数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        max_length: 最大文本长度
        val_ratio: 验证集比例
        random_seed: 随机种子
        
    Returns:
        tuple: (train_loader, val_loader, vocab, vocab_size)
    """
    # 设置随机种子
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 读取数据文件
    data_path = os.path.join(data_dir, 'train.tsv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
    data = pd.read_csv(data_path, sep='\t', header=0)
    data = data.dropna(subset=['Phrase']).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    data['Sentiment'] = data['Sentiment'].astype(int)
    
    # 分割数据集
    val_size = int(len(data) * val_ratio)
    train_data = data[val_size:]
    val_data = data[:val_size]
    
    # 创建数据集
    train_dataset = TextDataset(
        train_data['Phrase'].tolist(),
        train_data['Sentiment'].tolist(),
        max_length=max_length
    )
    
    val_dataset = TextDataset(
        val_data['Phrase'].tolist(),
        val_data['Sentiment'].tolist(),
        vocab=train_dataset.vocab,
        max_length=max_length
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"词表大小: {len(train_dataset.vocab)}")
    print(f"训练集: {len(train_dataset)}个样本, 验证集: {len(val_dataset)}个样本")
    
    return train_loader, val_loader, train_dataset.vocab, len(train_dataset.vocab)


def create_test_loader(test_file, vocab, batch_size=32, max_length=200):
    """创建测试数据加载器
    
    Args:
        test_file: 测试数据文件路径
        vocab: 词表
        batch_size: 批次大小
        max_length: 最大文本长度
        
    Returns:
        DataLoader: 测试数据加载器
    """
    try:
        print(f"加载测试文件: {test_file}")
        test_data = pd.read_csv(test_file, sep='\t')
        
        # 获取文本列
        if 'Phrase' in test_data.columns:
            texts = test_data['Phrase'].values
            print(f"找到{len(texts)}条测试样本")
        else:
            # 如果没有指定的列名，假设第一列是文本
            texts = test_data.iloc[:, 0].values
            print(f"未找到'Phrase'列，使用第一列作为文本，共{len(texts)}条测试样本")
            
        # 创建测试数据集
        test_dataset = TextDataset(
            texts,
            [0] * len(texts),  # 占位标签
            vocab=vocab,
            max_length=max_length
        )
        
        # 创建数据加载器
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    except Exception as e:
        print(f"加载测试数据时出错: {e}")
        import traceback
        traceback.print_exc()
        raise





