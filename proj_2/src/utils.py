import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import requests
import zipfile
import io
import random
import matplotlib.font_manager as fm

def set_seed(seed: int = 42) -> None:
    """设置随机种子以确保结果可重现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """获取可用的设备（CPU/GPU）
    
    Returns:
        torch.device: 可用的设备
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def download_glove_embeddings(glove_dir: str, glove_name: str = 'glove.6B.300d') -> str:
    """下载GloVe预训练词向量
    
    Args:
        glove_dir: GloVe文件保存目录
        glove_name: GloVe模型名称，默认使用300维的glove.6B
        
    Returns:
        str: GloVe文件路径
    """
    # 创建目录
    os.makedirs(glove_dir, exist_ok=True)
    
    # 文件路径
    glove_file = os.path.join(glove_dir, f'{glove_name}.txt')
    
    # 如果文件已存在，直接返回路径
    if os.path.exists(glove_file):
        print(f'GloVe文件已存在: {glove_file}')
        return glove_file
    
    # 下载文件
    print('正在下载GloVe词向量...')
    url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    response = requests.get(url)
    
    # 解压文件
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extract(f'{glove_name}.txt', glove_dir)
    
    print(f'GloVe词向量已下载并解压到: {glove_file}')
    return glove_file

def load_glove_embeddings(glove_file: str, vocab: dict, embed_dim: int = 300) -> torch.Tensor:
    """加载GloVe预训练词向量
    
    Args:
        glove_file: GloVe文件路径
        vocab: 词表字典
        embed_dim: 嵌入维度
        
    Returns:
        torch.Tensor: 预训练的嵌入矩阵
    """
    print(f'加载预训练词向量: {glove_file}')
    
    # 初始化嵌入矩阵
    embedding_matrix = torch.zeros((len(vocab), embed_dim))
    
    # 为特殊标记使用随机初始化
    for special_token in ['<unk>', '<pad>']:
        if special_token in vocab:
            embedding_matrix[vocab[special_token]] = torch.randn(embed_dim) * 0.1
    
    # 读取GloVe文件
    found_count = 0
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            
            # 如果词在词表中，则更新嵌入矩阵
            if word in vocab:
                vector = torch.FloatTensor([float(val) for val in values[1:]])
                embedding_matrix[vocab[word]] = vector
                found_count += 1
    
    print(f'词表中的{len(vocab)}个词中有{found_count}个在预训练向量中找到')
    return embedding_matrix

# 绘制混淆矩阵
def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         labels: List[str],
                         save_path: Optional[str] = None) -> None:
    """绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 标签名称列表
        save_path: 保存图片的路径（可选）
    """
    # 添加中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

