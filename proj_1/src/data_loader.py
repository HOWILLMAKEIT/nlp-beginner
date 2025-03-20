import pandas as pd
import numpy as np
import re
import random

class DataLoader:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        
    def load_data(self, train_path='data/train.tsv', test_path='data/test.tsv'):
        """加载数据集"""
        # 加载训练集，包含列：PhraseId, SentenceId, Phrase, Sentiment
        # 不过这个句子id SentenceId似乎没什么用
        self.train_data = pd.read_csv(train_path, sep='\t')
        # 加载测试集，包含列：PhraseId, SentenceId, Phrase
        if test_path:
            self.test_data = pd.read_csv(test_path, sep='\t')
        return self
        
    def preprocess_text(self, text):
        """预处理文本"""
        if not isinstance(text, str):
            return ""
        # 转为小写
        text = text.lower()
        # 去除标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 去除多余的空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_data(self):
        """对数据集进行预处理"""
        # 预处理训练集文本
        self.train_data['processed_text'] = self.train_data['Phrase'].apply(self.preprocess_text)
        if self.test_data is not None:
            self.test_data['processed_text'] = self.test_data['Phrase'].apply(self.preprocess_text)
        return self
    
    def split_train_val(self, val_ratio=0.2, random_state=20):
        """将训练集分为训练集和验证集"""
        random.seed(random_state)
        texts = self.train_data['processed_text'].values
        labels = self.train_data['Sentiment'].values
        indices = list(range(len(texts)))
        random.shuffle(indices)
        # 分割
        split = int(val_ratio * len(texts))
        val_indices = indices[:split]
        train_indices = indices[split:]
        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_texts = [texts[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        return train_texts, train_labels, val_texts, val_labels
    
    def get_test_data(self):
        if self.test_data is None:
            raise ValueError("测试集未加载")
        return self.test_data['processed_text'].values, self.test_data['PhraseId'].values
    
    
    def create_submission(self, phrase_ids, predictions, output_path='./submission.csv'):
        """创建提交文件
        Args:
            phrase_ids: 测试集短语ID列表
            predictions: 预测结果列表
            output_path: 输出文件路径
        """
        submission = pd.DataFrame({
            'PhraseId': phrase_ids,
            'Sentiment': predictions
        })
        submission.to_csv(output_path, index=False)
        print(f"提交文件已保存至 {output_path}")
        