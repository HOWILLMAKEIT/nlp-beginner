import numpy as np
from collections import Counter

class NGramExtractor:
    
    def __init__(self, dimension=1, min_freq=5, max_features=10000):
        """
        Args:
            dimension: 最大的gram数，例如dimension=2表示同时使用1-gram和2-gram
            min_freq: 词频阈值，只保留出现次数大于等于此值的n-gram
            max_features: 最大特征数量，如果n-gram总数超过此值，只保留最高频的max_features个
        """
        self.dimension = dimension  # 最大gram数
        self.min_freq = min_freq  # 最小词频
        self.max_features = max_features  # 最大特征数量
        self.ngram_vocab = {}  # n-gram词汇表（键值对）
        self.vocab_size = 0  # 词汇表大小
    
    def fit(self, texts):
        """创建n-gram词汇表，只保留频率大于min_freq的词"""
        # 统计所有n-gram的出现次数
        ngram_counter = Counter()
        
        # 提取文本中的所有n-gram并统计频率
        for n in range(1, self.dimension + 1):
            for text in texts:
                words = text.lower().split()
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    ngram_counter[ngram] += 1
        
        # 只保留出现频率大于等于min_freq的n-gram
        filtered_ngrams = [ngram for ngram, count in ngram_counter.items() 
                          if count >= self.min_freq]
        
        # 如果特征数量超过max_features，只保留最高频的max_features个
        if len(filtered_ngrams) > self.max_features:
            most_common = ngram_counter.most_common(self.max_features)
            filtered_ngrams = [ngram for ngram, _ in most_common]
        
        # 创建词汇表映射
        self.ngram_vocab = {ngram: idx for idx, ngram in enumerate(filtered_ngrams)}
        self.vocab_size = len(self.ngram_vocab)
        
        print(f"原始n-gram数量: {len(ngram_counter)}, 过滤后n-gram数量: {self.vocab_size}")
        return self
    
    def transform(self, texts):
        """将文本转换为n-gram特征向量"""
        features = np.zeros((len(texts), self.vocab_size), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            for n in range(1, self.dimension + 1):
                for j in range(len(words) - n + 1):
                    ngram = ' '.join(words[j:j+n])
                    if ngram in self.ngram_vocab:
                        features[i, self.ngram_vocab[ngram]] = 1
        return features
    
    def fit_transform(self, texts):
        """先构建词汇表，再转换为特征向量"""
        return self.fit(texts).transform(texts)


