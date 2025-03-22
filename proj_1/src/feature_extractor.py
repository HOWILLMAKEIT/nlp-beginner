import numpy as np
from collections import Counter

class NGramExtractor:
    def __init__(self, dimension=1, min_freq=1, max_features=None, binary=False, use_cumulative=False):
        """
        用于从文本中提取n-gram特征的类
        
        参数:
            dimension: int, n-gram的n值，默认为1
            min_freq: int, n-gram的最小频率阈值，低于此频率的n-gram将被过滤掉
            max_features: int, 要保留的最大特征数量，如果为None则保留全部
            binary: bool, 如果为True，特征值将为二进制（0或1），否则为计数值
            use_cumulative: bool, 如果为True，特征将包括1-gram到n-gram
        """
        self.dimension = dimension
        self.min_freq = min_freq
        self.max_features = max_features
        self.binary = binary
        self.use_cumulative = use_cumulative
        self.vocab = {}  # 词汇表，将n-gram映射到索引
        self.vocab_size = 0  # 词汇表大小
        
    def _extract_ngrams(self, text, n):
        """
        从文本中提取n-gram
        
        参数:
            text: str, 输入文本
            n: int, n-gram的n值
            
        返回:
            n-gram列表
        """
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        
    def fit(self, texts):
        """
        构建词汇表
        
        参数:
            texts: list, 文本列表
            
        返回:
            self
        """
        # 计数器，用于统计n-gram频率
        ngram_counts = Counter()
        
        # 如果use_cumulative为True，则提取1-gram到n-gram
        dimensions = range(1, self.dimension + 1) if self.use_cumulative else [self.dimension]
        
        # 统计n-gram频率
        for text in texts:
            for n in dimensions:
                ngrams = self._extract_ngrams(text, n)
                ngram_counts.update(ngrams)
        
        # 过滤低频n-gram
        filtered_ngrams = [(ngram, count) for ngram, count in ngram_counts.items() if count >= self.min_freq]
        
        # 按频率降序排序
        filtered_ngrams.sort(key=lambda x: x[1], reverse=True)
        
        # 如果指定了max_features，则只保留top-K个n-gram
        if self.max_features is not None and len(filtered_ngrams) > self.max_features:
            filtered_ngrams = filtered_ngrams[:self.max_features]
        
        # 构建词汇表
        self.vocab = {ngram: idx for idx, (ngram, _) in enumerate(filtered_ngrams)}
        self.vocab_size = len(self.vocab)
        
        return self
        
    def transform(self, texts):
        """
        将文本转换为特征向量
        
        参数:
            texts: list, 文本列表
            
        返回:
            特征矩阵，每行对应一个文本，每列对应一个n-gram特征
        """
        # 创建一个全零矩阵
        features = np.zeros((len(texts), self.vocab_size), dtype=np.float32)
        
        dimensions = range(1, self.dimension + 1) if self.use_cumulative else [self.dimension]
        
        for text_idx, text in enumerate(texts):
            # 提取n-gram并计数
            text_ngram_counts = Counter()
            for n in dimensions:
                ngrams = self._extract_ngrams(text, n)
                text_ngram_counts.update(ngrams)
            
            # 更新特征矩阵
            for ngram, count in text_ngram_counts.items():
                if ngram in self.vocab:
                    features[text_idx, self.vocab[ngram]] = 1 if self.binary else count
        
        return features
        
    def fit_transform(self, texts):
        """
        构建词汇表并将文本转换为特征向量
        
        参数:
            texts: list, 文本列表
            
        返回:
            特征矩阵，每行对应一个文本，每列对应一个n-gram特征
        """
        self.fit(texts)
        return self.transform(texts)


