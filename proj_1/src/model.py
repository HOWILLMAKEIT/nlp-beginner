import numpy as np

class SentimentClassifier:
    """情感分类器模型"""
    def __init__(self, input_dim, hidden_dim=None, output_dim=5, use_bias=True):
        """初始化模型参数
        Args:
            input_dim: 输入特征维度，即使ngram vocab的size
            hidden_dim: 隐藏层维度，如果为None则不使用隐藏层
            output_dim: 输出维度，即五种情感
            use_bias: 是否使用偏置项
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        self.initialize_weights()
        # 缓存中间结果，用于反向传播
        self.cache = {}
    
    def initialize_weights(self):
        """初始化模型权重"""
        # 根据是否有隐藏层，初始化不同的权重
        if self.hidden_dim is None:
            # 无隐藏层，直接从输入到输出
            self.W = np.random.randn(self.input_dim, self.output_dim).astype(np.float32) * 0.01
            if self.use_bias:
                self.b = np.zeros(self.output_dim, dtype=np.float32)
        else:
            # 有隐藏层，初始化两层权重
            self.W1 = np.random.randn(self.input_dim, self.hidden_dim).astype(np.float32) * 0.01
            self.W2 = np.random.randn(self.hidden_dim, self.output_dim).astype(np.float32) * 0.01
            if self.use_bias:
                self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
                self.b2 = np.zeros(self.output_dim, dtype=np.float32)
    
    def forward(self, X):
        # 确保输入是float32类型
        X = X.astype(np.float32)
        # 缓存输入用于反向传播
        self.cache['X'] = X
        if self.hidden_dim is None:
            if self.use_bias:
                z = np.dot(X, self.W) + self.b
            else:
                z = np.dot(X, self.W)
            # 应用softmax
            self.cache['z'] = z
            return self.softmax(z)
        else:
            # 有隐藏层的情况
            if self.use_bias:
                z1 = np.dot(X, self.W1) + self.b1
            else:
                z1 = np.dot(X, self.W1)
            # eLU激活
            a1 = self.relu(z1)
            self.cache['z1'] = z1
            self.cache['a1'] = a1
            if self.use_bias:
                z2 = np.dot(a1, self.W2) + self.b2
            else:
                z2 = np.dot(a1, self.W2)
            # softmax
            self.cache['z2'] = z2
            return self.softmax(z2)
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def backward(self, y_true, learning_rate=0.01):
        batch_size = len(y_true)
        # 将真实标签转换为one-hot编码
        y_one_hot = self.to_one_hot(y_true, self.output_dim)
        # 计算交叉熵损失
        if self.hidden_dim is None:
            # 无隐藏层
            y_pred = self.softmax(self.cache['z'])
            loss = self.cross_entropy_loss(y_pred, y_one_hot)
            # 计算softmax的梯度
            dz = y_pred - y_one_hot
            # 计算权重和偏置的梯度
            dW = np.dot(self.cache['X'].T, dz) / batch_size
            if self.use_bias:
                db = np.sum(dz, axis=0) / batch_size
            # 更新权重和偏置
            self.W -= learning_rate * dW
            if self.use_bias:
                self.b -= learning_rate * db
        else:
            # 有隐藏层
            y_pred = self.softmax(self.cache['z2'])
            loss = self.cross_entropy_loss(y_pred, y_one_hot)
            
            # 计算softmax的梯度
            dz2 = y_pred - y_one_hot
            
            # 计算第二层权重和偏置的梯度
            dW2 = np.dot(self.cache['a1'].T, dz2) / batch_size
            if self.use_bias:
                db2 = np.sum(dz2, axis=0) / batch_size
            
            # 计算隐藏层的梯度
            da1 = np.dot(dz2, self.W2.T)
            # 直接计算ReLU导数：如果z1>0，导数为1；否则为0
            relu_grad = (self.cache['z1'] > 0).astype(float)
            dz1 = da1 * relu_grad
            
            # 计算第一层权重和偏置的梯度
            dW1 = np.dot(self.cache['X'].T, dz1) / batch_size
            if self.use_bias:
                db1 = np.sum(dz1, axis=0) / batch_size
            
            # 更新权重和偏置
            self.W2 -= learning_rate * dW2
            self.W1 -= learning_rate * dW1
            if self.use_bias:
                self.b2 -= learning_rate * db2
                self.b1 -= learning_rate * db1
        
        return loss
    
    def softmax(self, z):
        # 为了数值稳定性，减去每行的最大值
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def cross_entropy_loss(self, y_pred, y_true):
        # 添加一个小的常数epsilon，防止对数中出现0
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_pred)
        return loss
    
    def to_one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]


class Optimizer:
    """优化器基类"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, model, X, y):
        """更新模型参数
        
        Args:
            model: 模型
            X: 输入特征
            y: 目标标签
            
        Returns:
            损失值
        """
        raise NotImplementedError("子类必须实现update方法")


class SGD(Optimizer):
    """随机梯度下降"""
    
    def update(self, model, X, y):
        n_samples = X.shape[0]
        total_loss = 0
        
        # 创建随机索引
        indices = np.random.permutation(n_samples)
        for i in range(n_samples):
            idx = indices[i]
            X_sample = X[idx:idx+1].astype(np.float32)  # 确保float32类型
            y_sample = np.array([y[idx]])
            y_pred = model.forward(X_sample)
            loss = model.backward(y_sample, self.learning_rate)
            total_loss += loss
        return total_loss / n_samples


class MBGD(Optimizer):
    """小批量梯度下降"""
    
    def __init__(self, learning_rate=0.01, batch_size=32):
        super().__init__(learning_rate)
        self.batch_size = batch_size
    
    def update(self, model, X, y):
        n_samples = X.shape[0]
        # 创建随机索引
        indices = np.random.permutation(n_samples)
        total_loss = 0
        num_batches = 0
        for i in range(0, n_samples, self.batch_size):
            end = min(i + self.batch_size, n_samples)
            batch_indices = indices[i:end]
            X_batch = X[batch_indices].astype(np.float32)  # 确保float32类型
            y_batch = np.array([y[idx] for idx in batch_indices])
            model.forward(X_batch)
            loss = model.backward(y_batch, self.learning_rate)
            total_loss += loss
            num_batches += 1
        return total_loss / num_batches


class BGD(Optimizer):
    """批量梯度下降"""
    
    def update(self, model, X, y):
        # 始终使用分批处理，不管数据集大小
        batch_size = 1000  # 使用固定批次大小，避免内存溢出
        n_samples = X.shape[0]
        
        # 分批处理数据集
        total_loss = 0
        num_batches = 0
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            X_batch = X[i:end].astype(np.float32)  # 确保float32类型
            y_batch = np.array([y[idx] for idx in range(i, end)])
            model.forward(X_batch)
            loss = model.backward(y_batch, self.learning_rate)
            total_loss += loss
            num_batches += 1
        return total_loss / num_batches


def train_model(model, optimizer, X_train, y_train, X_val, y_val, epochs=10, compute_history=False, batch_size=1000):
    """
    训练模型
    
    Args:
        model: 模型实例
        optimizer: 优化器实例
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        epochs: 训练轮数
        compute_history: 是否计算训练历史
        batch_size: 用于评估的批次大小，避免内存溢出
        
    Returns:
        训练历史
    """
    if compute_history:
        history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # 训练一轮
        train_loss = optimizer.update(model, X_train, y_train)
        
        # 在验证集上分批评估，避免内存溢出
        val_loss = 0
        num_val_batches = 0
        for i in range(0, len(X_val), batch_size):
            end = min(i + batch_size, len(X_val))
            X_val_batch = X_val[i:end]
            y_val_batch = y_val[i:end]
            val_preds = model.forward(X_val_batch)
            val_loss += model.cross_entropy_loss(val_preds, model.to_one_hot(y_val_batch, model.output_dim))
            num_val_batches += 1
        val_loss /= num_val_batches
        
        # 打印进度
        if epoch % 10 == 0:
            # 分批计算训练准确率
            train_correct = 0
            for i in range(0, len(X_train), batch_size):
                end = min(i + batch_size, len(X_train))
                X_train_batch = X_train[i:end]
                y_train_batch = y_train[i:end]
                train_correct += np.sum(model.predict(X_train_batch) == y_train_batch)
            train_acc = train_correct / len(y_train)
            
            # 分批计算验证准确率
            val_correct = 0
            for i in range(0, len(X_val), batch_size):
                end = min(i + batch_size, len(X_val))
                X_val_batch = X_val[i:end]
                y_val_batch = y_val[i:end]
                val_correct += np.sum(model.predict(X_val_batch) == y_val_batch)
            val_acc = val_correct / len(y_val)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        # 记录历史
        if compute_history:
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
    
    if compute_history:
        return history
    return None


def evaluate_model(model, X, y, batch_size=1000):
    """
    评估模型，使用分批处理避免内存溢出
    
    Args:
        model: 模型实例
        X: 特征
        y: 标签
        batch_size: 批次大小
        
    Returns:
        准确率
    """
    correct = 0
    for i in range(0, len(X), batch_size):
        end = min(i + batch_size, len(X))
        X_batch = X[i:end]
        y_batch = y[i:end]
        y_pred_batch = model.predict(X_batch)
        correct += np.sum(y_pred_batch == y_batch)
    
    accuracy = correct / len(y)
    return accuracy 