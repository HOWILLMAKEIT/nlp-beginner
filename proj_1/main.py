import numpy as np
from src.data_loader import DataLoader
from src.feature_extractor import NGramExtractor
from src.model import SentimentClassifier, SGD, MBGD, BGD, train_model, evaluate_model
import wandb  # 导入wandb库

# 用于观察大概多少epoch收敛
def experiment0():
    # 初始化wandb项目
    wandb.init(project="sentiment-analysis", name="experiment0", config={
        "optimizer": "MBGD",
        "learning_rate": 0.1,
        "batch_size": 32,
        "epochs": 50
    })
    
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
    
    # 记录特征维度到wandb
    wandb.config.update({"feature_dimension": feature_extractor.vocab_size})
    
    # MBGD优化器
    print("训练单个MBGD模型测试...")
    mbgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=None, output_dim=5)
    mbgd_optimizer = MBGD(learning_rate=0.1, batch_size=32)
    mbgd_history = train_model(mbgd_model, mbgd_optimizer, train_features, train_labels, val_features, val_labels, epochs=50, compute_history=True)
    mbgd_train_accuracy = evaluate_model(mbgd_model, train_features, train_labels)
    mbgd_val_accuracy = evaluate_model(mbgd_model, val_features, val_labels)
    print(f"MBGD训练准确率: {mbgd_train_accuracy:.4f}, 验证准确率: {mbgd_val_accuracy:.4f}")
    
    # 记录最终结果到wandb
    wandb.log({"final_train_accuracy": mbgd_train_accuracy, "final_val_accuracy": mbgd_val_accuracy})
    
    # 将每一轮的损失记录到wandb
    for epoch, (train_loss, val_loss) in enumerate(zip(mbgd_history['train_loss'], mbgd_history['val_loss'])):
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
    
    # 完成wandb实验
    wandb.finish()

def experiment1():
    # 初始化wandb项目
    wandb.init(project="sentiment-analysis", name="experiment1", config={
        "epochs": 30,
        "feature_extractor": "NGram",
        "dimension": 2,
        "min_freq": 10,
        "max_features": 5000
    })
    
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
    
    # 更新wandb配置
    wandb.config.update({"feature_dimension": feature_extractor.vocab_size})

    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0,10.0,100.0,1000.0]
    
    # 记录每种优化器在不同学习率下的训练准确率和验证准确率
    sgd_train_accuracies = []
    sgd_val_accuracies = []
    mbgd_train_accuracies = []
    mbgd_val_accuracies = []
    bgd_train_accuracies = []
    bgd_val_accuracies = []
    
    # 创建wandb表格来记录所有结果
    results_table = wandb.Table(columns=["learning_rate", "optimizer", "train_accuracy", "val_accuracy"])
    
    # 创建自定义图表的数据
    train_accuracy_data = []
    val_accuracy_data = []
    
    # 对每个学习率进行实验
    for lr in learning_rates:
        print(f"\n学习率: {lr}")
        
        # SGD优化器
        print(f"使用SGD优化器训练，学习率: {lr}...")
        sgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=None, output_dim=5)
        sgd_optimizer = SGD(learning_rate=lr)
        sgd_history = train_model(sgd_model, sgd_optimizer, train_features, train_labels, val_features, val_labels, epochs=30, compute_history=True)
        sgd_train_accuracy = evaluate_model(sgd_model, train_features, train_labels)
        sgd_val_accuracy = evaluate_model(sgd_model, val_features, val_labels)
        sgd_train_accuracies.append(sgd_train_accuracy)
        sgd_val_accuracies.append(sgd_val_accuracy)
        print(f"SGD训练准确率: {sgd_train_accuracy:.4f}, 验证准确率: {sgd_val_accuracy:.4f}")
        
        # 将SGD结果添加到表格
        results_table.add_data(lr, "SGD", sgd_train_accuracy, sgd_val_accuracy)
        
        # 记录SGD每轮训练损失
        for epoch, (train_loss, val_loss) in enumerate(zip(sgd_history['train_loss'], sgd_history['val_loss'])):
            wandb.log({
                "optimizer": "SGD",
                "learning_rate": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch
            })
        
        # MBGD优化器
        print(f"使用MBGD优化器训练，学习率: {lr}...")
        mbgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=None, output_dim=5)
        mbgd_optimizer = MBGD(learning_rate=lr, batch_size=32)
        mbgd_history = train_model(mbgd_model, mbgd_optimizer, train_features, train_labels, val_features, val_labels, epochs=30, compute_history=True)
        mbgd_train_accuracy = evaluate_model(mbgd_model, train_features, train_labels)
        mbgd_val_accuracy = evaluate_model(mbgd_model, val_features, val_labels)
        mbgd_train_accuracies.append(mbgd_train_accuracy)
        mbgd_val_accuracies.append(mbgd_val_accuracy)
        print(f"MBGD训练准确率: {mbgd_train_accuracy:.4f}, 验证准确率: {mbgd_val_accuracy:.4f}")
        
        # 将MBGD结果添加到表格
        results_table.add_data(lr, "MBGD", mbgd_train_accuracy, mbgd_val_accuracy)
        
        # 记录MBGD每轮训练损失
        for epoch, (train_loss, val_loss) in enumerate(zip(mbgd_history['train_loss'], mbgd_history['val_loss'])):
            wandb.log({
                "optimizer": "MBGD",
                "learning_rate": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch
            })
        
        # BGD优化器
        print(f"使用BGD优化器训练，学习率: {lr}...")
        bgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=None, output_dim=5)
        bgd_optimizer = BGD(learning_rate=lr)
        bgd_history = train_model(bgd_model, bgd_optimizer, train_features, train_labels, val_features, val_labels, epochs=30, compute_history=True)
        bgd_train_accuracy = evaluate_model(bgd_model, train_features, train_labels)
        bgd_val_accuracy = evaluate_model(bgd_model, val_features, val_labels)
        bgd_train_accuracies.append(bgd_train_accuracy)
        bgd_val_accuracies.append(bgd_val_accuracy)
        print(f"BGD训练准确率: {bgd_train_accuracy:.4f}, 验证准确率: {bgd_val_accuracy:.4f}")
        
        # 将BGD结果添加到表格
        results_table.add_data(lr, "BGD", bgd_train_accuracy, bgd_val_accuracy)
        
        # 记录BGD每轮训练损失
        for epoch, (train_loss, val_loss) in enumerate(zip(bgd_history['train_loss'], bgd_history['val_loss'])):
            wandb.log({
                "optimizer": "BGD",
                "learning_rate": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch
            })
        
        # 收集数据用于后续绘制学习率与准确率的关系图
        train_accuracy_data.append([lr, sgd_train_accuracy, mbgd_train_accuracy, bgd_train_accuracy])
        val_accuracy_data.append([lr, sgd_val_accuracy, mbgd_val_accuracy, bgd_val_accuracy])
    
    # 记录结果表格
    wandb.log({"results": results_table})
    
    # 创建自定义图表比较不同学习率下的性能
    train_accuracy_table = wandb.Table(data=train_accuracy_data, 
                                      columns=["learning_rate", "SGD", "MBGD", "BGD"])
    
    val_accuracy_table = wandb.Table(data=val_accuracy_data, 
                                    columns=["learning_rate", "SGD", "MBGD", "BGD"])
    
    # 创建学习率与训练准确率的自定义图表
    wandb.log({"training_accuracy_vs_lr": wandb.plot.line(
        train_accuracy_table, 
        "learning_rate", 
        ["SGD", "MBGD", "BGD"],
        title="不同优化器的训练准确率随学习率变化")
    })
    
    # 创建学习率与验证准确率的自定义图表
    wandb.log({"validation_accuracy_vs_lr": wandb.plot.line(
        val_accuracy_table, 
        "learning_rate", 
        ["SGD", "MBGD", "BGD"],
        title="不同优化器的验证准确率随学习率变化")
    })
    
    # 输出结果数据
    print("\n学习率与训练准确率数据:")
    for i, lr in enumerate(learning_rates):
        print(f"学习率: {lr}, SGD: {sgd_train_accuracies[i]:.4f}, MBGD: {mbgd_train_accuracies[i]:.4f}, BGD: {bgd_train_accuracies[i]:.4f}")
    
    print("\n学习率与验证准确率数据:")
    for i, lr in enumerate(learning_rates):
        print(f"学习率: {lr}, SGD: {sgd_val_accuracies[i]:.4f}, MBGD: {mbgd_val_accuracies[i]:.4f}, BGD: {bgd_val_accuracies[i]:.4f}")
    
    # 完成wandb实验
    wandb.finish()

def experiment2():
    # 初始化wandb项目
    wandb.init(project="sentiment-analysis", name="experiment2", config={
        "optimizer": "MBGD",
        "learning_rate": 0.1,
        "batch_size": 32,
        "epochs": 30,
        "min_freq": 10,
        "max_features": 15000
    })
    
    np.random.seed(20)
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_data()
    train_texts, train_labels, val_texts, val_labels = data_loader.split_train_val(val_ratio=0.2)
    print(f"训练数据: {len(train_texts)}个样本, 验证数据: {len(val_texts)}个样本")
    
    n_gram_dimensions = [1, 2, 3]  # 要测试的n-gram维度
    
    # 存储所有n-gram维度的训练和验证损失历史以及准确率
    all_train_losses = {}
    all_val_losses = {}
    all_train_accs = {}
    all_val_accs = {}
    all_feature_dims = {}
    
    # 对每个n-gram维度进行实验
    for n_gram in n_gram_dimensions:
        print(f"\nn-gram维度: {n_gram}")
        
        # 特征提取
        feature_extractor = NGramExtractor(dimension=n_gram, min_freq=10, max_features=15000, use_cumulative=True)  
        train_features = feature_extractor.fit_transform(train_texts)
        val_features = feature_extractor.transform(val_texts)
        print(f"特征维度: {feature_extractor.vocab_size}")
        
        # MBGD优化器
        print(f"使用MBGD优化器训练，n-gram维度: {n_gram}...")
        mbgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=None, output_dim=5)
        mbgd_optimizer = MBGD(learning_rate=0.1, batch_size=32)
        
        # 使用原始的train_model函数进行训练并获取历史记录
        model_name = f"{n_gram}-gram"
        mbgd_history = train_model(mbgd_model, mbgd_optimizer, train_features, train_labels, 
                                   val_features, val_labels, epochs=30, compute_history=True)
        
        # 将每个epoch的训练和验证损失记录到wandb
        for epoch, (train_loss, val_loss) in enumerate(zip(mbgd_history['train_loss'], mbgd_history['val_loss'])):
            wandb.log({
                "epoch": epoch,
                f"{model_name}/train_loss": train_loss,
                f"{model_name}/val_loss": val_loss
            })
        
        # 评估最终准确率
        mbgd_train_accuracy = evaluate_model(mbgd_model, train_features, train_labels)
        mbgd_val_accuracy = evaluate_model(mbgd_model, val_features, val_labels)
        print(f"n-gram={n_gram} 训练准确率: {mbgd_train_accuracy:.4f}, 验证准确率: {mbgd_val_accuracy:.4f}")
        
        # 存储训练和验证损失历史及准确率
        all_train_losses[model_name] = mbgd_history['train_loss']
        all_val_losses[model_name] = mbgd_history['val_loss']
        all_train_accs[model_name] = mbgd_train_accuracy
        all_val_accs[model_name] = mbgd_val_accuracy
        all_feature_dims[model_name] = feature_extractor.vocab_size
        
        # 记录最终结果
        wandb.log({
            f"{model_name}/final_train_accuracy": mbgd_train_accuracy,
            f"{model_name}/final_val_accuracy": mbgd_val_accuracy,
            f"{model_name}/feature_dimension": feature_extractor.vocab_size
        })
    
    # 创建结果表格
    results_data = []
    for n_gram in n_gram_dimensions:
        model_name = f"{n_gram}-gram"
        results_data.append([
            n_gram, 
            all_feature_dims[model_name], 
            all_train_accs[model_name], 
            all_val_accs[model_name]
        ])
    
    results_table = wandb.Table(
        columns=["n_gram", "feature_dimension", "train_accuracy", "val_accuracy"],
        data=results_data
    )
    wandb.log({"n_gram_results": results_table})
    
    # 创建准确率对比条形图
    train_acc_data = []
    for n_gram in n_gram_dimensions:
        model_name = f"{n_gram}-gram"
        train_acc_data.append([model_name, all_train_accs[model_name]])
    
    train_acc_table = wandb.Table(
        columns=["模型", "准确率"],
        data=train_acc_data
    )
    
    wandb.log({"训练准确率对比": wandb.plot.bar(
        train_acc_table,
        "模型",
        "准确率",
        title="不同n-gram维度的训练准确率对比")
    })
    
    # 创建验证准确率对比条形图
    val_acc_data = []
    for n_gram in n_gram_dimensions:
        model_name = f"{n_gram}-gram"
        val_acc_data.append([model_name, all_val_accs[model_name]])
    
    val_acc_table = wandb.Table(
        columns=["模型", "准确率"],
        data=val_acc_data
    )
    
    wandb.log({"验证准确率对比": wandb.plot.bar(
        val_acc_table,
        "模型",
        "准确率",
        title="不同n-gram维度的验证准确率对比")
    })
    
    # 创建n-gram维度与准确率关系图
    acc_vs_ngram_data = []
    for n_gram in n_gram_dimensions:
        model_name = f"{n_gram}-gram"
        acc_vs_ngram_data.append([
            n_gram, 
            all_train_accs[model_name], 
            all_val_accs[model_name]
        ])
    
    acc_vs_ngram_table = wandb.Table(
        columns=["n_gram", "训练准确率", "验证准确率"],
        data=acc_vs_ngram_data
    )
    
    wandb.log({"准确率随n-gram变化": wandb.plot.line(
        acc_vs_ngram_table,
        "n_gram",
        ["训练准确率", "验证准确率"],
        title="准确率随n-gram维度变化")
    })
    
    # 创建综合准确率对比条形图
    combined_acc_data = [
        ["训练", all_train_accs["1-gram"], all_train_accs["2-gram"], all_train_accs["3-gram"]],
        ["验证", all_val_accs["1-gram"], all_val_accs["2-gram"], all_val_accs["3-gram"]]
    ]
    
    combined_acc_table = wandb.Table(
        columns=["数据集", "1-gram", "2-gram", "3-gram"],
        data=combined_acc_data
    )
    
    wandb.log({"准确率综合对比": wandb.plot.bar(
        combined_acc_table,
        "数据集",
        ["1-gram", "2-gram", "3-gram"],
        title="不同n-gram在训练集和验证集上的准确率对比")
    })
    
    wandb.finish()

def experiment3():
    # 初始化wandb项目
    wandb.init(project="sentiment-analysis", name="experiment3", config={
        "optimizer": "MBGD",
        "learning_rate": 0.1,
        "batch_size": 32,
        "epochs": 50,
        "n_gram": 2,
        "min_freq": 10,
        "max_features": 15000
    })
    
    np.random.seed(20)
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_data()
    train_texts, train_labels, val_texts, val_labels = data_loader.split_train_val(val_ratio=0.2)
    print(f"训练数据: {len(train_texts)}个样本, 验证数据: {len(val_texts)}个样本")

    # 特征提取 - 使用2-gram
    feature_extractor = NGramExtractor(dimension=2, min_freq=10, max_features=15000)  
    train_features = feature_extractor.fit_transform(train_texts)
    val_features = feature_extractor.transform(val_texts)
    print(f"特征维度: {feature_extractor.vocab_size}")
    
    # 更新wandb配置
    wandb.config.update({"feature_dimension": feature_extractor.vocab_size})
    
    # 存储两种模型的训练和验证损失历史以及准确率
    all_train_losses = {}
    all_val_losses = {}
    all_train_accs = {}
    all_val_accs = {}
    model_types = ["原始模型", "改进模型"]
    
    # 模型1: 仅全连接层 (原始模型)
    print("\n训练原始模型 (仅全连接层)...")
    original_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=None, output_dim=5)
    original_optimizer = MBGD(learning_rate=0.1, batch_size=32)
    
    # 使用原始的train_model函数进行训练并获取历史记录
    original_history = train_model(original_model, original_optimizer, train_features, train_labels, 
                                   val_features, val_labels, epochs=50, compute_history=True)
    
    # 将每个epoch的训练和验证损失记录到wandb
    for epoch, (train_loss, val_loss) in enumerate(zip(original_history['train_loss'], original_history['val_loss'])):
        wandb.log({
            "epoch": epoch,
            "原始模型/train_loss": train_loss,
            "原始模型/val_loss": val_loss
        })
    
    original_train_accuracy = evaluate_model(original_model, train_features, train_labels)
    original_val_accuracy = evaluate_model(original_model, val_features, val_labels)
    print(f"原始模型训练准确率: {original_train_accuracy:.4f}, 验证准确率: {original_val_accuracy:.4f}")
    
    # 存储原始模型的损失历史和准确率
    all_train_losses["原始模型"] = original_history['train_loss']
    all_val_losses["原始模型"] = original_history['val_loss']
    all_train_accs["原始模型"] = original_train_accuracy
    all_val_accs["原始模型"] = original_val_accuracy
    
    # 记录最终结果
    wandb.log({
        "原始模型/final_train_accuracy": original_train_accuracy,
        "原始模型/final_val_accuracy": original_val_accuracy
    })
    
    # 模型2: 带有ReLU激活函数和隐藏层的模型
    print("\n训练加入隐藏层 (带ReLU和隐藏层)...")
    improved_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=100, output_dim=5)
    improved_optimizer = MBGD(learning_rate=0.1, batch_size=32)
    
    # 使用原始的train_model函数进行训练并获取历史记录
    improved_history = train_model(improved_model, improved_optimizer, train_features, train_labels, 
                                   val_features, val_labels, epochs=50, compute_history=True)
    
    # 将每个epoch的训练和验证损失记录到wandb
    for epoch, (train_loss, val_loss) in enumerate(zip(improved_history['train_loss'], improved_history['val_loss'])):
        wandb.log({
            "epoch": epoch,
            "改进模型/train_loss": train_loss,
            "改进模型/val_loss": val_loss
        })
    
    improved_train_accuracy = evaluate_model(improved_model, train_features, train_labels)
    improved_val_accuracy = evaluate_model(improved_model, val_features, val_labels)
    print(f"加入隐藏层训练准确率: {improved_train_accuracy:.4f}, 验证准确率: {improved_val_accuracy:.4f}")
    
    # 存储改进模型的损失历史和准确率
    all_train_losses["改进模型"] = improved_history['train_loss']
    all_val_losses["改进模型"] = improved_history['val_loss']
    all_train_accs["改进模型"] = improved_train_accuracy
    all_val_accs["改进模型"] = improved_val_accuracy
    
    # 记录最终结果
    wandb.log({
        "改进模型/final_train_accuracy": improved_train_accuracy,
        "改进模型/final_val_accuracy": improved_val_accuracy
    })
    
    # 创建结果表格
    results_table = wandb.Table(
        columns=["model_type", "train_accuracy", "val_accuracy"],
        data=[
            ["原始模型 (仅全连接层)", original_train_accuracy, original_val_accuracy],
            ["加入隐藏层 (带ReLU和隐藏层)", improved_train_accuracy, improved_val_accuracy]
        ]
    )
    wandb.log({"model_comparison_table": results_table})
    
    # 创建训练准确率对比条形图
    train_acc_data = [[model_type, acc] for model_type, acc in all_train_accs.items()]
    train_acc_table = wandb.Table(
        columns=["模型", "准确率"],
        data=train_acc_data
    )
    
    wandb.log({"训练准确率对比": wandb.plot.bar(
        train_acc_table,
        "模型",
        "准确率",
        title="不同模型的训练准确率对比")
    })
    
    # 创建验证准确率对比条形图
    val_acc_data = [[model_type, acc] for model_type, acc in all_val_accs.items()]
    val_acc_table = wandb.Table(
        columns=["模型", "准确率"],
        data=val_acc_data
    )
    
    wandb.log({"验证准确率对比": wandb.plot.bar(
        val_acc_table,
        "模型",
        "准确率",
        title="不同模型的验证准确率对比")
    })
    
    # 创建综合准确率对比条形图（既包含训练又包含验证）
    combined_acc_data = [
        ["训练", original_train_accuracy, improved_train_accuracy],
        ["验证", original_val_accuracy, improved_val_accuracy]
    ]
    
    combined_acc_table = wandb.Table(
        columns=["数据集", "原始模型", "改进模型"],
        data=combined_acc_data
    )
    
    wandb.log({"准确率综合对比": wandb.plot.bar(
        combined_acc_table, 
        "数据集", 
        ["原始模型", "改进模型"],
        title="不同模型在训练集和验证集上的准确率对比")
    })
    
    # 完成wandb实验
    wandb.finish()

def experiment4():
    # 初始化wandb项目
    wandb.init(project="sentiment-analysis", name="experiment4", config={
        "optimizer": "MBGD",
        "learning_rate": 0.1,
        "batch_size": 32,
        "epochs": 30,
        "n_gram": 1,
        "min_freq": 10,
        "max_features": 5000
    })
    
    np.random.seed(20)
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_data()
    train_texts, train_labels, val_texts, val_labels = data_loader.split_train_val(val_ratio=0.2)
    print(f"训练数据: {len(train_texts)}个样本, 验证数据: {len(val_texts)}个样本")

    # 存储两种特征表示的训练和验证损失历史以及准确率
    all_train_losses = {}
    all_val_losses = {}
    all_train_accs = {}
    all_val_accs = {}
    all_feature_dims = {}
    feature_types = ["二进制特征", "词频特征"]
    
    # 二进制特征（baseline - 只关注特征是否出现）
    print("\n使用二进制特征表示（baseline）...")
    binary_extractor = NGramExtractor(dimension=1, min_freq=10, max_features=5000, binary=True)  
    binary_train_features = binary_extractor.fit_transform(train_texts)
    binary_val_features = binary_extractor.transform(val_texts)
    print(f"二进制特征维度: {binary_extractor.vocab_size}")
    
    # 训练二进制特征模型
    binary_model = SentimentClassifier(input_dim=binary_extractor.vocab_size, hidden_dim=None, output_dim=5)
    binary_optimizer = MBGD(learning_rate=0.1, batch_size=32)
    
    # 使用原始的train_model函数进行训练并获取历史记录
    binary_history = train_model(binary_model, binary_optimizer, binary_train_features, train_labels, 
                                 binary_val_features, val_labels, epochs=30, compute_history=True)
    
    # 将每个epoch的训练和验证损失记录到wandb
    for epoch, (train_loss, val_loss) in enumerate(zip(binary_history['train_loss'], binary_history['val_loss'])):
        wandb.log({
            "epoch": epoch,
            "二进制特征/train_loss": train_loss,
            "二进制特征/val_loss": val_loss
        })
    
    binary_train_accuracy = evaluate_model(binary_model, binary_train_features, train_labels)
    binary_val_accuracy = evaluate_model(binary_model, binary_val_features, val_labels)
    print(f"二进制特征 - 训练准确率: {binary_train_accuracy:.4f}, 验证准确率: {binary_val_accuracy:.4f}")
    
    # 存储二进制特征的损失历史和准确率
    all_train_losses["二进制特征"] = binary_history['train_loss']
    all_val_losses["二进制特征"] = binary_history['val_loss']
    all_train_accs["二进制特征"] = binary_train_accuracy
    all_val_accs["二进制特征"] = binary_val_accuracy
    all_feature_dims["二进制特征"] = binary_extractor.vocab_size
    
    # 记录最终结果
    wandb.log({
        "二进制特征/final_train_accuracy": binary_train_accuracy,
        "二进制特征/final_val_accuracy": binary_val_accuracy,
        "二进制特征/feature_dimension": binary_extractor.vocab_size
    })
    
    # 词频特征（统计每个n-gram出现的频率）
    print("\n使用词频特征表示...")
    freq_extractor = NGramExtractor(dimension=1, min_freq=10, max_features=5000, binary=False)  
    freq_train_features = freq_extractor.fit_transform(train_texts)
    freq_val_features = freq_extractor.transform(val_texts)
    print(f"词频特征维度: {freq_extractor.vocab_size}")
    
    # 训练词频特征模型
    freq_model = SentimentClassifier(input_dim=freq_extractor.vocab_size, hidden_dim=None, output_dim=5)
    freq_optimizer = MBGD(learning_rate=0.1, batch_size=32)
    
    # 使用原始的train_model函数进行训练并获取历史记录
    freq_history = train_model(freq_model, freq_optimizer, freq_train_features, train_labels, 
                               freq_val_features, val_labels, epochs=30, compute_history=True)
    
    # 将每个epoch的训练和验证损失记录到wandb
    for epoch, (train_loss, val_loss) in enumerate(zip(freq_history['train_loss'], freq_history['val_loss'])):
        wandb.log({
            "epoch": epoch,
            "词频特征/train_loss": train_loss,
            "词频特征/val_loss": val_loss
        })
    
    freq_train_accuracy = evaluate_model(freq_model, freq_train_features, train_labels)
    freq_val_accuracy = evaluate_model(freq_model, freq_val_features, val_labels)
    print(f"词频特征 - 训练准确率: {freq_train_accuracy:.4f}, 验证准确率: {freq_val_accuracy:.4f}")
    
    # 存储词频特征的损失历史和准确率
    all_train_losses["词频特征"] = freq_history['train_loss']
    all_val_losses["词频特征"] = freq_history['val_loss']
    all_train_accs["词频特征"] = freq_train_accuracy
    all_val_accs["词频特征"] = freq_val_accuracy
    all_feature_dims["词频特征"] = freq_extractor.vocab_size
    
    # 记录最终结果
    wandb.log({
        "词频特征/final_train_accuracy": freq_train_accuracy,
        "词频特征/final_val_accuracy": freq_val_accuracy,
        "词频特征/feature_dimension": freq_extractor.vocab_size
    })
    
    # 创建结果表格
    results_table = wandb.Table(
        columns=["feature_type", "feature_dimension", "train_accuracy", "val_accuracy"],
        data=[
            ["二进制特征", all_feature_dims["二进制特征"], all_train_accs["二进制特征"], all_val_accs["二进制特征"]],
            ["词频特征", all_feature_dims["词频特征"], all_train_accs["词频特征"], all_val_accs["词频特征"]]
        ]
    )
    wandb.log({"feature_representation_table": results_table})
    
    # 创建训练准确率对比条形图
    train_acc_data = [[feature_type, acc] for feature_type, acc in all_train_accs.items()]
    train_acc_table = wandb.Table(
        columns=["特征表示", "准确率"],
        data=train_acc_data
    )
    
    wandb.log({"训练准确率对比": wandb.plot.bar(
        train_acc_table,
        "特征表示",
        "准确率",
        title="不同特征表示的训练准确率对比")
    })
    
    # 创建验证准确率对比条形图
    val_acc_data = [[feature_type, acc] for feature_type, acc in all_val_accs.items()]
    val_acc_table = wandb.Table(
        columns=["特征表示", "准确率"],
        data=val_acc_data
    )
    
    wandb.log({"验证准确率对比": wandb.plot.bar(
        val_acc_table,
        "特征表示",
        "准确率",
        title="不同特征表示的验证准确率对比")
    })
    
    # 创建综合准确率对比条形图（既包含训练又包含验证）
    combined_acc_data = [
        ["训练", binary_train_accuracy, freq_train_accuracy],
        ["验证", binary_val_accuracy, freq_val_accuracy]
    ]
    
    combined_acc_table = wandb.Table(
        columns=["数据集", "二进制特征", "词频特征"],
        data=combined_acc_data
    )
    
    wandb.log({"准确率综合对比": wandb.plot.bar(
        combined_acc_table, 
        "数据集", 
        ["二进制特征", "词频特征"],
        title="不同特征表示在训练集和验证集上的准确率对比")
    })

    # 结果比较
    print("\n特征表示方法比较:")
    print(f"二进制特征 - 训练准确率: {binary_train_accuracy:.4f}, 验证准确率: {binary_val_accuracy:.4f}")
    print(f"词频特征 - 训练准确率: {freq_train_accuracy:.4f}, 验证准确率: {freq_val_accuracy:.4f}")
    print(f"训练准确率差异: {freq_train_accuracy - binary_train_accuracy:.4f}")
    print(f"验证准确率差异: {freq_val_accuracy - binary_val_accuracy:.4f}")
    
    # 完成wandb实验
    wandb.finish()

def experiment5():
    # 初始化wandb项目
    wandb.init(project="sentiment-analysis", name="experiment5", config={
        "optimizer": "MBGD",
        "learning_rate": 0.1,
        "batch_size": 32,
        "epochs": 50 ,
        "n_gram": 3,
        "min_freq": 10,
        "max_features": 20000,
        "gram_num": 3,
        "use_cumulative": True,
        "hidden_dim": None,
        "binary": True 
    })

    np.random.seed(20)
    data_loader = DataLoader()
    data_loader.load_data()
    data_loader.preprocess_data()
    train_texts, train_labels, val_texts, val_labels = data_loader.split_train_val(val_ratio=0.2)
    print(f"训练数据: {len(train_texts)}个样本, 验证数据: {len(val_texts)}个样本")

    # 设置最小词频和最大特征数，减少内存占用
    feature_extractor = NGramExtractor(dimension=3, min_freq=10, max_features=20000, binary=True, use_cumulative=True)  
    train_features = feature_extractor.fit_transform(train_texts)
    val_features = feature_extractor.transform(val_texts)
    print(f"特征维度: {feature_extractor.vocab_size}")
    
    # 记录特征维度到wandb
    wandb.config.update({"feature_dimension": feature_extractor.vocab_size})
    
    # MBGD优化器
    print("训练单个MBGD模型测试...")
    mbgd_model = SentimentClassifier(input_dim=feature_extractor.vocab_size, hidden_dim=None, output_dim=5)
    mbgd_optimizer = MBGD(learning_rate=0.1, batch_size=32)
    mbgd_history = train_model(mbgd_model, mbgd_optimizer, train_features, train_labels, val_features, val_labels, epochs=50, compute_history=True)
    mbgd_train_accuracy = evaluate_model(mbgd_model, train_features, train_labels)
    mbgd_val_accuracy = evaluate_model(mbgd_model, val_features, val_labels)
    print(f"MBGD训练准确率: {mbgd_train_accuracy:.4f}, 验证准确率: {mbgd_val_accuracy:.4f}")
    
    # 记录最终结果到wandb
    wandb.log({"final_train_accuracy": mbgd_train_accuracy, "final_val_accuracy": mbgd_val_accuracy})
    
    # 将每一轮的损失记录到wandb
    for epoch, (train_loss, val_loss) in enumerate(zip(mbgd_history['train_loss'], mbgd_history['val_loss'])):
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
    
    # 完成wandb实验
    wandb.finish()



if __name__ == "__main__":
    # experiment0()
    # experiment1()
    # experiment2()
    # experiment3()
    # experiment4()
    experiment5()