import pandas as pd
train_sample = pd.read_csv('data/train.tsv', sep='\t', nrows=5)
print("训练集示例:")
print(train_sample)
print("\n训练集列名:", train_sample.columns.tolist())
test_sample = pd.read_csv('data/test.tsv', sep='\t', nrows=5)
print("\n测试集示例:")
print(test_sample)
print("\n测试集列名:", test_sample.columns.tolist()) 