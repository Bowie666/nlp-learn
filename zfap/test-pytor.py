import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd

# 加载Bert tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 加载数据集
class SentimentDataset(Dataset):
    def __init__(self, file_path):
        self.sentences = []
        self.labels = []
        data = pd.read_csv(file_path)
        for _, row in data.iterrows():
            self.sentences.append(row['sentence'])
            self.labels.append(row['label'])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

dataset = SentimentDataset('sentiment_data.csv')

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 定义dataloader
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
num_epochs = 5
for epoch in range(num_epochs):
    # 训练
    model.train()
    for batch in train_dataloader:
        sentences, labels = batch
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    correct = 0
    total = 0
    for batch in val_dataloader:
        sentences, labels = batch
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Epoch {epoch+1} --- Validation Accuracy: {accuracy}")


### 第二波


import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')

# 定义BiLSTM模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda()
        out, _ = self.bilstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 初始化模型
model = BiLSTM(input_size=768, hidden_size=128, num_layers=2, num_classes=2).cuda()


# 定义超参数
batch_size = 32
learning_rate = 2e-5
num_epochs = 10

# 加载数据集
train_dataset = ...
test_dataset = ...
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (texts, labels) in enumerate(train_loader):
        # 将文本转换为BERT的输入格式
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to('cuda')
        
        # 运行BERT模型，获取文本的表示
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state
        
        # 运行BiLSTM模型，获取分类结果
        outputs = model(embeddings)
        
        # 计算损失函数并进行反向传播
        labels = labels.to('cuda')
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 输出训练信息
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# 测试模型
with torch.no_grad():
    correct = 0
    total = 0
    for texts, labels in test_loader:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to('cuda')
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state
        outputs = model(embeddings)
        _, predicted = torch.max(outputs.data, 1)
        labels = labels.to('cuda')
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))


BERT-base-chinese是一个基于中文的BERT模型，其参数信息如下：

vocab_size：中文BERT模型的词汇表大小为21128，即模型中包含了21128个不同的中文词汇。

hidden_size：中文BERT模型的隐藏层大小为768，即每个隐藏层包含768个神经元。

num_hidden_layers：中文BERT模型的隐藏层数为12，即模型中包含12个隐藏层。

num_attention_heads：中文BERT模型的注意力头数为12，即每个隐藏层中包含12个注意力头。

intermediate_size：中文BERT模型中间层大小为3072，即每个隐藏层的中间层包含3072个神经元。

hidden_act：中文BERT模型的激活函数为gelu，即Gaussian Error Linear Units。

hidden_dropout_prob：中文BERT模型的隐藏层dropout概率为0.1，即在训练过程中，每个神经元有10%的概率被随机忽略。

attention_probs_dropout_prob：中文BERT模型的注意力dropout概率为0.1，即在训练过程中，每个注意力头有10%的概率被随机忽略。

max_position_embeddings：中文BERT模型的最大序列长度为512，即模型可以处理最长512个中文词汇组成的序列。

type_vocab_size：中文BERT模型的类型词汇表大小为2，即模型可以处理两种不同类型的输入序列（例如，句子A和句子B）。

initializer_range：中文BERT模型的初始化范围为0.02，即模型中的参数在初始化时随机采样的范围为[-0.02, 0.02]。

总体来说，中文BERT模型的参数信息与英文BERT模型基本相同，只是在词汇表大小和类型词汇表大小上有所不同。这些参数信息决定了中文BERT模型的能力和性能，在自然语言处理任务中发挥着重要的作用。



import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载BERT tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 准备训练和测试数据
train_texts = ["我很喜欢这部电影。", "这本书很无聊。"]
train_labels = [1, 0]
test_texts = ["这家餐厅的菜很好吃。", "我不太喜欢这个游戏。"]

# 对文本进行预处理
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# 将数据转化为PyTorch张量
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                               torch.tensor(train_encodings['attention_mask']),
                                               torch.tensor(train_labels))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']),
                                              torch.tensor(test_encodings['attention_mask']))

# 定义训练和测试函数
def train(model, train_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(3):
        for batch in train_loader:
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

def test(model, test_dataset):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0]
            attention_mask = batch[1]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            print(predictions)

# 训练和测试模型
train(model, train_dataset)
test(model, test_dataset)



import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

import random
import numpy as np
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train(model, train_data_loader, val_data_loader, epochs, optimizer, device):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_steps = 0
        
        for batch in train_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_steps += 1
        
        avg_loss = total_loss / total_steps
        print('Epoch:', epoch+1, 'Train loss:', avg_loss)
        
        val_loss, val_accuracy = evaluate(model, val_data_loader, device)
        print('Epoch:', epoch+1, 'Val loss:', val_loss, 'Val accuracy:', val_accuracy)
    
    print('Training finished.')


def evaluate(model, data_loader, device):
    model.eval()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            _, predicted_labels = torch.max(logits, 1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


RANDOM_SEED = 42
MAX_LEN = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 加载数据集，texts为文本列表，labels为对应的情感标签列表
texts = ['文本1', '文本2', ...]
labels = [0, 1, ...]  # 0代表负向情感，1代表正向情感

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=RANDOM_SEED)

# 初始化tokenizer和模型
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# 准备数据加载器
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将模型和数据加载器移至设备上
model = model.to(device)
train_data_loader = train_data_loader.to(device)
val_data_loader = val_data_loader.to(device)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

train(model, train_data_loader, val_data_loader, NUM_EPOCHS, optimizer, device)






