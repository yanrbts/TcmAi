# Copyright 2024 yanruibing@gmail.com All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 自定义数据集类
class CustomTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 加载清洗后的章节和药方数据
def load_data():
    with open('../purse/chapters.json', 'r', encoding='utf-8') as file:
        chapters_data = json.load(file)
    with open('../purse/prescriptions.json', 'r', encoding='utf-8') as file:
        prescriptions_data = json.load(file)
    with open('../purse/metadata.json', 'r', encoding='utf-8') as file:
        metadata_data = json.load(file)
    return chapters_data, prescriptions_data, metadata_data

# 准备训练数据
chapters_data, prescriptions_data, metadata_data = load_data()
texts = [content for book in chapters_data for content in book.values()]
texts += [content for book in prescriptions_data for content in book.values()]
texts += [content for book in metadata_data for content in book.values()]

# 二分类标签示例（0 和 1），根据实际任务设置
labels = [1] * len(texts)  # 示例标签，可以自行修改

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# 加载 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 对文本进行编码
def encode_texts(texts, tokenizer, max_length=128):
    return tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

train_encoded = encode_texts(train_texts, tokenizer)
val_encoded = encode_texts(val_texts, tokenizer)

# 创建数据集和数据加载器
train_dataset = CustomTextDataset(train_encoded, train_labels)
val_dataset = CustomTextDataset(val_encoded, val_labels)

# 加载预训练模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    logging_dir='./logs'
)

# 使用 Trainer API 进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# 保存训练好的模型
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')