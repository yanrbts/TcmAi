import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# 自定义数据集类
class CustomTextDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer, max_length=64):
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.outputs[idx]
        
        # Tokenize the input and target text
        input_encoding = self.tokenizer(
            input_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100  # Mask padding tokens for loss calculation
        
        return {
            "input_ids": input_encoding["input_ids"].flatten(),
            "attention_mask": input_encoding["attention_mask"].flatten(),
            "labels": labels.flatten()
        }

    def __len__(self):
        return len(self.inputs)

# 加载数据
def load_data():
    with open('../purse/metadata.json', 'r', encoding='utf-8') as file:
        metadata_data = json.load(file)
    return metadata_data

# 准备训练数据（示例性问题-答案对）
data = load_data()
answers = [content for book in data for content in book.values()]  # 示例性答案
questions = ["这本书的作者是谁？"] * len(answers)  # 示例性问题，按实际数据替换


# 划分训练集和验证集
train_questions, val_questions, train_answers, val_answers = train_test_split(questions, answers, test_size=0.2)

# 加载 T5 Tokenizer 和模型
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# 创建数据集
train_dataset = CustomTextDataset(train_questions, train_answers, tokenizer)
val_dataset = CustomTextDataset(val_questions, val_answers, tokenizer)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
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

# 使用模型进行问答
def ask_question(question, model, tokenizer, max_length=50):
    inputs = tokenizer.encode("问答: " + question, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 示例提问
question = "这本书的作者是谁？"
answer = ask_question(question, model, tokenizer)
print("Answer:", answer)
