from transformers import BertTokenizer, BertForSequenceClassification

# 加载训练好的模型
model = BertForSequenceClassification.from_pretrained('./trained_model')
tokenizer = BertTokenizer.from_pretrained('./trained_model')

# 预测新文本
new_text = "新的测试文本"
inputs = tokenizer(new_text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1)
print("Prediction label:", prediction.item())