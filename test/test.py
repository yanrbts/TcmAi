import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载模型和分词器
tokenizer = T5Tokenizer.from_pretrained("../train/trained_model")
model = T5ForConditionalGeneration.from_pretrained("../train/trained_model")

# 提问并获取回答
def ask_question(question, model, tokenizer, max_length=50):
    inputs = tokenizer.encode("问答: " + question, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 示例提问
question = "子午流注说难作者是谁"
answer = ask_question(question, model, tokenizer)
print("Answer:", answer)
