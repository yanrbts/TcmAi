from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# 加载预训练的 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 对清洗后的数据进行编码
def encode_texts(texts, tokenizer, max_length=128):
    return tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='tf')

# 读取已清洗的文本数据
with open('../puredata/cleaned_data.txt', 'r', encoding='utf-8') as file:
    cleaned_data = file.readlines()

# 假设标签全为0，你需要根据实际任务替换
labels = tf.zeros(len(cleaned_data))

# 编码文本数据
encoded_data = encode_texts(cleaned_data, tokenizer)

# 创建 TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(encoded_data),
    labels
)).batch(16)

# 加载预训练的 BERT 模型
model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 训练模型
model.fit(train_dataset, epochs=3)

# 保存模型和 Tokenizer
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')