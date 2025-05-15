"""
原理：通过低秩分解在原始权重矩阵旁添加适配器（Adapter），仅训练低秩矩阵，冻结原模型参数。公式：W_new = W + A*B，其中A、B为低秩矩阵。

特点：高效微调，适配多种任务。
"""
from datasets import load_dataset

dataset = load_dataset("ag_news")
print(dataset["train"][0])  # 输出示例：{'text': '...', 'label': 3}
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

# LoRA配置
lora_config = LoraConfig(
    r=8,  # 低秩维度
    lora_alpha=16,
    target_modules=["query", "value"],  # 修改注意力层的query和value
    lora_dropout=0.1,
    bias="none",
)

# 加载模型并注入LoRA
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 输出可训练参数占比（通常<1%）

# 训练
training_args = TrainingArguments(
    output_dir="./lora_finetuning",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    learning_rate=1e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].select(range(1000)),
    eval_dataset=dataset["test"].select(range(100)),
)
trainer.train()