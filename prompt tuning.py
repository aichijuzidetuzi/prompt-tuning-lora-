"""
原理：在输入前添加可训练的软提示（Soft Prompt），
通过调整提示向量引导模型输出目标结果。模型主体参数冻结，仅训练提示部分
特点：参数高效，适合少样本场景
"""
from datasets import load_dataset

dataset = load_dataset("ag_news")
print(dataset["train"][0])  # 输出示例：{'text': '...', 'label': 3}
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss

# 自定义模型：添加可训练提示向量
class PromptTuningModel(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased", prompt_length=10):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt_embeddings = torch.nn.Parameter(
            torch.randn(prompt_length, self.model.config.hidden_size)
        )
        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        # 获取输入嵌入
        inputs_embeds = self.model.bert.embeddings.word_embeddings(input_ids)
        # 拼接提示向量
        batch_size = inputs_embeds.size(0)
        prompt = self.prompt_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)
        # 调整attention_mask
        attention_mask = torch.cat([
            torch.ones(batch_size, prompt.size(1), device=inputs_embeds.device),
            attention_mask
        ], dim=1)
        # 前向传播
        outputs = self.model.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = self.model.classifier(outputs.pooler_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 4), labels.view(-1))
        return {"loss": loss, "logits": logits}

# 训练参数
training_args = TrainingArguments(
    output_dir="./prompt_tuning",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=1e-4,
)

# 训练器
model = PromptTuningModel()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].select(range(1000)),  # 示例简化
    eval_dataset=dataset["test"].select(range(100)),
)
trainer.train()