from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
import os

# 设置环境变量以启用 Flash Attention 2
os.environ["USE_FLASH_ATTENTION"] = "1"

# 加载数据集
dataset = load_dataset("yuebanlaosiji/e-girl")

# 初始化模型和tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,  # 自动选择最佳dtype
    load_in_4bit=True,  # 使用4bit量化
)

# 准备训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    optim="adamw_torch",
)

# 准备训练数据
def preprocess_function(examples):
    # 组合instruction和input
    prompts = [
        f"Instruction: {instruction}\nInput: {input_text}\nOutput: "
        for instruction, input_text in zip(examples["instruction"], examples["input"])
    ]
    targets = examples["output"]
    
    # 编码
    model_inputs = tokenizer(prompts, truncation=True, padding=True, max_length=512)
    labels = tokenizer(targets, truncation=True, padding=True, max_length=512)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 处理数据集
processed_dataset = dataset["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# 开始训练
trainer = FastLanguageModel.get_trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=processed_dataset,
)

trainer.train()

# 保存模型
model.save_pretrained("./e-girl-qwen-model")
tokenizer.save_pretrained("./e-girl-qwen-model") 