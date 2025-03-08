import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from unsloth import FastLanguageModel
import numpy as np

def load_couplets_dataset(data_path):
    """加载对联数据集"""
    dataset = load_dataset("json", data_files=data_path)
    return dataset

def preprocess_function(examples, tokenizer, max_length=128):
    """预处理数据集"""
    # 将上下联组合成训练文本
    texts = [f"上联：{up}\n下联：{down}" for up, down in zip(examples["up"], examples["down"])]
    
    # tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    return tokenized

def main():
    # 配置参数
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    data_path = "data/couplets.json"  # 对联数据集路径
    output_dir = "output/couplets_model"
    
    # 加载模型和tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=128,
        dtype=None,  # 自动选择dtype
        load_in_4bit=True,  # 使用4bit量化
    )
    
    # 加载数据集
    dataset = load_couplets_dataset(data_path)
    
    # 数据预处理
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # 创建trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main() 