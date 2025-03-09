# 导入必要的库
from unsloth import FastLanguageModel  # 导入unsloth的快速语言模型
import torch  # PyTorch深度学习框架
from datasets import load_dataset  # Hugging Face数据集加载工具
from transformers import TrainingArguments, EarlyStoppingCallback  # 训练参数配置和早停回调
import os
import json
from trl import SFTTrainer  # 监督微调训练器
from unsloth import is_bfloat16_supported  # 检查是否支持bfloat16
import logging  # 日志记录
from datetime import datetime
import sys
from huggingface_hub import create_repo  # HuggingFace仓库创建工具

# 设置环境变量以启用 Flash Attention 2（用于加速注意力计算）
os.environ["USE_FLASH_ATTENTION"] = "1"

# 配置日志系统
log_dir = "data_create_dianzinvyou/output/logs"  # 日志保存目录
os.makedirs(log_dir, exist_ok=True)  # 创建日志目录（如果不存在）
log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')  # 生成带时间戳的日志文件名

# 设置日志配置：同时输出到文件和控制台
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式：时间-级别-消息
    handlers=[
        logging.FileHandler(log_file),  # 文件处理器
        logging.StreamHandler(sys.stdout)  # 控制台处理器
    ]
)

# 创建模型输出目录
output_base_dir = "data_create_dianzinvyou/output"  # 基础输出目录
model_output_dir = os.path.join(output_base_dir, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')  # 带时间戳的模型输出目录
os.makedirs(model_output_dir, exist_ok=True)  # 创建输出目录

# 模型训练配置参数
MODEL_CONFIG = {
    "model_name": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",  # 基础模型名称
    "max_seq_length": 2048,  # 最大序列长度
    "per_device_batch_size": 4,  # 每个设备的批次大小
    "gradient_accumulation_steps": 2,  # 减少梯度累积步数
    "num_train_epochs": 3,  # 减少训练轮数
    "learning_rate": 1e-4,  # 降低学习率
    "lora_r": 16,  # LoRA秩
    "lora_alpha": 16,  # LoRA缩放因子
    "lora_dropout": 0.2,  # 增加dropout防止过拟合
    "weight_decay": 0.02,  # 增加权重衰减
    "model_description": "基于DeepSeek-R1-Distill-Qwen-1.5B模型训练的AI女友模型，具有温柔、体贴、善解人意的特点。"
}

# 加载预训练模型和分词器
model_name = MODEL_CONFIG["model_name"]
model, tokenizer = FastLanguageModel.from_pretrained(model_name, trust_remote_code=True)

# 获取结束符标记
EOS_TOKEN = tokenizer.eos_token

# 定义提示模板
PROMPT_TEMPLATE = """你现在是一个温柔、包容、善解人意的女友。你需要以女友的身份回复用户的消息。
    你的性格特点：
    1. 温柔体贴，善于倾听
    2. 积极向上，富有正能量
    3. 会适当撒娇，但不会过分
    4. 懂得关心对方的工作和生活
    5. 会给予对方鼓励和支持

    用户消息: {user_message}

    思考过程:
    <thinking>
    {reasoning_content}
    </thinking>
    女友回复:
    {output_text}
    
    """

def fromat_dataset_func(dataset):
    """
    数据集格式化函数
    Args:
        dataset: 原始数据集
    Returns:
        dict: 格式化后的数据集
    """
    input_texts = dataset['input']  # 用户输入
    output_texts = dataset['output']  # 模型输出
    reasoning_contents = dataset['reasoning_content']  # 推理过程

    text_list = []
    # 组合输入、推理过程和输出
    for input_text, output_text, reasoning_content in zip(input_texts, output_texts, reasoning_contents):
        text = PROMPT_TEMPLATE.format(
            user_message=input_text,
            output_text=output_text,
            reasoning_content=reasoning_content
        ) + EOS_TOKEN
        text_list.append(text)
    return {"text": text_list}

def main():
    """主函数：执行模型训练和保存的完整流程"""
    try:
        # 记录训练配置
        logging.info("开始训练，配置信息如下：")
        for key, value in MODEL_CONFIG.items():
            logging.info(f"{key}: {value}")

        # 检查HuggingFace令牌
        token = os.getenv("HuggingfaceToken")
        if not token:
            raise ValueError("未设置HuggingfaceToken环境变量")

        # 加载数据集
        logging.info("正在加载数据集...")
        dataset = load_dataset("yuebanlaosiji/e-girl", trust_remote_code=True)
        logging.info(f"数据集加载完成，训练样本数量：{len(dataset['train'])}")

        # 格式化数据集
        logging.info("正在格式化数据集...")
        formatted_dataset = dataset['train'].map(fromat_dataset_func, batched=True)
        
        # 分割数据集为训练集和验证集 (90% 训练, 10% 验证)
        train_val_split = formatted_dataset.train_test_split(test_size=0.1, seed=3407)
        train_dataset = train_val_split['train']
        eval_dataset = train_val_split['test']
        logging.info(f"数据集分割完成，训练集大小：{len(train_dataset)}，验证集大小：{len(eval_dataset)}")

        # 加载模型和分词器
        logging.info(f"正在加载模型：{MODEL_CONFIG['model_name']}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_CONFIG['model_name'],
            trust_remote_code=True
        )

        # 准备训练：转换为训练模式并应用LoRA
        logging.info("正在准备模型训练...")
        FastLanguageModel.for_training(model)
        new_model = FastLanguageModel.get_peft_model(
            model,
            r=MODEL_CONFIG['lora_r'],  # LoRA秩
            target_modules=["q_proj", "v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],  # 目标模块
            lora_alpha=MODEL_CONFIG['lora_alpha'],  # 缩放因子
            lora_dropout=MODEL_CONFIG['lora_dropout'],  # Dropout率
            bias="none",  # 不使用偏置
            use_gradient_checkpointing="unsloth",  # 使用梯度检查点
            random_state=3407,  # 随机种子
            use_rslora=False,  # 不使用RSLoRA
            loftq_config=None,  # 不使用LoftQ
        )

        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=model_output_dir,  # 输出目录
            per_device_train_batch_size=MODEL_CONFIG['per_device_batch_size'],  # 每设备批次大小
            per_device_eval_batch_size=MODEL_CONFIG['per_device_batch_size'],  # 评估时的批次大小
            gradient_accumulation_steps=MODEL_CONFIG['gradient_accumulation_steps'],  # 梯度累积步数
            num_train_epochs=MODEL_CONFIG['num_train_epochs'],  # 训练轮数
            warmup_ratio=0.1,  # 预热比例
            learning_rate=MODEL_CONFIG['learning_rate'],  # 学习率
            fp16=not is_bfloat16_supported(),  # 是否使用FP16
            bf16=is_bfloat16_supported(),  # 是否使用BF16
            logging_steps=10,  # 日志记录步数
            save_strategy="epoch",  # 每个epoch保存一次
            evaluation_strategy="epoch",  # 每个epoch评估一次
            save_total_limit=3,  # 保存最近3个检查点
            load_best_model_at_end=True,  # 训练结束后加载最佳模型
            metric_for_best_model="loss",  # 使用损失作为最佳模型指标
            greater_is_better=False,  # 损失值越小越好
            optim="adamw_torch",  # 优化器
            weight_decay=MODEL_CONFIG['weight_decay'],  # 权重衰减
            lr_scheduler_type="cosine",  # 学习率调度器类型
            seed=3407,  # 随机种子
        )

        # 创建训练器
        trainer = SFTTrainer(
            model=new_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # 添加验证集
            dataset_text_field="text",
            tokenizer=tokenizer,
            max_seq_length=MODEL_CONFIG['max_seq_length'],
            dataset_num_proc=4,  # 数据处理进程数
            packing=True,  # 启用序列打包
            args=training_args
        )

        # 开始训练
        logging.info("开始训练模型...")
        status = trainer.train()
        logging.info(f"训练完成，状态：{status}")

        # 保存最终模型
        final_model_path = os.path.join(model_output_dir, "final_model")
        logging.info(f"正在保存最终模型到：{final_model_path}")
        
        # 1. 保存模型权重和配置
        new_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        # 2. 创建详细的README.md
        readme_content = f"""# AI女友模型 - 基于DeepSeek-R1-Distill-Qwen-1.5B

## 模型信息
- 基础模型：{MODEL_CONFIG['model_name']}
- 训练时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 模型描述：{MODEL_CONFIG['model_description']}

## 训练参数
- 最大序列长度：{MODEL_CONFIG['max_seq_length']}
- 批次大小：{MODEL_CONFIG['per_device_batch_size']}
- 训练轮数：{MODEL_CONFIG['num_train_epochs']}
- 学习率：{MODEL_CONFIG['learning_rate']}
- LoRA配置：
  - Rank (r): {MODEL_CONFIG['lora_r']}
  - Alpha: {MODEL_CONFIG['lora_alpha']}
  - Dropout: {MODEL_CONFIG['lora_dropout']}

## 使用方法

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型和分词器
model_name = "yuebanlaosiji/e-girl-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto"
)

# 对话模板
template = '''你现在是一个温柔、包容、善解人意的女友。你需要以女友的身份回复用户的消息。
用户消息: {user_input}
女友回复:'''

# 生成回复
def chat(user_input):
    prompt = template.format(user_input=user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=2048,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("女友回复:")[-1].strip()

# 使用示例
response = chat("今天工作好累啊")
print(response)
```

## 模型特点
1. 温柔体贴，善于倾听
2. 积极向上，富有正能量
3. 会适当撒娇，但不会过分
4. 懂得关心对方的工作和生活
5. 会给予对方鼓励和支持

## 训练数据
- 训练集大小：{len(train_dataset)}
- 验证集大小：{len(eval_dataset)}

## 训练结果
- 最终训练损失：{status.training_loss:.4f}
- 训练用时：{status.metrics['train_runtime']/60:.2f}分钟
- 训练速度：{status.metrics['train_samples_per_second']:.2f} samples/second

## 注意事项
1. 模型输出可能带有主观性和不确定性
2. 建议在适当的场景下使用
3. 模型输出仅供参考，请勿过分依赖

## 许可证
本项目采用 MIT 许可证
"""
        
        with open(os.path.join(final_model_path, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # 3. 保存训练配置和性能指标
        training_info = {
            "model_config": MODEL_CONFIG,
            "training_metrics": {
                "final_loss": status.training_loss,
                "train_runtime": status.metrics['train_runtime'],
                "train_samples_per_second": status.metrics['train_samples_per_second'],
                "train_steps_per_second": status.metrics['train_steps_per_second'],
                "train_dataset_size": len(train_dataset),
                "eval_dataset_size": len(eval_dataset)
            },
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(final_model_path, "training_info.json"), "w", encoding="utf-8") as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)

        # 上传到HuggingFace Hub
        repo_name = "yuebanlaosiji/e-girl-model"
        logging.info(f"正在创建/更新仓库：{repo_name}")
        create_repo(repo_id=repo_name, repo_type="model", token=token, exist_ok=True)

        logging.info("正在上传模型到HuggingFace Hub...")
        new_model.push_to_hub(
            repo_name,
            tokenizer=tokenizer,
            token=token,
            commit_message=f"Update model with training loss {status.training_loss:.4f}"
        )
        logging.info("模型上传完成！")

    except Exception as e:
        logging.error(f"训练失败，错误信息：{str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()






