# 导入必要的库
from unsloth import FastLanguageModel  # 导入unsloth的快速语言模型
import torch  # PyTorch深度学习框架
from datasets import load_dataset  # Hugging Face数据集加载工具
from transformers import TrainingArguments  # 训练参数配置
import os
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
    "gradient_accumulation_steps": 4,  # 梯度累积步数
    "num_train_epochs": 5,  # 训练轮数
    "learning_rate": 2e-4,  # 学习率
    "lora_r": 16,  # LoRA秩
    "lora_alpha": 16,  # LoRA缩放因子
    "lora_dropout": 0.1  # LoRA dropout率
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
        dataset = dataset['train'].map(fromat_dataset_func, batched=True)
        logging.info("数据集格式化完成")

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
            gradient_accumulation_steps=MODEL_CONFIG['gradient_accumulation_steps'],  # 梯度累积步数
            num_train_epochs=MODEL_CONFIG['num_train_epochs'],  # 训练轮数
            warmup_ratio=0.1,  # 预热比例
            learning_rate=MODEL_CONFIG['learning_rate'],  # 学习率
            fp16=not is_bfloat16_supported(),  # 是否使用FP16
            bf16=is_bfloat16_supported(),  # 是否使用BF16
            logging_steps=10,  # 日志记录步数
            save_steps=100,  # 保存检查点步数
            eval_steps=100,  # 评估步数
            save_total_limit=3,  # 保存最近3个检查点
            load_best_model_at_end=True,  # 训练结束后加载最佳模型
            metric_for_best_model="loss",  # 使用损失作为最佳模型指标
            greater_is_better=False,  # 损失值越小越好
            optim="adamw_torch",  # 优化器
            weight_decay=0.01,  # 权重衰减
            lr_scheduler_type="cosine",  # 学习率调度器类型
            seed=3407,  # 随机种子
        )

        # 创建训练器
        trainer = SFTTrainer(
            model=new_model,
            train_dataset=dataset,
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
        new_model.save_pretrained(final_model_path)

        # 上传到HuggingFace Hub
        repo_name = "yuebanlaosiji/e-girl-model"
        logging.info(f"正在创建/更新仓库：{repo_name}")
        create_repo(repo_id=repo_name, repo_type="model", token=token, exist_ok=True)

        logging.info("正在上传模型到HuggingFace Hub...")
        new_model.push_to_hub(repo_name, tokenizer=tokenizer, token=token)
        logging.info("模型上传完成！")

    except Exception as e:
        logging.error(f"训练失败，错误信息：{str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()






