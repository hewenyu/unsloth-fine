from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
import os
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
import logging
from datetime import datetime
import sys
from dotenv import load_dotenv
from huggingface_hub import create_repo

# Load environment variables
load_dotenv()

# 设置环境变量以启用 Flash Attention 2
os.environ["USE_FLASH_ATTENTION"] = "1"

# 设置日志
log_dir = "data_create_dianzinvyou/output/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# 确保输出目录存在
output_base_dir = "data_create_dianzinvyou/output"
model_output_dir = os.path.join(output_base_dir, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
os.makedirs(model_output_dir, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "model_name": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    "max_seq_length": 2048,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 5,
    "learning_rate": 2e-4,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.1
}

model_name = MODEL_CONFIG["model_name"]
model,tokenizer = FastLanguageModel.from_pretrained(model_name, trust_remote_code=True)

# 添加 EOS 标记
EOS_TOKEN = tokenizer.eos_token

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
    # 多个文件
    # ['instruction', 'input', 'reasoning_content', 'output']
    input_texts = dataset['input']
    output_texts = dataset['output']
    reasoning_contents = dataset['reasoning_content']

    text_list = []

    for input_text,output_text,reasoning_content in zip(input_texts,output_texts,reasoning_contents):
        text = PROMPT_TEMPLATE.format(user_message=input_text,output_text=output_text,reasoning_content=reasoning_content) + EOS_TOKEN
        text_list.append(text)
    return {"text":text_list}
   
def main():
    try:
        # Log configuration
        logging.info("Starting training with configuration:")
        for key, value in MODEL_CONFIG.items():
            logging.info(f"{key}: {value}")

        # Check for required environment variables
        token = os.getenv("HuggingfaceToken")
        if not token:
            raise ValueError("HuggingfaceToken environment variable is not set")

        # 加载数据集
        logging.info("Loading dataset...")
        dataset = load_dataset("yuebanlaosiji/e-girl", trust_remote_code=True)
        logging.info(f"Dataset loaded with {len(dataset['train'])} training examples")

        # 格式化数据集
        logging.info("Formatting dataset...")
        dataset = dataset['train'].map(fromat_dataset_func, batched=True)
        logging.info("Dataset formatting completed")

        # Load model and tokenizer
        logging.info(f"Loading model: {MODEL_CONFIG['model_name']}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_CONFIG['model_name'],
            trust_remote_code=True
        )

        # 转换为训练模式
        FastLanguageModel.for_training(model)
        new_model = FastLanguageModel.get_peft_model(
            model,
            r=MODEL_CONFIG['lora_r'],
            target_modules=["q_proj", "v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_alpha=MODEL_CONFIG['lora_alpha'],
            lora_dropout=MODEL_CONFIG['lora_dropout'],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        training_args = TrainingArguments(
            output_dir=model_output_dir,
            per_device_train_batch_size=MODEL_CONFIG['per_device_batch_size'],
            gradient_accumulation_steps=MODEL_CONFIG['gradient_accumulation_steps'],
            num_train_epochs=MODEL_CONFIG['num_train_epochs'],
            warmup_ratio=0.1,
            learning_rate=MODEL_CONFIG['learning_rate'],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_total_limit=3,  # Keep only the last 3 checkpoints
            load_best_model_at_end=True,  # Load the best model when training is finished
            metric_for_best_model="loss",  # Use loss to determine the best model
            greater_is_better=False,  # Lower loss is better
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
        )

        trainer = SFTTrainer(
            model=new_model,
            train_dataset=dataset,
            dataset_text_field="text",
            tokenizer=tokenizer,
            max_seq_length=MODEL_CONFIG['max_seq_length'],
            dataset_num_proc=4,
            packing=True,
            args=training_args
        )

        logging.info("Starting training...")
        status = trainer.train()
        logging.info(f"Training completed with status: {status}")

        # Save the final model
        final_model_path = os.path.join(model_output_dir, "final_model")
        logging.info(f"Saving final model to {final_model_path}")
        new_model.save_pretrained(final_model_path)

        # Upload to HuggingFace Hub
        repo_name = "yuebanlaosiji/e-girl-model"
        logging.info(f"Creating/updating repository: {repo_name}")
        create_repo(repo_id=repo_name, repo_type="model", token=token, exist_ok=True)

        logging.info("Pushing model to HuggingFace Hub...")
        new_model.push_to_hub(repo_name, tokenizer=tokenizer, token=token)
        logging.info("Model successfully pushed to HuggingFace Hub")

    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()






