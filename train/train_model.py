from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
import os
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
import logging
from datetime import datetime

# 设置环境变量以启用 Flash Attention 2
os.environ["USE_FLASH_ATTENTION"] = "1"
# 确保输出目录存在
os.makedirs("data_create_dianzinvyou/output", exist_ok=True)

model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
model,tokenizer = FastLanguageModel.from_pretrained(model_name, trust_remote_code=True)

# 添加 EOS 标记
EOS_TOKEN = tokenizer.eos_token

# 设置日志
logging.basicConfig(
    filename=f'data_create_dianzinvyou/output/generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


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
    input_text = dataset['input']
    output_text = dataset['output']
    reasoning_content = dataset['reasoning_content']

    text_list = []

    for input_text,output_text,reasoning_content in zip(input_text,output_text,reasoning_content):
        text = PROMPT_TEMPLATE.format(user_message=input_text,output_text=output_text,reasoning_content=reasoning_content) + EOS_TOKEN
        text_list.append(text)
    return text_list
   
def main():

    try:
        # 加载数据集
        dataset = load_dataset("yuebanlaosiji/e-girl",trust_remote_code=True)
        # 多个文件
        print(dataset['train'].column_names)

        # 格式化数据集
        text_list = fromat_dataset_func(dataset['train'])

        print(text_list[0])

        # 转换为训练模式
        FastLanguageModel.for_training(model)
        new_model = FastLanguageModel.get_peft_model(
            model,
            r=16, # 设置lora 秩
            target_modules=["q_proj", "v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"], # 设置lora 目标模块
            lora_alpha=16, # 设置lora 缩放因子
            lora_dropout=0.1, # 设置lora 丢弃率 ,防止过拟合
            bias="none", # 设置偏置,none 不使用偏置
            use_gradient_checkpointing="unsloth", # 设置梯度检查点
            random_state=3407, # 设置随机种子, 确保每次训练结果一致
            use_rslora=False, # 设置rslora,False 不使用rslora
            loftq_config= None, # 设置loftq 配置,None 不使用loftq
        )

        # args_info
        training_args = TrainingArguments(
            output_dir="output",
            per_device_train_batch_size=2, # 设置每个设备训练批次大小
            gradient_accumulation_steps=4, # 设置梯度累积步数,用于模拟大batch_size
            max_steps=75, # 设置最大训练步数
            warmup_steps=5, # 设置预热步数
            learning_rate=1e-4, # 设置学习率
            fp16=not is_bfloat16_supported(), # 设置fp16
            bf16=is_bfloat16_supported(), # 设置bf16
            logging_steps=1, # 设置日志步数
            optim="adamw_torch", # 设置优化器
            weight_decay=0.01, # 设置权重衰减
            lr_scheduler_type="linear", # 设置学习率调度器类型 有cosine_with_restarts,linear,constant,cosine ，默认linear
            seed=3407, # 设置随机种子
            report_to="None", # 设置报告到wandb
        )
        # max_seq_length 越大越好
        max_seq_length = 2048

        # 训练模型  
        trainer = SFTTrainer(
            model=new_model,
            train_dataset=text_list,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            dataset_num_proc=4, # 设置数据集处理进程数
            packing=True, # 设置数据集打包
            args=training_args
        )

        trainer.train()

    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
        raise e




if __name__ == "__main__":
    main()






