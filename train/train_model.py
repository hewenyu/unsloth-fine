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
        # 加载数据集
        dataset = load_dataset("yuebanlaosiji/e-girl",trust_remote_code=True)
        # 多个文件
        print(dataset['train'].column_names)

        # 格式化数据集
        dataset = dataset['train'].map(fromat_dataset_func, batched=True)

        print(dataset["text"][0])

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
            per_device_train_batch_size=4, # 设置每个设备训练批次大小
            gradient_accumulation_steps=4, # 设置梯度累积步数,用于模拟大batch_size
            num_train_epochs=5, # 设置训练轮数,2.5K数据建议3-5轮
            warmup_ratio=0.1, # 设置预热比例,总步数的10%
            learning_rate=2e-4, # 设置学习率
            fp16=not is_bfloat16_supported(), # 设置fp16
            bf16=is_bfloat16_supported(), # 设置bf16
            logging_steps=10, # 每10步记录一次日志
            save_steps=100, # 每100步保存一次模型
            eval_steps=100, # 每100步评估一次
            optim="adamw_torch", # 设置优化器
            weight_decay=0.01, # 设置权重衰减
            lr_scheduler_type="cosine", # 余弦退火学习率
            seed=3407, # 设置随机种子
        )
        # max_seq_length 越大越好
        max_seq_length = 2048

        # 训练模型  
        trainer = SFTTrainer(
            model=new_model,
            train_dataset=dataset, 
            dataset_text_field="text",
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            dataset_num_proc=4, # 设置数据集处理进程数
            packing=True, # 设置数据集打包
            args=training_args
        )

        status = trainer.train()

        print(status)

        # 保存模型
        new_model.save_pretrained("output/model")
        # tokenizer.save_pretrained("output/tokenizer")

        from huggingface_hub import create_repo
        token = os.getenv("HuggingfaceToken")

        repo_name = "yuebanlaosiji/e-girl-model"
        create_repo(repo_id=repo_name,repo_type="model",token=token,exist_ok=True)

        # 上传模型到huggingface
        new_model.push_to_hub(repo_name,tokenizer=tokenizer,token=token)

    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
        raise e




if __name__ == "__main__":
    main()






