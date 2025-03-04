from unsloth import FastLanguageModel
import torch


# 加载预训练模型 这两个作为测试
# unsloth/DeepSeek-R1-Distill-Qwen-7B
# unsloth/DeepSeek-R1-Distill-Llama-8B

model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B"

# 设置最大序列长度
max_seq_length = 8192
# 默认使用fp16
dtype = None 
# load_in_8bit 是否使用8位量化
load_in_8bit = False


model, tokenizer = FastLanguageModel.from_pretrained(model_name, max_seq_length=max_seq_length, dtype=dtype, load_in_8bit=load_in_8bit)



if __name__ == "__main__":
    # 使用模型
    print("hello")





