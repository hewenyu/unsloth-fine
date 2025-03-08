import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
from tqdm import tqdm
import os

# 确保输出目录存在
os.makedirs("data_create_dianzinvyou/output", exist_ok=True)

# 初始化模型和tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

# 定义提示模板
PROMPT_TEMPLATE = """你现在是一个温柔、包容、善解人意的女友。你需要以女友的身份回复用户的消息。
你的性格特点：
1. 温柔体贴，善于倾听
2. 积极向上，富有正能量
3. 会适当撒娇，但不会过分
4. 懂得关心对方的工作和生活
5. 会给予对方鼓励和支持

用户消息: {user_message}
女友回复:"""

# 准备一些常见的用户消息场景
user_messages = [
    "今天工作好累啊，感觉压力好大",
    "我最近在学习编程，但是感觉好难",
    "想你了，在干什么呢？",
    "今天考试成绩出来了，没考好，有点沮丧",
    "周末要不要一起出去玩？",
    "今天遇到一个难题，觉得很困扰",
    "最近状态不是很好，需要一些建议",
    "我有一个好消息要告诉你！",
    "今天天气真好，想和你一起散步",
    "工作上遇到了一些困难，需要一些鼓励",
    # 添加更多场景...
]

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("女友回复:")[-1].strip()

# 生成数据集
dataset = []
num_samples = 100  # 生成100个样本

for _ in tqdm(range(num_samples)):
    user_message = random.choice(user_messages)
    prompt = PROMPT_TEMPLATE.format(user_message=user_message)
    response = generate_response(prompt)
    
    # 构建数据样本
    sample = {
        "instruction": "你是一个温柔体贴的女友，请以女友的身份回复以下消息",
        "input": user_message,
        "output": response
    }
    dataset.append(sample)

# 保存数据集
output_file = "data_create_dianzinvyou/output/girlfriend_dataset.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"数据集已生成并保存到: {output_file}") 