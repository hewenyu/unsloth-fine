import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
import random
from typing import List, Dict

# 扩展词汇库
NOUNS = [
    '春', '秋', '风', '雨', '山', '水', '花', '鸟', '月', '日', '天', '地', 
    '云', '雾', '江', '湖', '海', '河', '松', '竹', '梅', '兰', '草', '树',
    '星', '雪', '霜', '露', '龙', '凤', '虎', '豹', '琴', '棋', '书', '画',
    '亭', '台', '楼', '阁', '庭', '院', '门', '窗', '桥', '路', '溪', '涧'
]

VERBS = [
    '飞', '落', '升', '沉', '来', '去', '进', '退', '开', '合', '起', '落',
    '游', '走', '跑', '跳', '唱', '和', '吟', '咏', '望', '看', '思', '忆',
    '笑', '哭', '醉', '醒', '栖', '居', '寄', '托', '种', '植', '采', '摘'
]

ADJECTIVES = [
    '红', '绿', '青', '白', '高', '低', '远', '近', '深', '浅', '明', '暗',
    '古', '今', '雅', '俗', '清', '浊', '善', '恶', '美', '丑', '真', '假',
    '冷', '热', '干', '湿', '轻', '重', '快', '慢', '硬', '软', '苦', '甜'
]

def setup_model_and_tokenizer():
    """设置模型和分词器"""
    print("正在加载模型和分词器...")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=128,
        dtype=None,  # 自动选择dtype
        load_in_4bit=True,  # 使用4bit量化
    )
    
    return model, tokenizer

def generate_couplet_first_line():
    """生成上联的提示词列表"""
    # 这里可以根据需要扩充或修改
    themes = ["春", "秋", "月", "花", "山", "水", "风", "雨", "梅", "竹"]
    lengths = [5, 7]  # 五言或七言
    
    prompts = []
    for theme in themes:
        for length in lengths:
            prompt = f"请生成一个包含「{theme}」字的{length}言上联"
            prompts.append(prompt)
    
    return prompts

def generate_couplets(model, tokenizer, num_pairs=1000, temperature=0.7, max_length=128):
    """生成对联数据集"""
    prompts = generate_couplet_first_line()
    couplets = []
    
    print(f"开始生成{num_pairs}对对联...")
    
    with torch.no_grad():
        for _ in tqdm(range(0, num_pairs, len(prompts))):
            # 随机选择一个提示词
            prompt = random.choice(prompts)
            
            # 生成上联
            input_text = f"请生成一个对联。{prompt}。"
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 解析生成的文本，提取上联
            try:
                # 假设模型会以某种格式输出上联
                if "上联：" in generated_text:
                    first_line = generated_text.split("上联：")[1].split("\n")[0].strip()
                else:
                    continue
                
                # 使用上联生成下联
                input_text = f"已知上联：{first_line}\n请生成对应的下联，要求字数相同，平仄协调。"
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    num_return_sequences=1,
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 解析下联
                if "下联：" in generated_text:
                    second_line = generated_text.split("下联：")[1].split("\n")[0].strip()
                else:
                    continue
                
                # 确保上下联长度相等
                if len(first_line) == len(second_line):
                    couplets.append({
                        "up": first_line,
                        "down": second_line
                    })
            except Exception as e:
                print(f"处理生成的文本时出错: {e}")
                continue
            
            # 如果已经收集够了足够的对联，就退出
            if len(couplets) >= num_pairs:
                break
    
    return couplets

def save_dataset(couplets, output_dir="data"):
    """保存数据集"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "couplets.json")
    
    # 划分训练集和验证集
    random.shuffle(couplets)
    split_idx = int(len(couplets) * 0.9)  # 90% 作为训练集
    
    dataset = {
        "train": couplets[:split_idx],
        "validation": couplets[split_idx:]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据集已保存到 {output_path}")
    print(f"训练集大小: {len(dataset['train'])} 对")
    print(f"验证集大小: {len(dataset['validation'])} 对")

def main():
    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)
    
    # 加载模型和分词器
    model, tokenizer = setup_model_and_tokenizer()
    
    # 生成对联
    num_pairs = 1000  # 设置要生成的对联数量
    couplets = generate_couplets(model, tokenizer, num_pairs=num_pairs)
    
    # 保存数据集
    save_dataset(couplets)

if __name__ == "__main__":
    main() 