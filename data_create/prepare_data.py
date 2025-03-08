import json
import os
from typing import List, Dict
import random

def load_raw_couplets(file_path: str) -> List[Dict[str, str]]:
    """加载原始对联数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip().split('\t') for line in f if line.strip()]

def process_couplets(couplets: List[List[str]]) -> List[Dict[str, str]]:
    """处理对联数据，转换为所需格式"""
    processed = []
    for up, down in couplets:
        if len(up) == len(down):  # 确保上下联长度相等
            processed.append({
                "up": up,
                "down": down
            })
    return processed

def split_dataset(data: List[Dict[str, str]], train_ratio: float = 0.9) -> Dict[str, List[Dict[str, str]]]:
    """将数据集分割为训练集和验证集"""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return {
        "train": data[:split_idx],
        "validation": data[split_idx:]
    }

def save_dataset(dataset: Dict[str, List[Dict[str, str]]], output_dir: str):
    """保存处理后的数据集"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "couplets.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def main():
    # 配置路径
    raw_data_path = "data/raw/couplets.txt"  # 原始对联数据文件路径
    output_dir = "data"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载并处理数据
    raw_couplets = load_raw_couplets(raw_data_path)
    processed_couplets = process_couplets(raw_couplets)
    
    # 分割数据集
    dataset = split_dataset(processed_couplets)
    
    # 保存处理后的数据集
    save_dataset(dataset, output_dir)
    
    print(f"数据集处理完成！共处理 {len(processed_couplets)} 对对联")
    print(f"训练集: {len(dataset['train'])} 对")
    print(f"验证集: {len(dataset['validation'])} 对")

if __name__ == "__main__":
    main() 