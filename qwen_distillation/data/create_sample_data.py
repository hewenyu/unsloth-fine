"""
Script to create a sample duilian (couplet) dataset for testing
"""

import os
import json
import argparse
from typing import Dict, List

# Sample couplets for testing
SAMPLE_COUPLETS = [
    {
        "first_line": "春风送暖入屠苏",
        "second_line": "夜雨闻铃已陆离"
    },
    {
        "first_line": "风吹柳絮满店香",
        "second_line": "月照梅花一院春"
    },
    {
        "first_line": "海上生明月",
        "second_line": "天涯共此时"
    },
    {
        "first_line": "锦瑟无端五十弦",
        "second_line": "一弦一柱思华年"
    },
    {
        "first_line": "谁道人生无再少",
        "second_line": "门前流水尚能西"
    },
    {
        "first_line": "大漠孤烟直",
        "second_line": "长河落日圆"
    },
    {
        "first_line": "春眠不觉晓",
        "second_line": "处处闻啼鸟"
    },
    {
        "first_line": "欲穷千里目",
        "second_line": "更上一层楼"
    },
    {
        "first_line": "会当凌绝顶",
        "second_line": "一览众山小"
    },
    {
        "first_line": "青山遮不住",
        "second_line": "毕竟东流去"
    },
]


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """
    Create a sample duilian dataset by duplicating and slightly modifying the sample couplets
    
    Args:
        output_path: Path to save the dataset
        num_samples: Number of samples to generate
    """
    data = []
    for i in range(num_samples):
        # Choose a sample couplet
        sample_idx = i % len(SAMPLE_COUPLETS)
        couplet = SAMPLE_COUPLETS[sample_idx].copy()
        
        # Add to the dataset
        data.append(couplet)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Created sample dataset with {len(data)} samples at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a sample duilian dataset")
    parser.add_argument("--output", type=str, default="../data/duilian_dataset.json", 
                        help="Path to save the dataset")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to generate")
    
    args = parser.parse_args()
    create_sample_dataset(args.output, args.samples) 