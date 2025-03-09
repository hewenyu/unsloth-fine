"""
Utilities for processing and preparing data for distillation training
"""

import os
import json
import random
from typing import Dict, List, Tuple, Union
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def read_duilian_data(data_path: str) -> List[Dict]:
    """
    Read duilian (couplet) data from a JSON file
    
    Args:
        data_path: Path to the JSON file containing duilian data
        
    Returns:
        List of dictionaries with 'first_line' and 'second_line' keys
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def format_duilian_prompt(first_line: str, template: str = "请为下面的上联创作一个下联：\n{input}\n下联：") -> str:
    """
    Format the first line of a couplet into a prompt for the model
    
    Args:
        first_line: The first line of the couplet
        template: Template string for formatting the prompt
        
    Returns:
        Formatted prompt
    """
    return template.format(input=first_line)


def prepare_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    prompt_template: str,
    validation_split: float = 0.05,
    seed: int = 42,
    max_length: int = 1024
) -> Tuple[Dataset, Dataset]:
    """
    Prepare dataset for distillation training
    
    Args:
        data_path: Path to the JSON file containing duilian data
        tokenizer: Tokenizer for the model
        prompt_template: Template for formatting prompts
        validation_split: Proportion of data to use for validation
        seed: Random seed for reproducibility
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    random.seed(seed)
    data = read_duilian_data(data_path)
    random.shuffle(data)
    
    # Split data into train and validation
    val_size = int(len(data) * validation_split)
    train_data = data[val_size:]
    val_data = data[:val_size]
    
    # Create datasets
    train_dataset = DuilianDataset(train_data, tokenizer, prompt_template, max_length)
    val_dataset = DuilianDataset(val_data, tokenizer, prompt_template, max_length)
    
    return train_dataset, val_dataset


class DuilianDataset(Dataset):
    """
    Dataset for duilian (couplet) generation
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        prompt_template: str,
        max_length: int = 1024
    ):
        """
        Initialize dataset
        
        Args:
            data: List of dictionaries with 'first_line' and 'second_line' keys
            tokenizer: Tokenizer for the model
            prompt_template: Template for formatting prompts
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        first_line = item['first_line']
        second_line = item['second_line']
        
        # Format the prompt
        prompt = format_duilian_prompt(first_line, self.prompt_template)
        
        # The full text includes the prompt and the answer
        full_text = prompt + second_line
        
        # Tokenize inputs
        tokenized_full = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Compute the position where the answer starts
        prompt_tokens = self.tokenizer(
            prompt, 
            max_length=self.max_length, 
            truncation=True,
            return_tensors="pt"
        )
        prompt_len = prompt_tokens['input_ids'].shape[1]
        
        # Create labels: -100 for prompt tokens (we don't want to compute loss on them)
        labels = tokenized_full['input_ids'].clone()
        labels[:, :prompt_len] = -100
        
        return {
            'input_ids': tokenized_full['input_ids'].squeeze(0),
            'attention_mask': tokenized_full['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        } 