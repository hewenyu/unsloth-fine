"""
Script to run inference with the distilled model for generating duilian (couplets)
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from unsloth import FastLanguageModel

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_distillation.utils.data_utils import format_duilian_prompt
from qwen_distillation.configs.config import (
    DATA_CONFIG,
    STUDENT_MODEL_ID,
    STUDENT_MODEL_LOAD_KWARGS,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with the distilled model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="../trained_models/distilled_qwen_duilian",
        help="Path to the distilled model or model ID"
    )
    
    parser.add_argument(
        "--first_line",
        type=str,
        required=True,
        help="First line of the couplet"
    )
    
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=DATA_CONFIG["prompt_template"],
        help="Template for formatting the prompt"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p for nucleus sampling"
    )
    
    parser.add_argument(
        "--use_unsloth",
        action="store_true",
        help="Whether to use Unsloth for faster inference"
    )
    
    return parser.parse_args()


def main():
    """Run inference with the model"""
    args = parse_args()
    
    print(f"Loading model from {args.model_path}")
    
    # Check if the model path exists, if not assume it's a HuggingFace model ID
    if not os.path.exists(args.model_path):
        print(f"Model path {args.model_path} does not exist, using model ID {STUDENT_MODEL_ID}")
        args.model_path = STUDENT_MODEL_ID
    
    # Load the model
    if args.use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            **STUDENT_MODEL_LOAD_KWARGS
        )
        # Convert to HF model for generation
        hf_model = model.to_hf_model()
    else:
        # Regular loading
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Format the prompt
    prompt = format_duilian_prompt(args.first_line, args.prompt_template)
    print(f"Prompt: {prompt}")
    
    # Create a generation pipeline
    generator = pipeline(
        "text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Generate the response
    outputs = generator(prompt, return_full_text=False)
    
    # Print the generated second line
    second_line = outputs[0]["generated_text"].strip()
    print(f"Generated second line: {second_line}")
    
    # Display the full couplet
    print("\nFull Couplet:")
    print(f"上联: {args.first_line}")
    print(f"下联: {second_line}")


if __name__ == "__main__":
    main() 