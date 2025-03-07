import warnings
warnings.filterwarnings("ignore")

import sys
import torch
import os
from transformers import logging
import time

# 设置 unsloth 缓存目录，使用当前目录
cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "unsloth_compiled_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["UNSLOTH_CACHE_DIR"] = cache_dir
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"  # 临时禁用编译
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"   # 临时禁用 Triton

os.environ["XFORMERS_MORE_DETAILS"] = "1"

# 设置环境变量
os.environ["TRANSFORMERS_VERBOSITY"] = "debug"  # 改为 debug 级别
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

# 设置详细的日志级别
logging.set_verbosity_debug()
logging.enable_explicit_format()

if __name__ == "__main__":
    try:
        print("开始初始化...")
        start_time = time.time()
        
        # 显示当前设备信息
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 先导入 unsloth，看看是否能成功
        print("导入 unsloth...")
        from unsloth import FastLanguageModel
        print("unsloth 导入成功！")
        
        # 使用较小的模型
        print("\n开始加载模型...")
        model_name = "facebook/opt-125m"  # 使用小模型测试
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=512,  # 减小序列长度
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,  # 禁用缓存
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        end_time = time.time()
        loading_time = end_time - start_time
        print(f"\n模型加载成功！耗时: {loading_time:.2f} 秒")
        
        # 测试模型
        print("\n进行简单的测试...")
        test_input = "Hello, please introduce yourself."
        print(f"输入: {test_input}")
        
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        
        print("生成回答中...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=1,
                do_sample=True,
                temperature=0.7
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"输出: {response}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        print(traceback.format_exc())





