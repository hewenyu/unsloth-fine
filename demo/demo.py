import sys
import torch
import os
from transformers import logging, AutoModelForCausalLM, AutoTokenizer
import time

# 设置环境变量启用详细日志
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

# 设置详细的日志级别
logging.set_verbosity_info()
logging.enable_explicit_format()

# 添加 huggingface_hub 的日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    try:
        print("开始加载模型...")
        start_time = time.time()
        
        # 使用小模型测试
        model_name = "facebook/opt-125m"
        print(f"正在加载模型: {model_name}")
        
        # 显示当前设备信息
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        print("\n开始加载模型到内存...")
        
        # 加载tokenizer
        print("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        print("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
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





