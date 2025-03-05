from unsloth import FastLanguageModel
import sys
import torch
import importlib

def check_dependencies():
    """检查所需的依赖包"""
    print("检查依赖包...")
    
    try:
        import transformers
        print(f"transformers 版本: {transformers.__version__}")
    except ImportError as e:
        print(f"transformers 导入错误: {str(e)}")
        print("请尝试: pip install transformers")
        raise

    try:
        import accelerate
        print(f"accelerate 版本: {accelerate.__version__}")
    except ImportError as e:
        print(f"accelerate 导入错误: {str(e)}")
        print("请尝试: pip install accelerate")
        raise

    try:
        import triton
        print(f"triton 版本: {triton.__version__}")
    except ImportError:
        print("\n注意: triton 包未安装。")
        print("这不会影响基本功能，但可能会影响性能。")
        print("如果之后想安装 triton，可以尝试：")
        print("pip install pytorch-triton --no-deps")
        # 不抛出异常，继续运行

    try:
        import unsloth
        print(f"unsloth 版本: {unsloth.__version__ if hasattr(unsloth, '__version__') else '未知'}")
    except ImportError as e:
        print(f"unsloth 导入错误: {str(e)}")
        print("请尝试: pip install unsloth")
        raise

def check_environment():
    """检查运行环境"""
    print("=== 环境检查 ===")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"检测到 {device_count} 个 GPU:")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\n错误: 未检测到 CUDA！")
        print("请按以下步骤解决：")
        print("1. 卸载当前的 PyTorch")
        print("   pip uninstall torch torchvision torchaudio")
        print("2. 安装 CUDA 版本的 PyTorch（适用于 CUDA 12.8）：")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        raise RuntimeError("需要 CUDA 环境才能继续")
    print("================")

# 加载预训练模型 这两个作为测试
# unsloth/DeepSeek-R1-Distill-Qwen-7B
# unsloth/DeepSeek-R1-Distill-Llama-8B



if __name__ == "__main__":
    try:
        # # 首先检查环境
        # check_environment()
        # # 检查依赖
        # check_dependencies()

        from unsloth import FastLanguageModel

       
        model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B"

        # 设置最大序列长度
        max_seq_length = 8192
        # 默认使用fp16
        dtype = None 
        # load_in_8bit 是否使用8位量化
        load_in_4bit = True
        
        # 加载模型
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit
        )
        
        # 使用模型
        print("模型加载成功！")
        
    except Exception as e:
        print(f"错误: {str(e)}")





