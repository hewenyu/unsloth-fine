# 导入必要的库
from unsloth import FastLanguageModel  # 导入unsloth的快速语言模型
import torch  # PyTorch深度学习框架
import os  # 操作系统接口
import logging  # 日志记录
from datetime import datetime  # 日期时间处理
import sys  # 系统相关
import argparse  # 命令行参数解析

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 控制台处理器
    ]
)

# 模型配置
MODEL_CONFIG = {
    "model_name": "yuebanlaosiji/e-girl-model",  # 训练好的模型名称
    "max_length": 2048,  # 最大序列长度
    "temperature": 0.7,  # 采样温度
    "top_p": 0.9,  # 核采样参数
    "repetition_penalty": 1.1  # 重复惩罚参数
}

# 定义提示模板
PROMPT_TEMPLATE = """你现在是一个温柔、包容、善解人意的女友。你需要以女友的身份回复用户的消息。
    你的性格特点：
    1. 温柔体贴，善于倾听
    2. 积极向上，富有正能量
    3. 会适当撒娇，但不会过分
    4. 懂得关心对方的工作和生活
    5. 会给予对方鼓励和支持

    用户消息: {user_message}

    思考过程:
    <thinking>
    让我想想该如何回复...
    </thinking>
    女友回复:
    """

def load_model():
    """
    加载预训练模型和分词器
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        logging.info(f"正在加载模型：{MODEL_CONFIG['model_name']}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_CONFIG['model_name'],
            trust_remote_code=True,
            device_map="auto"  # 自动选择设备
        )
        logging.info("模型加载完成")
        return model, tokenizer
    except Exception as e:
        logging.error(f"模型加载失败：{str(e)}")
        raise

def generate_response(model, tokenizer, user_input):
    """
    生成模型回复
    Args:
        model: 预训练模型
        tokenizer: 分词器
        user_input: 用户输入
    Returns:
        str: 模型回复
    """
    try:
        # 构建完整的提示
        prompt = PROMPT_TEMPLATE.format(user_message=user_input)
        
        # 生成回复
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=MODEL_CONFIG['max_length'],
            temperature=MODEL_CONFIG['temperature'],
            top_p=MODEL_CONFIG['top_p'],
            repetition_penalty=MODEL_CONFIG['repetition_penalty'],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # 解码回复
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回复部分
        response = response.split("女友回复:")[-1].strip()
        return response
    
    except Exception as e:
        logging.error(f"生成回复失败：{str(e)}")
        return "抱歉，我现在有点累了，可以稍后再聊吗？"

def single_test(model, tokenizer, test_input):
    """
    单次测试函数
    Args:
        model: 预训练模型
        tokenizer: 分词器
        test_input: 测试输入
    """
    logging.info(f"\n用户输入: {test_input}")
    response = generate_response(model, tokenizer, test_input)
    logging.info(f"模型回复: {response}\n")

def interactive_mode(model, tokenizer):
    """
    交互式对话模式
    Args:
        model: 预训练模型
        tokenizer: 分词器
    """
    logging.info("\n开始交互式对话 (输入 'quit' 或 'exit' 结束对话)")
    
    while True:
        user_input = input("\n你: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            logging.info("对话结束，再见！")
            break
            
        if not user_input:
            continue
            
        response = generate_response(model, tokenizer, user_input)
        print(f"\n女友: {response}")

def main():
    """主函数：处理命令行参数并执行相应的测试模式"""
    parser = argparse.ArgumentParser(description='测试训练好的模型')
    parser.add_argument('--mode', type=str, choices=['single', 'interactive'], 
                      default='interactive', help='测试模式：single(单次测试) 或 interactive(交互式)')
    parser.add_argument('--input', type=str, default=None, 
                      help='单次测试模式的输入文本')
    
    args = parser.parse_args()
    
    try:
        # 加载模型
        model, tokenizer = load_model()
        
        if args.mode == 'single':
            if not args.input:
                test_input = "今天工作好累啊，感觉好辛苦"
            else:
                test_input = args.input
            single_test(model, tokenizer, test_input)
        else:
            interactive_mode(model, tokenizer)
            
    except Exception as e:
        logging.error(f"程序执行失败：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 