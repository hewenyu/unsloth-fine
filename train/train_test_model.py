from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型和分词器
model_name = "yuebanlaosiji/e-girl-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto"
)

# 示例对话
prompt = '''你现在是一个温柔、包容、善解人意的女友。你需要以女友的身份回复用户的消息。
用户消息: 今天工作好累啊
女友回复:'''

# 生成回复
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_length=2048,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

if __name__ == "__main__":
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response.split("女友回复:")[-1].strip())