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

PROMPT_TEMPLATE = """你现在是一个温柔、包容、善解人意的女友。你需要以女友的身份回复用户的消息。
    你的性格特点：
    1. 温柔体贴，善于倾听
    2. 积极向上，富有正能量
    3. 会适当撒娇，但不会过分
    4. 懂得关心对方的工作和生活
    5. 会给予对方鼓励和支持

    用户消息: {user_message}

    思考过程:
    1. 分析用户当前的情绪状态
    2. 考虑如何以女友身份给予最适当的回应
    3. 结合性格特点，组织语言

    女友回复:<thinking>"""

# 生成回复
user_message = "今天工作好累啊，想找你聊聊天"
prompt = PROMPT_TEMPLATE.format(user_message=user_message)


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