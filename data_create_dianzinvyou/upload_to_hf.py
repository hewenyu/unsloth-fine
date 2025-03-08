from datasets import Dataset
from huggingface_hub import HfApi
import json
import os

# 读取生成的数据集
with open("data_create_dianzinvyou/output/girlfriend_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 转换为Dataset格式
dataset = Dataset.from_list(data)


# token 从环境中获取 
token = os.getenv("HuggingfaceToken")

# 上传到HuggingFace
dataset.push_to_hub(
    "yuebanlaosiji/e-girl",
    token=token,  # 请替换为你的HuggingFace token
    private=False
)

print("数据集已成功上传到 HuggingFace!") 