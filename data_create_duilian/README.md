# 对联数据集生成

这个项目使用 DeepSeek-R1-Distill-Qwen-32B 模型来生成高质量的对联数据集。

## 环境要求

- Python 3.8+
- CUDA 支持的 GPU（建议至少 16GB 显存）
- 至少 32GB 系统内存

## 安装

1. 创建并激活虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 生成数据集

运行数据生成脚本：

```bash
python generate_couplets_dataset.py
```

这将使用 DeepSeek-R1-Distill-Qwen-32B 模型生成对联数据集，默认会生成 1000 对对联，并保存在 `data/couplets.json` 文件中。

### 数据生成参数说明

在 `generate_couplets_dataset.py` 中可以调整以下参数：

1. 生成参数：
   - `num_pairs`: 要生成的对联数量，默认 1000
   - `temperature`: 生成的创造性程度，默认 0.7
   - `max_length`: 生成文本的最大长度，默认 128
   - `top_p`: 采样概率阈值，默认 0.9
   - `top_k`: 候选词数量，默认 50

2. 主题设置：
   - 在 `generate_couplet_first_line()` 函数中可以修改主题词
   - 支持生成五言和七言对联
   - 可以自定义添加更多主题词

3. 数据集划分：
   - 默认将数据集按 9:1 的比例划分为训练集和验证集
   - 可以在 `save_dataset()` 函数中调整划分比例

### 生成的数据格式

生成的数据将保存为 JSON 格式：

```json
{
  "train": [
    {
      "up": "上联文本",
      "down": "下联文本"
    },
    ...
  ],
  "validation": [
    {
      "up": "上联文本",
      "down": "下联文本"
    },
    ...
  ]
}
```

## 注意事项

1. 确保有足够的 GPU 显存和系统内存
2. 生成过程中会使用 4bit 量化以减少显存占用
3. 生成速度取决于 GPU 性能和生成参数设置
4. 可以通过调整 temperature 参数来控制生成的创造性
5. 建议使用 wandb 来监控生成过程

## 许可证

本项目使用 MIT 许可证。请注意，使用的 DeepSeek-R1-Distill-Qwen-32B 模型可能有其自己的使用限制。