# Qwen 对对子蒸馏项目

这个项目实现了一个使用教师-学生模型进行知识蒸馏的框架，用于训练中文对联（对对子）生成模型。

## 项目架构

```
qwen_distillation/
├── configs/           # 配置文件
├── data/              # 数据相关脚本和数据集
├── utils/             # 工具函数
├── train_distillation.py  # 训练脚本
├── run_inference.py   # 推理脚本
└── requirements.txt   # 依赖包
```

## 模型介绍

- **教师模型**: `unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit` - 一个32B参数的大型模型
- **学生模型**: `Qwen/Qwen2.5-0.5B` - 一个只有500M参数的小型模型

通过知识蒸馏，学生模型可以学习教师模型的"知识"，在保持较小模型大小的同时，提高生成对对子的能力。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集准备

项目包含一个创建示例对联数据集的脚本：

```bash
cd qwen_distillation
python data/create_sample_data.py --output ../data/duilian_dataset.json --samples 1000
```

实际应用中，您应该准备更大规模的真实对联数据集，并确保数据格式为：

```json
[
  {
    "first_line": "上联内容",
    "second_line": "下联内容"
  },
  ...
]
```

## 训练模型

使用以下命令开始训练：

```bash
python train_distillation.py \
  --data_path ../data/duilian_dataset.json \
  --output_dir ../trained_models/distilled_qwen_duilian \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5 \
  --use_wandb  # 可选，启用Weights & Biases进行训练监控
```

更多训练参数可以查看 `train_distillation.py` 中的 `parse_args` 函数。

## 模型推理

训练完成后，您可以使用以下命令测试模型：

```bash
python run_inference.py \
  --model_path ../trained_models/distilled_qwen_duilian \
  --first_line "春风送暖入屠苏" \
  --temperature 0.7 \
  --use_unsloth  # 可选，使用unsloth加速推理
```

## 蒸馏原理

本项目使用知识蒸馏技术，通过以下关键步骤实现：

1. **软标签生成**：教师模型（32B）生成带有概率分布的"软标签"
2. **温度缩放**：使用温度系数调整概率分布的"软硬程度"
3. **损失函数组合**：
   - 蒸馏损失：学生模型输出与教师模型软标签之间的KL散度
   - 标准损失：学生模型输出与真实标签之间的交叉熵损失
4. **加权组合**：使用α和β参数权衡两种损失的比重

## 自定义配置

您可以在 `configs/config.py` 文件中修改各种配置参数：

- 模型配置
- 训练参数
- 蒸馏超参数
- 数据配置

## 许可证

本项目使用 MIT 许可证。 