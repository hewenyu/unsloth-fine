"""
Configuration settings for distillation of Qwen2.5-0.5B with DeepSeek-R1-Distill-Qwen-32B
"""

# Teacher model configuration
TEACHER_MODEL_ID = "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"
TEACHER_MODEL_LOAD_KWARGS = {
    "device_map": "auto",
    "load_in_4bit": True,
    "use_flash_attention_2": True,
}

# Student model configuration
STUDENT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
STUDENT_MODEL_LOAD_KWARGS = {
    "device_map": "auto",
    "use_flash_attention_2": True,
}

# Training configuration
TRAINING_CONFIG = {
    "output_dir": "../trained_models/distilled_qwen_duilian",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "optim": "adamw_torch",
    "logging_steps": 10,
    "save_steps": 500,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "bf16": True,  # Use mixed precision training
    "max_seq_length": 1024,
    "lazy_preprocess": False,
}

# Distillation configuration
DISTILLATION_CONFIG = {
    "temperature": 2.0,  # Temperature for softening the teacher's outputs
    "alpha": 0.5,  # Weight for the distillation loss (vs standard LM loss)
    "beta": 0.5,  # Weight for the normal training loss
}

# Data configuration
DATA_CONFIG = {
    "train_file": "../data/duilian_dataset.json",
    "prompt_template": "请为下面的上联创作一个下联：\n{input}\n下联：",
    "validation_split": 0.05,
    "seed": 42,
} 