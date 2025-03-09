"""
Main script for training the student model with knowledge distillation
"""

import os
import sys
import argparse
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    set_seed,
)
from unsloth import FastLanguageModel
import wandb
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_distillation.utils.data_utils import prepare_dataset
from qwen_distillation.utils.distillation_utils import DistillationTrainer
from qwen_distillation.configs.config import (
    TEACHER_MODEL_ID,
    TEACHER_MODEL_LOAD_KWARGS,
    STUDENT_MODEL_ID,
    STUDENT_MODEL_LOAD_KWARGS,
    TRAINING_CONFIG,
    DISTILLATION_CONFIG,
    DATA_CONFIG,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"distillation_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    """
    Custom trainer for distillation that overrides the compute_loss method
    """
    
    def __init__(self, distillation_trainer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_trainer = distillation_trainer
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override the compute_loss method to use our distillation loss
        """
        # Forward pass through the student model and compute distillation loss
        loss_dict = self.distillation_trainer.forward_train_step(inputs)
        
        # Log the losses
        self.log({
            "total_loss": loss_dict["loss"].item(),
            "distillation_loss": loss_dict["distillation_loss"].item(),
            "standard_loss": loss_dict["standard_loss"].item(),
        })
        
        return (loss_dict["loss"], None) if return_outputs else loss_dict["loss"]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a model with distillation")
    
    parser.add_argument(
        "--teacher_model_id",
        type=str,
        default=TEACHER_MODEL_ID,
        help="ID of the teacher model"
    )
    
    parser.add_argument(
        "--student_model_id",
        type=str,
        default=STUDENT_MODEL_ID,
        help="ID of the student model"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_CONFIG["train_file"],
        help="Path to the training data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=TRAINING_CONFIG["output_dir"],
        help="Output directory for the model"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=DISTILLATION_CONFIG["temperature"],
        help="Temperature for distillation"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=DISTILLATION_CONFIG["alpha"],
        help="Weight for distillation loss"
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        default=DISTILLATION_CONFIG["beta"],
        help="Weight for standard loss"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=DATA_CONFIG["seed"],
        help="Random seed"
    )
    
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=TRAINING_CONFIG["num_train_epochs"],
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=TRAINING_CONFIG["per_device_train_batch_size"],
        help="Batch size per device for training"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=TRAINING_CONFIG["gradient_accumulation_steps"],
        help="Number of gradient accumulation steps"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=TRAINING_CONFIG["learning_rate"],
        help="Learning rate"
    )
    
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights and Biases for logging"
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set the seed for reproducibility
    set_seed(args.seed)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="qwen-distillation",
            name=f"distill-{os.path.basename(args.student_model_id)}",
            config=vars(args)
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading teacher model: {args.teacher_model_id}")
    # Load the teacher model (with Unsloth for faster inference)
    teacher_model, teacher_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.teacher_model_id,
        **TEACHER_MODEL_LOAD_KWARGS
    )
    teacher_model = teacher_model.get_peft_model()
    
    logger.info(f"Loading student model: {args.student_model_id}")
    # Load the student model
    student_model, student_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.student_model_id,
        **STUDENT_MODEL_LOAD_KWARGS
    )
    
    # Make the model trainable
    student_model = FastLanguageModel.get_peft_model(
        student_model,
        r=16,  # LoRA attention dimension
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
    )
    
    # Prepare the dataset
    logger.info(f"Preparing dataset from {args.data_path}")
    train_dataset, val_dataset = prepare_dataset(
        data_path=args.data_path,
        tokenizer=student_tokenizer,
        prompt_template=DATA_CONFIG["prompt_template"],
        validation_split=DATA_CONFIG["validation_split"],
        seed=args.seed,
        max_length=TRAINING_CONFIG["max_seq_length"]
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create the distillation trainer
    distillation_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=student_tokenizer,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta
    )
    
    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=TRAINING_CONFIG["weight_decay"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        bf16=TRAINING_CONFIG["bf16"],
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        report_to="wandb" if args.use_wandb else "none",
        save_total_limit=2,
        remove_unused_columns=False,  # Important for custom training
    )
    
    # Create the trainer
    trainer = CustomTrainer(
        distillation_trainer=distillation_trainer,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {args.output_dir}")
    student_model.save_pretrained(args.output_dir)
    student_tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training complete")


if __name__ == "__main__":
    main() 