"""
Utilities for knowledge distillation from teacher to student model
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Tuple, Optional, Union, Any


class DistillationTrainer:
    """
    Trainer for knowledge distillation from a teacher model to a student model
    """
    
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        student_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.5
    ):
        """
        Initialize the distillation trainer
        
        Args:
            teacher_model: The teacher model (unsloth/DeepSeek-R1-Distill-Qwen-32B)
            student_model: The student model (Qwen/Qwen2.5-0.5B)
            tokenizer: Tokenizer for the models
            temperature: Temperature for softening the teacher's outputs
            alpha: Weight for the distillation loss
            beta: Weight for the standard training loss
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        # Put teacher in evaluation mode
        self.teacher_model.eval()
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the distillation loss
        
        Args:
            student_logits: Logits from the student model
            teacher_logits: Logits from the teacher model
            labels: Ground truth labels for computing the standard loss
            attention_mask: Attention mask for ignoring padding tokens
            
        Returns:
            Tuple of (total_loss, distillation_loss, standard_loss)
        """
        # Compute the standard loss (cross-entropy)
        # For tokens where labels is -100, the loss will be ignored
        standard_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        standard_loss = standard_loss_fct(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        # Compute the distillation loss (KL-divergence)
        # Apply temperature scaling to logits
        soft_student_logits = student_logits / self.temperature
        soft_teacher_logits = teacher_logits / self.temperature
        
        # Create a mask to ignore padding tokens and prompt tokens
        # For distillation, we only want to compute loss on tokens that are not padding
        # and not part of the prompt (where labels != -100)
        mask = (labels != -100).unsqueeze(-1).expand_as(soft_student_logits)
        
        # Apply softmax to get probabilities
        soft_student_probs = F.softmax(soft_student_logits, dim=-1)
        soft_teacher_probs = F.softmax(soft_teacher_logits, dim=-1)
        
        # Compute KL divergence loss
        distillation_loss = F.kl_div(
            F.log_softmax(soft_student_logits, dim=-1),
            soft_teacher_probs,
            reduction='none'
        ).sum(-1)
        
        # Apply the mask and average
        distillation_loss = (distillation_loss * mask.float()).sum() / mask.float().sum()
        
        # Scale the distillation loss by temperature squared (to account for temperature scaling)
        distillation_loss *= self.temperature ** 2
        
        # Combine losses with weights
        total_loss = self.alpha * distillation_loss + self.beta * standard_loss
        
        return total_loss, distillation_loss, standard_loss
    
    def get_teacher_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the logits from the teacher model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Teacher's logits
        """
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            return teacher_outputs.logits
    
    def forward_train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a forward training step
        
        Args:
            batch: Batch of data containing input_ids, attention_mask, and labels
            
        Returns:
            Dictionary with losses
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Get teacher logits
        teacher_logits = self.get_teacher_logits(input_ids, attention_mask)
        
        # Get student logits
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        student_logits = student_outputs.logits
        
        # Compute distillation loss
        total_loss, distillation_loss, standard_loss = self.compute_distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            attention_mask=attention_mask
        )
        
        return {
            "loss": total_loss,
            "distillation_loss": distillation_loss,
            "standard_loss": standard_loss
        } 