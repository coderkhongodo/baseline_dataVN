#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cấu hình tối ưu cho huấn luyện với 2x RTX A5000 (24GB VRAM mỗi card)
Tổng cộng: 48GB VRAM - Rất mạnh cho các model Vietnamese
"""

import torch
from dataclasses import dataclass
from typing import Dict, Any, List

# ==================== HARDWARE SPECS ====================
HARDWARE_SPECS = {
    "gpu_count": 2,
    "gpu_memory_per_card": "24GB",
    "total_memory": "48GB", 
    "gpu_model": "RTX A5000",
    "compute_capability": "8.6",
    "recommended_precision": "mixed_precision_fp16"
}

# ==================== BERT FAMILY CONFIGS ====================

@dataclass
class PhoBERTConfig:
    """Cấu hình cho PhoBERT trên 2x RTX A5000"""
    
    # Model settings
    model_name: str = "vinai/phobert-base"
    max_length: int = 256
    num_labels: int = 2
    
    # Training hyperparameters - Tối ưu cho 2x A5000
    batch_size_per_gpu: int = 32    # Có thể lên 48 với PhoBERT-base
    gradient_accumulation_steps: int = 2
    effective_batch_size: int = 128  # 32 * 2 GPUs * 2 accumulation
    
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Training schedule
    epochs: int = 5
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Optimization
    optimizer: str = "AdamW"
    scheduler: str = "linear"
    fp16: bool = True  # Essential cho A5000
    dataloader_num_workers: int = 8
    
    # Multi-GPU settings
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    
    @property
    def total_batch_size(self):
        return self.batch_size_per_gpu * HARDWARE_SPECS["gpu_count"] * self.gradient_accumulation_steps

@dataclass 
class PhoBERTLargeConfig(PhoBERTConfig):
    """Cấu hình cho PhoBERT-large (cần nhiều VRAM hơn)"""
    
    model_name: str = "vinai/phobert-large"
    batch_size_per_gpu: int = 16    # Giảm batch size cho large model
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 128  # 16 * 2 * 4 = 128
    learning_rate: float = 1e-5     # Learning rate thấp hơn cho large model

@dataclass
class XLMRobertaConfig(PhoBERTConfig):
    """Cấu hình cho XLM-RoBERTa"""
    
    model_name: str = "xlm-roberta-base"
    batch_size_per_gpu: int = 24
    gradient_accumulation_steps: int = 3
    learning_rate: float = 1.5e-5

@dataclass
class XLMRobertaLargeConfig(PhoBERTConfig):
    """Cấu hình cho XLM-RoBERTa-large"""
    
    model_name: str = "xlm-roberta-large"
    batch_size_per_gpu: int = 12
    gradient_accumulation_steps: int = 6
    learning_rate: float = 1e-5
    epochs: int = 4  # Ít epochs hơn cho large model

# ==================== LLM CONFIGS WITH LORA ====================

@dataclass
class VietnameseLLMConfig:
    """Base config cho Vietnamese LLMs với LoRA"""
    
    # Model settings
    model_name: str = "vilm/vistral-7b-chat"
    max_length: int = 512
    num_labels: int = 2
    
    # LoRA settings - Quan trọng cho tiết kiệm VRAM
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Training settings cho 7B model
    batch_size_per_gpu: int = 2      # Nhỏ cho 7B model
    gradient_accumulation_steps: int = 32
    effective_batch_size: int = 128   # 2 * 2 * 32 = 128
    
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    
    # Training schedule
    epochs: int = 3
    save_steps: int = 250
    eval_steps: int = 250
    logging_steps: int = 50
    
    # Optimization cho LLM
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    fp16: bool = True
    gradient_checkpointing: bool = True  # Quan trọng cho LLM
    dataloader_num_workers: int = 4
    
    # Multi-GPU
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    
    @property
    def total_batch_size(self):
        return self.batch_size_per_gpu * HARDWARE_SPECS["gpu_count"] * self.gradient_accumulation_steps
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                                       "gate_proj", "up_proj", "down_proj"]

@dataclass
class VistralConfig(VietnameseLLMConfig):
    """Cấu hình cho Vistral-7B"""
    model_name: str = "vilm/vistral-7b-chat"

@dataclass
class VinaLlamaConfig(VietnameseLLMConfig):
    """Cấu hình cho VinaLlama-7B"""
    model_name: str = "vilm/vinallama-7b-chat"

@dataclass
class SeaLLMConfig(VietnameseLLMConfig):
    """Cấu hình cho SeaLLM-7B"""
    model_name: str = "SeaLLMs/SeaLLM-7B-v2-5"
    lora_r: int = 32  # SeaLLM có thể cần LoRA rank cao hơn

# ==================== MEMORY OPTIMIZATION ====================

MEMORY_OPTIMIZATION_CONFIGS = {
    "phobert_base": {
        "gradient_checkpointing": False,
        "flash_attention": True,
        "max_batch_size_per_gpu": 48,
        "estimated_memory_per_gpu": "8GB"
    },
    
    "phobert_large": {
        "gradient_checkpointing": True,
        "flash_attention": True,
        "max_batch_size_per_gpu": 24,
        "estimated_memory_per_gpu": "12GB"
    },
    
    "xlm_roberta_base": {
        "gradient_checkpointing": False,
        "flash_attention": True,
        "max_batch_size_per_gpu": 32,
        "estimated_memory_per_gpu": "10GB"
    },
    
    "xlm_roberta_large": {
        "gradient_checkpointing": True,
        "flash_attention": True,
        "max_batch_size_per_gpu": 16,
        "estimated_memory_per_gpu": "16GB"
    },
    
    "vietnamese_7b_lora": {
        "gradient_checkpointing": True,
        "flash_attention": True,
        "max_batch_size_per_gpu": 4,
        "estimated_memory_per_gpu": "20GB"
    }
}

# ==================== TRAINING STRATEGIES ====================

TRAINING_STRATEGIES = {
    "quick_test": {
        "description": "Test nhanh với PhoBERT-base",
        "config": PhoBERTConfig(),
        "estimated_time": "30 phút",
        "expected_accuracy": "85-88%"
    },
    
    "best_vietnamese": {
        "description": "Tốt nhất cho tiếng Việt",
        "config": PhoBERTLargeConfig(),
        "estimated_time": "2-3 giờ",
        "expected_accuracy": "88-91%"
    },
    
    "multilingual": {
        "description": "Multilingual với XLM-RoBERTa",
        "config": XLMRobertaLargeConfig(),
        "estimated_time": "3-4 giờ", 
        "expected_accuracy": "85-89%"
    },
    
    "llm_lora": {
        "description": "LLM 7B với LoRA",
        "config": VistralConfig(),
        "estimated_time": "4-6 giờ",
        "expected_accuracy": "89-92%"
    }
}

# ==================== UTILITY FUNCTIONS ====================

def get_optimal_config(model_type: str, strategy: str = "balanced") -> Dict[str, Any]:
    """
    Lấy cấu hình tối ưu dựa trên loại model và strategy
    """
    configs = {
        "phobert-base": PhoBERTConfig(),
        "phobert-large": PhoBERTLargeConfig(),
        "xlm-roberta-base": XLMRobertaConfig(),
        "xlm-roberta-large": XLMRobertaLargeConfig(),
        "vistral-7b": VistralConfig(),
        "vinallama-7b": VinaLlamaConfig(),
        "seallm-7b": SeaLLMConfig()
    }
    
    if model_type not in configs:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    config = configs[model_type]
    
    # Điều chỉnh theo strategy
    if strategy == "fast":
        config.epochs = max(1, config.epochs // 2)
        config.eval_steps = config.eval_steps // 2
    elif strategy == "thorough":
        config.epochs = config.epochs + 2
        config.eval_steps = config.eval_steps // 2
    
    return config

def estimate_training_time(config) -> str:
    """
    Ước lượng thời gian training
    """
    # Dựa trên số samples (~3700) và cấu hình
    samples_per_epoch = 2588  # Train set size
    steps_per_epoch = samples_per_epoch // config.total_batch_size
    total_steps = steps_per_epoch * config.epochs
    
    # Ước lượng time per step (seconds)
    if "7b" in config.model_name.lower():
        time_per_step = 3.0  # LLM chậm hơn
    elif "large" in config.model_name.lower():
        time_per_step = 1.5
    else:
        time_per_step = 0.8
    
    total_minutes = (total_steps * time_per_step) / 60
    
    if total_minutes < 60:
        return f"{total_minutes:.0f} phút"
    else:
        hours = total_minutes / 60
        return f"{hours:.1f} giờ"

def print_hardware_recommendations():
    """
    In ra recommendations cho hardware setup
    """
    print("🖥️  HARDWARE RECOMMENDATIONS cho 2x RTX A5000")
    print("=" * 60)
    print(f"✅ GPU Memory: {HARDWARE_SPECS['total_memory']} - Đủ mạnh cho mọi model")
    print(f"✅ Compute: {HARDWARE_SPECS['compute_capability']} - Hỗ trợ mixed precision")
    print(f"✅ Multi-GPU: NCCL backend với 2 cards")
    print("")
    print("📊 CAPACITY ESTIMATES:")
    for model_type, config in MEMORY_OPTIMIZATION_CONFIGS.items():
        print(f"  {model_type}: {config['max_batch_size_per_gpu']} batch/GPU, "
              f"~{config['estimated_memory_per_gpu']} VRAM")
    print("")
    print("⚡ PERFORMANCE TIPS:")
    print("  • Sử dụng mixed precision (fp16) - Tăng tốc 1.5-2x")
    print("  • Gradient checkpointing cho large models")
    print("  • Flash Attention 2 nếu có")
    print("  • DataLoader workers = 8 (CPU cores)")

if __name__ == "__main__":
    print_hardware_recommendations()
    
    print("\n🚀 TRAINING STRATEGIES:")
    print("=" * 60)
    for name, strategy in TRAINING_STRATEGIES.items():
        config = strategy["config"]
        time_est = estimate_training_time(config)
        print(f"📋 {name.upper()}:")
        print(f"  Model: {config.model_name}")
        print(f"  Batch size: {config.total_batch_size}")
        print(f"  Time: {time_est} ({strategy['estimated_time']})")
        print(f"  Expected: {strategy['expected_accuracy']}")
        print("") 