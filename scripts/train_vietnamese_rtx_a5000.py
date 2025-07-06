#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script huấn luyện model phân loại clickbait tiếng Việt
Tối ưu cho 2x RTX A5000 (48GB VRAM total)
Hỗ trợ: PhoBERT, XLM-RoBERTa, Vietnamese LLMs với LoRA
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
    EarlyStoppingCallback, DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from datasets import Dataset as HFDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Import local configs
sys.path.append(str(Path(__file__).parent.parent))
from configs.rtx_a5000_configs import (
    PhoBERTConfig, PhoBERTLargeConfig, XLMRobertaConfig, XLMRobertaLargeConfig,
    VistralConfig, VinaLlamaConfig, SeaLLMConfig,
    get_optimal_config, HARDWARE_SPECS
)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ClickbaitDataset(Dataset):
    """Dataset cho phân loại clickbait tiếng Việt"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dữ liệu
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if item.get('title') and item.get('label'):
                        self.data.append(item)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
        
        # Label mapping
        self.label2id = {"non-clickbait": 0, "clickbait": 1}
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item['title'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label2id[item['label']], dtype=torch.long)
        }

def setup_model_and_tokenizer(config) -> Tuple[Any, Any]:
    """
    Setup model và tokenizer dựa trên config
    """
    logger.info(f"Setting up model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Thêm padding token nếu chưa có
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if hasattr(config, 'use_lora') and config.use_lora:
        # LLM với LoRA
        from peft import LoraConfig, get_peft_model, TaskType
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules
        )
        
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        
    else:
        # BERT family models
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            torch_dtype=torch.float16 if config.fp16 else torch.float32
        )
    
    logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters():,}")
    
    return model, tokenizer

def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Tính toán metrics cho evaluation
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'f1_clickbait': f1_per_class[1],
        'f1_non_clickbait': f1_per_class[0],
        'precision_clickbait': precision_per_class[1],
        'recall_clickbait': recall_per_class[1]
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return metrics

def setup_training_arguments(config, output_dir: str) -> TrainingArguments:
    """
    Setup training arguments cho Trainer
    """
    # Tính toán số steps
    train_samples = 2588  # Từ dữ liệu đã chia
    steps_per_epoch = train_samples // config.total_batch_size
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    args = TrainingArguments(
        output_dir=output_dir,
        
        # Training schedule
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size_per_gpu,
        per_device_eval_batch_size=config.batch_size_per_gpu,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Learning rate và optimization
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=warmup_steps,
        lr_scheduler_type=config.scheduler,
        
        # Evaluation và saving
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        
        # Performance optimization
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        gradient_checkpointing=getattr(config, 'gradient_checkpointing', False),
        
        # Multi-GPU settings
        ddp_backend=config.ddp_backend,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        
        # Monitoring
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # Saving
        save_total_limit=3,
        
        # Reporting
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
        
        # Other
        remove_unused_columns=False,
        push_to_hub=False
    )
    
    return args

def train_model(
    config,
    train_data_file: str,
    val_data_file: str,
    output_dir: str,
    model_name_suffix: str = ""
) -> str:
    """
    Main training function
    """
    logger.info("=" * 80)
    logger.info(f"🚀 STARTING TRAINING: {config.model_name}")
    logger.info(f"📊 Hardware: {HARDWARE_SPECS['gpu_count']}x {HARDWARE_SPECS['gpu_model']}")
    logger.info(f"🎯 Total batch size: {config.total_batch_size}")
    logger.info("=" * 80)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_simple_name = config.model_name.split('/')[-1]
    output_dir = Path(output_dir) / f"{model_simple_name}_{timestamp}{model_name_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup model và tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load datasets
    train_dataset = ClickbaitDataset(train_data_file, tokenizer, config.max_length)
    val_dataset = ClickbaitDataset(val_data_file, tokenizer, config.max_length)
    
    logger.info(f"📁 Train samples: {len(train_dataset)}")
    logger.info(f"📁 Val samples: {len(val_dataset)}")
    
    # Setup training arguments
    training_args = setup_training_arguments(config, str(output_dir))
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Save config
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(vars(config), f, indent=2, default=str)
    
    # Start training
    logger.info("🎯 Starting training...")
    start_time = datetime.now()
    
    try:
        trainer.train()
        
        # Save final model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        training_time = datetime.now() - start_time
        logger.info(f"✅ Training completed in {training_time}")
        
        # Final evaluation
        logger.info("📊 Running final evaluation...")
        eval_results = trainer.evaluate()
        
        # Save results
        with open(output_dir / "final_results.json", 'w') as f:
            json.dump({
                'eval_results': eval_results,
                'training_time': str(training_time),
                'config': vars(config)
            }, f, indent=2, default=str)
        
        logger.info(f"🎉 Model saved to: {output_dir}")
        logger.info(f"📊 Final F1: {eval_results.get('eval_f1', 0):.4f}")
        logger.info(f"📊 Final Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
        
        return str(output_dir)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

def test_sample_predictions(model_dir: str, test_samples: List[str]):
    """
    Test model với một vài samples
    """
    logger.info(f"🧪 Testing model: {model_dir}")
    
    # Load model và tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    label_map = {0: "non-clickbait", 1: "clickbait"}
    
    logger.info("📝 Sample predictions:")
    for i, sample in enumerate(test_samples):
        inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        result = label_map[predicted_class]
        logger.info(f"  {i+1}. '{sample[:60]}...'")
        logger.info(f"     -> {result} (confidence: {confidence:.3f})")

def main():
    parser = argparse.ArgumentParser(description="Train Vietnamese Clickbait Classification Model")
    parser.add_argument("--model-type", "-m", required=True,
                       choices=["phobert-base", "phobert-large", "xlm-roberta-base", 
                               "xlm-roberta-large", "vistral-7b", "vinallama-7b", "seallm-7b"],
                       help="Model type to train")
    parser.add_argument("--strategy", "-s", default="balanced",
                       choices=["fast", "balanced", "thorough"],
                       help="Training strategy")
    parser.add_argument("--train-data", default="data/train_dtVN/training_data.jsonl",
                       help="Training data file")
    parser.add_argument("--val-data", default="data/val_dtVN/training_data.jsonl",
                       help="Validation data file")
    parser.add_argument("--output-dir", "-o", default="models/vietnamese_clickbait",
                       help="Output directory")
    parser.add_argument("--test-samples", action="store_true",
                       help="Test với sample predictions sau khi train")
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! This script requires GPU.")
        sys.exit(1)
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"🖥️  Detected {gpu_count} GPU(s)")
    
    if gpu_count != 2:
        logger.warning(f"⚠️  Expected 2 RTX A5000 GPUs, found {gpu_count}")
    
    # Check data files
    if not Path(args.train_data).exists():
        logger.error(f"❌ Training data not found: {args.train_data}")
        logger.info("💡 Run: python scripts/prepare_training_data.py")
        sys.exit(1)
    
    if not Path(args.val_data).exists():
        logger.error(f"❌ Validation data not found: {args.val_data}")
        sys.exit(1)
    
    # Get config
    config = get_optimal_config(args.model_type, args.strategy)
    
    # Train model
    try:
        model_dir = train_model(
            config=config,
            train_data_file=args.train_data,
            val_data_file=args.val_data,
            output_dir=args.output_dir,
            model_name_suffix=f"_{args.strategy}"
        )
        
        # Test samples
        if args.test_samples:
            test_samples = [
                "Bí mật kinh hoàng mà bác sĩ không bao giờ tiết lộ!",
                "Nghiên cứu mới về tác động của caffeine đến sức khỏe",
                "Cách làm giàu nhanh trong 30 ngày mà ai cũng có thể làm",
                "Báo cáo kinh tế quý II năm 2025 của Ngân hàng Thế giới",
                "Sốc: Phát hiện bí mật động trời về sao Việt nổi tiếng!"
            ]
            test_sample_predictions(model_dir, test_samples)
        
        logger.info("🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 