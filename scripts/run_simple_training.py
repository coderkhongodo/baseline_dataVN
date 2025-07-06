#!/usr/bin/env python3
"""
Simple Vietnamese Clickbait Training - NO UNDERTHESEA
Script training đơn giản, không dùng underthesea, dễ customize
"""

import os
import sys
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from datetime import datetime

class ClickbaitDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path, max_samples=None):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line.strip()))
    
    texts = [item['title'] for item in data]
    labels = [1 if item['label'] == 'clickbait' else 0 for item in data]
    
    return texts, labels

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {'accuracy': accuracy, 'f1': f1}

def main():
    # =============================================================================
    # 🔧 CẤU HÌNH - THAY ĐỔI TẠI ĐÂY
    # =============================================================================
    
    CONFIG = {
        # Model settings
        'model_name': 'vinai/phobert-base',  # hoặc 'xlm-roberta-base'
        'max_length': 128,
        
        # Training settings
        'epochs': 3,
        'batch_size': 16,  # Giảm xuống nếu out of memory
        'learning_rate': 2e-5,
        
        # Data settings
        'data_dir': 'data',
        'use_subset': False,  # True = training nhanh với subset
        'subset_size': 500,   # Chỉ dùng khi use_subset=True
        
        # Device
        'device': 'auto',  # 'auto', 'cpu', hoặc 'cuda'
        
        # Output
        'save_model': True,
        'output_name': None  # None = auto-generate
    }
    
    # =============================================================================
    # 🚀 TRAINING
    # =============================================================================
    
    print("🚀 SIMPLE VIETNAMESE CLICKBAIT TRAINING (NO UNDERTHESEA)")
    print("=" * 60)
    
    # Device setup
    if CONFIG['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG['device'])
    
    print(f"💻 Device: {device}")
    print(f"🤖 Model: {CONFIG['model_name']}")
    print(f"📊 Epochs: {CONFIG['epochs']}, Batch: {CONFIG['batch_size']}")
    print(f"✅ Method: RAW text → Tokenizer (NO underthesea)")
    
    # Data paths
    train_file = os.path.join(CONFIG['data_dir'], "train_dtVN", "training_data.jsonl")
    val_file = os.path.join(CONFIG['data_dir'], "val_dtVN", "training_data.jsonl")
    test_file = os.path.join(CONFIG['data_dir'], "test_dtVN", "training_data.jsonl")
    
    # Load data
    print(f"\n📂 Loading data...")
    max_samples = CONFIG['subset_size'] if CONFIG['use_subset'] else None
    
    train_texts, train_labels = load_data(train_file, max_samples)
    val_texts, val_labels = load_data(val_file, max_samples//4 if max_samples else None)
    test_texts, test_labels = load_data(test_file, max_samples//4 if max_samples else None)
    
    print(f"   Train: {len(train_texts)} samples")
    print(f"   Val: {len(val_texts)} samples") 
    print(f"   Test: {len(test_texts)} samples")
    
    # Load model
    print(f"\n🔄 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'], 
        num_labels=2
    ).to(device)
    
    print(f"   Parameters: {model.num_parameters():,}")
    
    # Create datasets
    train_dataset = ClickbaitDataset(train_texts, train_labels, tokenizer, CONFIG['max_length'])
    val_dataset = ClickbaitDataset(val_texts, val_labels, tokenizer, CONFIG['max_length'])
    test_dataset = ClickbaitDataset(test_texts, test_labels, tokenizer, CONFIG['max_length'])
    
    # Training arguments
    output_dir = CONFIG['output_name'] or f"outputs_simple/{CONFIG['model_name'].replace('/', '_')}_{datetime.now().strftime('%m%d_%H%M')}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate'],
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=device.type == 'cuda',
        dataloader_num_workers=0,
        report_to=None,
        save_total_limit=1
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # Training
    print(f"\n🏋️ Training...")
    start_time = datetime.now()
    
    trainer.train()
    
    training_time = datetime.now() - start_time
    print(f"✅ Training completed in: {training_time}")
    
    # Test evaluation
    print(f"\n📊 Testing...")
    test_results = trainer.evaluate(test_dataset)
    
    accuracy = test_results['eval_accuracy']
    f1 = test_results['eval_f1']
    
    print(f"🎯 Test Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Save model
    if CONFIG['save_model']:
        print(f"\n💾 Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save config and results
        results = {
            'config': CONFIG,
            'accuracy': accuracy,
            'f1': f1,
            'training_time': str(training_time),
            'model_parameters': model.num_parameters(),
            'device': str(device)
        }
        
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n" + "="*60)
    print(f"📋 SUMMARY")
    print(f"="*60)
    print(f"🚀 Model: {CONFIG['model_name']}")
    print(f"⏰ Time: {training_time}")
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"📈 F1-Score: {f1:.4f}")
    print(f"💻 Device: {device}")
    print(f"📊 Samples: {len(train_texts)} train, {len(test_texts)} test")
    print(f"✅ Method: NO underthesea")
    if CONFIG['save_model']:
        print(f"💾 Saved: {output_dir}")
    
    print(f"\n🎉 Done!")

if __name__ == "__main__":
    main() 