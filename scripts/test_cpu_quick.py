#!/usr/bin/env python3
"""
Quick CPU Test for Vietnamese Clickbait Classification
Test nhanh trên CPU với dữ liệu nhỏ và model nhẹ
"""

import os
import sys
import json
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

def load_data_subset(file_path, n_samples=100):
    """Load subset of data for quick testing"""
    print(f"Loading subset từ {file_path}...")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            data.append(json.loads(line.strip()))
    
    texts = [item['title'] for item in data]
    # Convert string labels to integers: "clickbait" -> 1, "non-clickbait" -> 0
    labels = [1 if item['label'] == 'clickbait' else 0 for item in data]
    
    print(f"Loaded {len(data)} samples")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return texts, labels

def compute_metrics(eval_pred):
    """Compute accuracy and other metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

def main():
    print("=" * 60)
    print("🚀 VIETNAMESE CLICKBAIT CLASSIFICATION - CPU TEST")
    print("=" * 60)
    
    # Kiểm tra môi trường
    print(f"🔧 Python version: {sys.version}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    print(f"🔧 Device: CPU (forced)")
    print(f"🔧 Threads: {torch.get_num_threads()}")
    
    device = torch.device('cpu')
    
    # Cấu hình cho CPU test
    CONFIG = {
        'model_name': 'vinai/phobert-base',  # Model nhẹ nhất
        'max_length': 64,  # Giảm max_length
        'batch_size': 8,   # Batch size nhỏ
        'num_epochs': 2,   # Ít epoch
        'learning_rate': 2e-5,
        'train_samples': 200,  # Ít dữ liệu
        'val_samples': 50,
        'test_samples': 50
    }
    
    print(f"\n📋 CPU Test Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # Đường dẫn dữ liệu
    data_dir = "data"
    train_file = os.path.join(data_dir, "train_dtVN", "training_data.jsonl")
    val_file = os.path.join(data_dir, "val_dtVN", "training_data.jsonl")
    test_file = os.path.join(data_dir, "test_dtVN", "training_data.jsonl")
    
    # Kiểm tra dữ liệu
    for file_path in [train_file, val_file, test_file]:
        if not os.path.exists(file_path):
            print(f"❌ Không tìm thấy file: {file_path}")
            print("💡 Chạy scripts/prepare_training_data.py trước")
            return
    
    try:
        # Load dữ liệu subset
        print(f"\n📂 Loading data subsets...")
        train_texts, train_labels = load_data_subset(train_file, CONFIG['train_samples'])
        val_texts, val_labels = load_data_subset(val_file, CONFIG['val_samples'])
        test_texts, test_labels = load_data_subset(test_file, CONFIG['test_samples'])
        
        # Load tokenizer và model
        print(f"\n🤖 Loading model: {CONFIG['model_name']}")
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
        model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG['model_name'],
            num_labels=2
        ).to(device)
        
        print(f"📊 Model parameters: {model.num_parameters():,}")
        
        # Tạo datasets
        print(f"\n📊 Creating datasets...")
        train_dataset = ClickbaitDataset(train_texts, train_labels, tokenizer, CONFIG['max_length'])
        val_dataset = ClickbaitDataset(val_texts, val_labels, tokenizer, CONFIG['max_length'])
        test_dataset = ClickbaitDataset(test_texts, test_labels, tokenizer, CONFIG['max_length'])
        
        # Training arguments for CPU
        training_args = TrainingArguments(
            output_dir='./results_cpu_test',
            num_train_epochs=CONFIG['num_epochs'],
            per_device_train_batch_size=CONFIG['batch_size'],
            per_device_eval_batch_size=CONFIG['batch_size'],
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir='./logs_cpu_test',
            logging_steps=10,
            eval_strategy="epoch",  # Changed from evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            dataloader_num_workers=0,  # Tắt multiprocessing cho CPU
            fp16=False,  # Tắt mixed precision
            report_to=None,  # Tắt wandb/tensorboard
            save_total_limit=1
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Training
        print(f"\n🏋️ Starting training...")
        start_time = datetime.now()
        
        trainer.train()
        
        end_time = datetime.now()
        training_time = end_time - start_time
        print(f"✅ Training completed in: {training_time}")
        
        # Evaluation trên test set
        print(f"\n📊 Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        
        print(f"\n🎯 Test Results:")
        print(f"   Accuracy: {test_results['eval_accuracy']:.4f}")
        print(f"   Loss: {test_results['eval_loss']:.4f}")
        
        # Detailed predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = test_labels
        
        print(f"\n📈 Detailed Classification Report:")
        print(classification_report(y_true, y_pred, 
                                  target_names=['Non-Clickbait', 'Clickbait']))
        
        print(f"\n🎯 Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(f"   Non-CB predicted: {cm[0]}")
        print(f"   Clickbait predicted: {cm[1]}")
        
        # Test một vài examples
        print(f"\n🔍 Sample Predictions:")
        for i in range(min(5, len(test_texts))):
            text = test_texts[i]
            true_label = "Clickbait" if test_labels[i] == 1 else "Non-Clickbait"
            pred_label = "Clickbait" if y_pred[i] == 1 else "Non-Clickbait"
            status = "✅" if y_pred[i] == test_labels[i] else "❌"
            
            print(f"   {status} Text: {text[:50]}...")
            print(f"      True: {true_label}, Pred: {pred_label}")
        
        # Summary
        print(f"\n" + "="*60)
        print(f"📋 CPU TEST SUMMARY")
        print(f"="*60)
        print(f"🚀 Model: {CONFIG['model_name']}")
        print(f"⏰ Training time: {training_time}")
        print(f"🎯 Test accuracy: {test_results['eval_accuracy']:.4f}")
        print(f"📊 Train samples: {len(train_texts)}")
        print(f"📊 Val samples: {len(val_texts)}")
        print(f"📊 Test samples: {len(test_texts)}")
        print(f"💻 Device: CPU")
        print(f"⚙️ Batch size: {CONFIG['batch_size']}")
        print(f"🔄 Epochs: {CONFIG['num_epochs']}")
        
        print(f"\n✅ CPU test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 