#!/usr/bin/env python3
"""
General Vietnamese Clickbait Training - NO UNDERTHESEA
Script training tổng quát không sử dụng underthesea, chỉ dùng built-in tokenizer
Hỗ trợ CPU/GPU, multiple models, full dataset training
"""

import os
import sys
import json
import argparse
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np
from datetime import datetime
from pathlib import Path
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

def load_dataset_split(file_path):
    """Load full dataset split"""
    print(f"Loading dataset từ {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    texts = [item['title'] for item in data]
    # Convert string labels to integers: "clickbait" -> 1, "non-clickbait" -> 0  
    labels = [1 if item['label'] == 'clickbait' else 0 for item in data]
    
    print(f"Loaded {len(data)} samples")
    print(f"Label distribution: {np.bincount(labels)} (0=non-clickbait, 1=clickbait)")
    
    return texts, labels

def compute_metrics(eval_pred):
    """Compute comprehensive metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def get_model_config(model_name, device_type='cpu'):
    """Get model configuration"""
    configs = {
        'vinai/phobert-base': {
            'description': 'PhoBERT-base without word segmentation',
            'max_length': 128,
            'batch_size_cpu': 16,
            'batch_size_gpu': 32,
            'expected_accuracy': '75-85%'
        },
        'vinai/phobert-large': {
            'description': 'PhoBERT-large without word segmentation', 
            'max_length': 128,
            'batch_size_cpu': 8,
            'batch_size_gpu': 16,
            'expected_accuracy': '80-90%'
        },
        'xlm-roberta-base': {
            'description': 'XLM-RoBERTa-base multilingual (no segmentation needed)',
            'max_length': 128,
            'batch_size_cpu': 12,
            'batch_size_gpu': 24,
            'expected_accuracy': '75-85%'
        },
        'xlm-roberta-large': {
            'description': 'XLM-RoBERTa-large multilingual (no segmentation needed)',
            'max_length': 128,
            'batch_size_cpu': 6,
            'batch_size_gpu': 12,
            'expected_accuracy': '80-90%'
        }
    }
    
    if model_name not in configs:
        print(f"⚠️ Unknown model {model_name}, using default config")
        return {
            'description': 'Custom model',
            'max_length': 128,
            'batch_size_cpu': 8,
            'batch_size_gpu': 16,
            'expected_accuracy': 'Unknown'
        }
    
    config = configs[model_name].copy()
    config['batch_size'] = config[f'batch_size_{device_type}']
    return config

def save_model_and_results(model, tokenizer, trainer, test_results, args, config):
    """Save trained model and results"""
    
    # Create output directory
    model_name_safe = args.model_name.replace('/', '_')
    output_dir = f"outputs_no_underthesea/{model_name_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    print(f"💾 Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training results
    results = {
        'model_name': args.model_name,
        'model_config': config,
        'training_args': {
            'epochs': args.epochs,
            'batch_size': config['batch_size'],
            'learning_rate': args.learning_rate,
            'max_length': config['max_length'],
            'device': str(args.device)
        },
        'test_results': test_results,
        'training_time': str(datetime.now()),
        'method': 'NO_UNDERTHESEA',
        'dataset_info': {
            'data_dir': args.data_dir,
            'train_samples': len(trainer.train_dataset) if trainer.train_dataset else 0,
            'val_samples': len(trainer.eval_dataset) if trainer.eval_dataset else 0
        }
    }
    
    with open(f"{output_dir}/training_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Model và results đã lưu tại: {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Vietnamese Clickbait Training - No Underthesea")
    
    # Model arguments
    parser.add_argument("--model_name", default="vinai/phobert-base",
                       choices=[
                           "vinai/phobert-base", 
                           "vinai/phobert-large",
                           "xlm-roberta-base",
                           "xlm-roberta-large"
                       ],
                       help="Model to train")
    
    # Data arguments
    parser.add_argument("--data_dir", default="data", 
                       help="Data directory containing train_dtVN, val_dtVN, test_dtVN")
    parser.add_argument("--subset_size", type=int, default=None,
                       help="Use subset of data for quick testing (None = full dataset)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device to use")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (auto-detected based on model and device if None)")
    
    # Output arguments
    parser.add_argument("--output_dir", default=None,
                       help="Output directory (auto-generated if None)")
    parser.add_argument("--save_model", action="store_true",
                       help="Save trained model")
    
    # Evaluation arguments
    parser.add_argument("--skip_test", action="store_true",
                       help="Skip test evaluation")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 VIETNAMESE CLICKBAIT TRAINING - NO UNDERTHESEA")
    print("=" * 70)
    
    # Detect device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_type = "gpu" if torch.cuda.is_available() else "cpu"
    else:
        device = torch.device(args.device)
        device_type = "gpu" if args.device == "cuda" else "cpu"
    
    print(f"🔧 Python version: {sys.version}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    print(f"🔧 Device: {device} ({device_type})")
    print(f"✅ NO UNDERTHESEA - Sử dụng built-in tokenizer")
    
    # Get model configuration
    config = get_model_config(args.model_name, device_type)
    if args.batch_size:
        config['batch_size'] = args.batch_size
        
    print(f"\n📋 Model Configuration:")
    print(f"   Model: {args.model_name}")
    print(f"   Description: {config['description']}")
    print(f"   Expected accuracy: {config['expected_accuracy']}")
    print(f"   Max length: {config['max_length']}")
    print(f"   Batch size: {config['batch_size']}")
    
    print(f"\n📋 Training Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Device: {device}")
    print(f"   Subset size: {args.subset_size or 'Full dataset'}")
    
    # Data paths
    train_file = os.path.join(args.data_dir, "train_dtVN", "training_data.jsonl")
    val_file = os.path.join(args.data_dir, "val_dtVN", "training_data.jsonl")
    test_file = os.path.join(args.data_dir, "test_dtVN", "training_data.jsonl")
    
    # Check data files
    for file_path, name in [(train_file, "train"), (val_file, "val"), (test_file, "test")]:
        if not os.path.exists(file_path):
            print(f"❌ Không tìm thấy {name} file: {file_path}")
            print("💡 Chạy scripts/prepare_training_data.py trước")
            return
    
    try:
        # Load data
        print(f"\n📂 Loading datasets...")
        train_texts, train_labels = load_dataset_split(train_file)
        val_texts, val_labels = load_dataset_split(val_file)
        if not args.skip_test:
            test_texts, test_labels = load_dataset_split(test_file)
        
        # Use subset if specified
        if args.subset_size:
            print(f"\n📏 Using subset of {args.subset_size} samples for quick testing...")
            train_texts = train_texts[:args.subset_size]
            train_labels = train_labels[:args.subset_size]
            val_texts = val_texts[:args.subset_size//4]
            val_labels = val_labels[:args.subset_size//4]
            if not args.skip_test:
                test_texts = test_texts[:args.subset_size//4]
                test_labels = test_labels[:args.subset_size//4]
        
        # Load model and tokenizer
        print(f"\n🤖 Loading model: {args.model_name}")
        print(f"🔍 Method: Raw text → Tokenizer (NO word segmentation)")
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2
        ).to(device)
        
        print(f"📊 Model parameters: {model.num_parameters():,}")
        
        # Show sample processing
        print(f"\n🔍 Sample text processing:")
        for i, text in enumerate(train_texts[:3]):
            print(f"   {i+1}. Original: {text[:50]}...")
            tokens = tokenizer.tokenize(text)
            print(f"      Tokenized: {' '.join(tokens[:8])}...")
            print(f"      Length: {len(tokens)} tokens")
        
        # Create datasets
        print(f"\n📊 Creating datasets...")
        train_dataset = ClickbaitDataset(train_texts, train_labels, tokenizer, config['max_length'])
        val_dataset = ClickbaitDataset(val_texts, val_labels, tokenizer, config['max_length'])
        if not args.skip_test:
            test_dataset = ClickbaitDataset(test_texts, test_labels, tokenizer, config['max_length'])
        
        # Training arguments
        output_dir = args.output_dir or f"./results_general_no_underthesea/{args.model_name.replace('/', '_')}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'./logs_general_no_underthesea/{args.model_name.replace("/", "_")}',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            dataloader_num_workers=0 if device_type == "cpu" else 2,
            fp16=device_type == "gpu",  # Use mixed precision on GPU
            report_to=None,  # Disable wandb/tensorboard  
            save_total_limit=2,
            eval_accumulation_steps=10 if device_type == "cpu" else None
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Callbacks
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        
        # Training
        print(f"\n🏋️ Starting training...")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        if not args.skip_test:
            print(f"   Test samples: {len(test_dataset)}")
        
        start_time = datetime.now()
        trainer.train()
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"✅ Training completed in: {training_time}")
        
        # Test evaluation
        test_results = {}
        if not args.skip_test:
            print(f"\n📊 Evaluating on test set...")
            test_results = trainer.evaluate(test_dataset)
            
            print(f"\n🎯 Test Results:")
            print(f"   Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
            print(f"   F1-Score: {test_results.get('eval_f1', 0):.4f}")
            print(f"   Loss: {test_results.get('eval_loss', 0):.4f}")
            
            # Detailed analysis
            predictions = trainer.predict(test_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = test_labels if not args.subset_size else test_labels[:len(y_pred)]
            
            print(f"\n📈 Detailed Classification Report:")
            print(classification_report(y_true, y_pred, 
                                      target_names=['Non-Clickbait', 'Clickbait']))
            
            print(f"\n🎯 Confusion Matrix:")
            cm = confusion_matrix(y_true, y_pred)
            print(f"   Non-CB predicted: {cm[0]}")
            print(f"   Clickbait predicted: {cm[1]}")
            
            # Sample predictions
            print(f"\n🔍 Sample Predictions:")
            for i in range(min(5, len(test_texts))):
                text = test_texts[i]
                true_label = "Clickbait" if test_labels[i] == 1 else "Non-Clickbait"
                pred_label = "Clickbait" if y_pred[i] == 1 else "Non-Clickbait"
                status = "✅" if y_pred[i] == test_labels[i] else "❌"
                
                print(f"   {status} Text: {text[:50]}...")
                print(f"      True: {true_label}, Pred: {pred_label}")
        
        # Save model if requested
        saved_dir = None
        if args.save_model:
            saved_dir = save_model_and_results(model, tokenizer, trainer, test_results, args, config)
        
        # Final summary
        print(f"\n" + "="*70)
        print(f"📋 TRAINING SUMMARY (NO UNDERTHESEA)")
        print(f"="*70)
        print(f"🚀 Model: {args.model_name}")
        print(f"🔄 Method: NO word segmentation (raw text)")
        print(f"⏰ Training time: {training_time}")
        print(f"🎯 Test accuracy: {test_results.get('eval_accuracy', 'N/A')}")
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Val samples: {len(val_dataset)}")
        if not args.skip_test:
            print(f"📊 Test samples: {len(test_dataset)}")
        print(f"💻 Device: {device}")
        print(f"⚙️ Batch size: {config['batch_size']}")
        print(f"🔄 Epochs: {args.epochs}")
        print(f"📈 Expected accuracy: {config['expected_accuracy']}")
        print(f"✅ Dependencies: torch + transformers (NO underthesea)")
        if saved_dir:
            print(f"💾 Model saved: {saved_dir}")
        
        print(f"\n✅ Training completed successfully!")
        
        return {
            'accuracy': test_results.get('eval_accuracy', 0),
            'f1': test_results.get('eval_f1', 0),
            'training_time': training_time,
            'saved_dir': saved_dir
        }
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 