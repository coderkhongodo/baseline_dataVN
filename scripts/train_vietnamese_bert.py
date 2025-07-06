#!/usr/bin/env python3
"""
Train Vietnamese BERT Family Models for Clickbait Classification
Optimized for Vietnamese text processing with PhoBERT and multilingual models
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import underthesea
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

# Add configs to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.vietnamese_models import VIETNAMESE_BERT_CONFIGS, get_vietnamese_model_config

class VietnameseTextProcessor:
    """Vietnamese text processor for training"""
    
    def __init__(self, preprocessing_method: str = "word_segmentation"):
        self.preprocessing_method = preprocessing_method
        
    def preprocess_for_model(self, text: str, model_name: str) -> str:
        """Preprocess text dựa trên model requirements"""
        
        if "phobert" in model_name.lower() and self.preprocessing_method == "word_segmentation":
            # PhoBERT cần word segmentation
            try:
                segmented = underthesea.word_tokenize(text, format="text")
                return segmented
            except Exception as e:
                print(f"⚠️ Word segmentation failed: {e}")
                return text
        else:
            # Các models khác không cần word segmentation
            return text

def setup_vietnamese_model_and_tokenizer(config):
    """Setup Vietnamese model and tokenizer"""
    print(f"🇻🇳 Loading Vietnamese model: {config['model_name']}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=2
        )
        
        print(f"✅ Model loaded successfully")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        print(f"   Max position embeddings: {getattr(model.config, 'max_position_embeddings', 'N/A')}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading Vietnamese model: {e}")
        raise

def prepare_vietnamese_dataset(tokenizer, config, data_dir="data_vietnamese"):
    """Load and tokenize Vietnamese dataset"""
    print("📊 Loading Vietnamese dataset...")
    
    data_files = {
        "train": f"{data_dir}/train/data.jsonl",
        "validation": f"{data_dir}/val/data.jsonl",
        "test": f"{data_dir}/test/data.jsonl"
    }
    
    # Check if data files exist
    for split, file_path in data_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Vietnamese data file not found: {file_path}")
    
    dataset = load_dataset("json", data_files=data_files)
    
    # Setup text processor
    text_processor = VietnameseTextProcessor(config.get("preprocessing", "none"))
    
    def tokenize_function(examples):
        # Preprocess Vietnamese text if needed
        if config.get("preprocessing") == "word_segmentation":
            processed_texts = [
                text_processor.preprocess_for_model(text, config["model_name"]) 
                for text in examples["text"]
            ]
        else:
            processed_texts = examples["text"]
        
        return tokenizer(
            processed_texts,
            truncation=True,
            max_length=config["max_length"],
            padding=True
        )
    
    print("🔄 Tokenizing Vietnamese text...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Print dataset info
    for split in ["train", "validation", "test"]:
        clickbait_count = sum(tokenized_dataset[split]["label"])
        total_count = len(tokenized_dataset[split])
        print(f"✅ {split}: {total_count} samples ({clickbait_count} clickbait, {total_count-clickbait_count} no-clickbait)")
    
    return tokenized_dataset

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='weighted'),
        "f1_macro": f1_score(labels, predictions, average='macro'),
        "f1_binary": f1_score(labels, predictions, average='binary'),
        "f1_clickbait": f1_score(labels, predictions, pos_label=1),
        "f1_no_clickbait": f1_score(labels, predictions, pos_label=0)
    }

def train_vietnamese_model(model_key, hardware="rtx_4090", output_base_dir="outputs_vietnamese", data_dir="data_vietnamese"):
    """Train a Vietnamese model"""
    print(f"\n{'='*70}")
    print(f"🇻🇳 TRAINING VIETNAMESE MODEL: {model_key.upper()}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Get config for model and hardware
    config = get_vietnamese_model_config(model_key, hardware)
    print(f"📋 Model config: {config['description']}")
    print(f"🔧 Hardware: {hardware}")
    print(f"📊 Batch size: {config['batch_size']}, Max length: {config['max_length']}")
    
    # Setup paths
    output_dir = f"{output_base_dir}/{model_key}-vietnamese"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup model and tokenizer
    model, tokenizer = setup_vietnamese_model_and_tokenizer(config)
    
    # Prepare dataset
    tokenized_dataset = prepare_vietnamese_dataset(tokenizer, config, data_dir)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        warmup_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=config["fp16"],
        dataloader_num_workers=4,
        gradient_checkpointing=True,  # Save VRAM
        report_to=None,
        save_total_limit=2,  # Keep only best 2 checkpoints
        logging_dir=f"{output_dir}/logs",
        run_name=f"{model_key}-vietnamese-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("🏋️ Starting Vietnamese model training...")
    train_result = trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate on test set
    print("🧪 Evaluating on Vietnamese test set...")
    test_results = trainer.predict(tokenized_dataset["test"])
    test_predictions = test_results.predictions.argmax(axis=-1)
    test_labels = tokenized_dataset["test"]["label"]
    
    # Calculate metrics
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_f1 = f1_score(test_labels, test_predictions, average='weighted')
    test_f1_macro = f1_score(test_labels, test_predictions, average='macro')
    test_f1_binary = f1_score(test_labels, test_predictions, average='binary')
    test_f1_clickbait = f1_score(test_labels, test_predictions, pos_label=1)
    test_f1_no_clickbait = f1_score(test_labels, test_predictions, pos_label=0)
    
    # Training time
    training_time = time.time() - start_time
    
    # Detailed classification report
    class_report = classification_report(
        test_labels, 
        test_predictions,
        target_names=["Không phải clickbait", "Clickbait"],
        output_dict=True
    )
    
    # Save results
    results = {
        "model_name": config["model_name"],
        "model_key": model_key,
        "language": "vietnamese",
        "hardware": hardware,
        "training_config": config,
        "training_time_seconds": training_time,
        "training_time_formatted": f"{training_time//3600:.0f}h {(training_time%3600)//60:.0f}m {training_time%60:.0f}s",
        "test_metrics": {
            "accuracy": float(test_accuracy),
            "f1_weighted": float(test_f1),
            "f1_macro": float(test_f1_macro),
            "f1_binary": float(test_f1_binary),
            "f1_clickbait": float(test_f1_clickbait),
            "f1_no_clickbait": float(test_f1_no_clickbait)
        },
        "classification_report": class_report,
        "dataset_info": {
            "train_samples": len(tokenized_dataset["train"]),
            "val_samples": len(tokenized_dataset["validation"]),
            "test_samples": len(tokenized_dataset["test"])
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results to file
    results_file = f"{output_dir}/vietnamese_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"🎉 VIETNAMESE TRAINING COMPLETED: {model_key}")
    print(f"{'='*70}")
    print(f"📁 Model saved to: {output_dir}")
    print(f"⏱️  Training time: {results['training_time_formatted']}")
    print(f"🎯 Test Results:")
    print(f"   Accuracy: {test_accuracy:.4f}")
    print(f"   F1-score: {test_f1:.4f}")
    print(f"   F1 Clickbait: {test_f1_clickbait:.4f}")
    print(f"   F1 No-clickbait: {test_f1_no_clickbait:.4f}")
    
    # Vietnamese-specific sample predictions
    test_sample_vietnamese_predictions(trainer, tokenizer, config)
    
    return results

def test_sample_vietnamese_predictions(trainer, tokenizer, config):
    """Test model trên Vietnamese samples"""
    print(f"\n🧪 Testing on Vietnamese sample headlines:")
    
    vietnamese_samples = [
        {"text": "Chính phủ thông qua nghị định mới về thuế môi trường", "expected": 0},
        {"text": "Bạn sẽ không tin được điều mà cô gái này làm để kiếm tiền!", "expected": 1},
        {"text": "BIDV tăng lãi suất tiết kiệm lên 7.2% từ tháng tới", "expected": 0},
        {"text": "7 bí mật mà bác sĩ không muốn bạn biết về sức khỏe", "expected": 1},
        {"text": "Đội tuyển Việt Nam thắng 2-1 trước Thái Lan tại AFF Cup", "expected": 0},
        {"text": "Cách làm giàu mà 90% người Việt chưa biết - Bạn có tin không?", "expected": 1}
    ]
    
    # Process samples
    text_processor = VietnameseTextProcessor(config.get("preprocessing", "none"))
    
    for i, sample in enumerate(vietnamese_samples, 1):
        original_text = sample["text"]
        
        # Preprocess if needed
        if config.get("preprocessing") == "word_segmentation":
            processed_text = text_processor.preprocess_for_model(original_text, config["model_name"])
        else:
            processed_text = original_text
        
        # Tokenize and predict
        inputs = tokenizer(processed_text, return_tensors="pt", max_length=config["max_length"], truncation=True)
        with torch.no_grad():
            outputs = trainer.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # Results
        expected = sample["expected"]
        result_icon = "✅" if predicted_class == expected else "❌"
        class_name = "Clickbait" if predicted_class == 1 else "Không clickbait"
        
        print(f"   {i}. {result_icon} '{original_text[:60]}...'")
        print(f"      Predicted: {class_name} (confidence: {confidence:.3f})")
        if config.get("preprocessing") == "word_segmentation":
            print(f"      Segmented: {processed_text[:60]}...")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Vietnamese BERT models for clickbait classification")
    parser.add_argument("--model", required=True, choices=list(VIETNAMESE_BERT_CONFIGS.keys()) + ["all"], 
                      help="Vietnamese model to train")
    parser.add_argument("--hardware", default="rtx_4090", choices=["rtx_4090", "rtx_3080", "rtx_3060", "cpu_only"],
                      help="Hardware configuration")
    parser.add_argument("--data_dir", default="data_vietnamese", help="Vietnamese data directory")
    parser.add_argument("--output_dir", default="outputs_vietnamese", help="Output directory")
    
    args = parser.parse_args()
    
    print("🇻🇳 VIETNAMESE CLICKBAIT CLASSIFICATION - BERT TRAINING")
    print("=" * 70)
    
    if args.model == "all":
        # Train all Vietnamese BERT models
        models_to_train = list(VIETNAMESE_BERT_CONFIGS.keys())
        results = {}
        
        for model_key in models_to_train:
            try:
                result = train_vietnamese_model(model_key, args.hardware, args.output_dir, args.data_dir)
                results[model_key] = result
            except Exception as e:
                print(f"❌ Failed to train {model_key}: {e}")
                results[model_key] = {"error": str(e)}
        
        # Save combined results
        combined_results_file = f"{args.output_dir}/all_vietnamese_bert_results.json"
        with open(combined_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 ALL VIETNAMESE BERT MODELS TRAINING COMPLETED!")
        print(f"📊 Combined results saved to: {combined_results_file}")
        
    else:
        # Train single model
        result = train_vietnamese_model(args.model, args.hardware, args.output_dir, args.data_dir)
        print(f"🎉 Vietnamese model {args.model} training completed!")

if __name__ == "__main__":
    main() 