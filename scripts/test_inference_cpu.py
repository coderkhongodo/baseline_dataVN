#!/usr/bin/env python3
"""
Quick Inference Test for Vietnamese Clickbait Classification
Test nhanh inference trên CPU với pre-trained model
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_sample_texts():
    """Test với một số câu mẫu"""
    sample_texts = [
        # Clickbait examples
        "Bạn sẽ không tin nổi điều gì xảy ra tiếp theo!",
        "10 bí mật mà chỉ có 1% dân số biết",
        "Cách kiếm tiền online mà 99% người không biết",
        "Sự thật về căn bệnh này sẽ khiến bạn sốc",
        
        # Non-clickbait examples  
        "Thủ tướng chủ trì họp Chính phủ thường kỳ",
        "Giá xăng tăng 500 đồng từ 15h chiều nay",
        "Đại học Quốc gia Hà Nội tuyển sinh 2024",
        "Dự báo thời tiết: Bắc Bộ có mưa rào và dông"
    ]
    
    expected_labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1=clickbait, 0=non-clickbait
    
    return sample_texts, expected_labels

def load_test_data_subset(file_path, n_samples=20):
    """Load subset of real test data"""
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy file: {file_path}")
        return [], []
    
    print(f"Loading test data từ {file_path}...")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            data.append(json.loads(line.strip()))
    
    texts = [item['title'] for item in data]
    # Convert string labels to integers: "clickbait" -> 1, "non-clickbait" -> 0
    labels = [1 if item['label'] == 'clickbait' else 0 for item in data]
    
    return texts, labels

def test_model_inference(model_name, texts, true_labels=None):
    """Test model inference"""
    print(f"\n🤖 Testing model: {model_name}")
    
    try:
        # Load model
        print("📥 Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model parameters: {model.num_parameters():,}")
        print(f"⚠️ Note: Using pre-trained model (chưa fine-tune), predictions có thể random")
        
        # Manual inference (không dùng pipeline vì model chưa fine-tune)
        print(f"\n🔍 Running manual inference on {len(texts)} texts...")
        start_time = datetime.now()
        
        predictions = []
        confidences = []
        
        for i, text in enumerate(texts):
            # Tokenize
            inputs = tokenizer(
                text[:128], 
                truncation=True, 
                padding=True, 
                return_tensors="pt"
            )
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Get prediction and confidence
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = torch.max(probs, dim=-1).values.item()
            
            predictions.append(pred_idx)
            confidences.append(confidence)
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(texts)} texts...")
        
        end_time = datetime.now()
        inference_time = end_time - start_time
        
        print(f"⏰ Inference completed in: {inference_time}")
        print(f"🚀 Speed: {len(texts) / inference_time.total_seconds():.2f} texts/second")
        
        # Calculate accuracy if true labels provided
        accuracy = None
        if true_labels:
            correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
            accuracy = correct / len(true_labels)
            print(f"🎯 Accuracy: {accuracy:.4f} ({correct}/{len(true_labels)})")
            print(f"⚠️ Note: Pre-trained model chưa fine-tune cho clickbait, accuracy có thể thấp")
        
        # Show predictions
        print(f"\n📋 Sample Predictions:")
        for i in range(min(10, len(texts))):
            text = texts[i][:60] + "..." if len(texts[i]) > 60 else texts[i]
            pred_label = "Clickbait" if predictions[i] == 1 else "Non-Clickbait"
            conf = confidences[i]
            
            status = ""
            if true_labels:
                true_label = "Clickbait" if true_labels[i] == 1 else "Non-Clickbait"
                status = "✅" if predictions[i] == true_labels[i] else "❌"
                print(f"   {status} Text: {text}")
                print(f"      True: {true_label}, Pred: {pred_label} ({conf:.3f})")
            else:
                print(f"   Text: {text}")
                print(f"   Pred: {pred_label} ({conf:.3f})")
        
        return predictions, confidences, accuracy, inference_time
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def main():
    print("=" * 60)
    print("🚀 VIETNAMESE CLICKBAIT INFERENCE - CPU TEST")
    print("=" * 60)
    
    # System info
    print(f"🔧 Python version: {sys.version}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    print(f"🔧 Device: CPU")
    print(f"🔧 CPU threads: {torch.get_num_threads()}")
    
    # Models to test
    models_to_test = [
        "vinai/phobert-base",
        # Add more models if needed
    ]
    
    print(f"\n📋 Testing {len(models_to_test)} model(s)")
    
    # Test 1: Sample texts
    print(f"\n" + "="*50)
    print(f"🧪 TEST 1: Sample Vietnamese Texts")
    print(f"="*50)
    
    sample_texts, expected_labels = test_sample_texts()
    
    for model_name in models_to_test:
        test_model_inference(model_name, sample_texts, expected_labels)
    
    # Test 2: Real data (if available)
    print(f"\n" + "="*50)
    print(f"🧪 TEST 2: Real Test Data Subset")
    print(f"="*50)
    
    test_file = os.path.join("data", "test_dtVN", "training_data.jsonl")
    test_texts, test_labels = load_test_data_subset(test_file, n_samples=20)
    
    if test_texts:
        for model_name in models_to_test:
            test_model_inference(model_name, test_texts, test_labels)
    else:
        print("⚠️ Không có data thật để test. Chạy scripts/prepare_training_data.py trước.")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"✅ CPU INFERENCE TEST COMPLETED")
    print(f"="*60)
    print(f"💻 Device: CPU")
    print(f"🔧 Models tested: {len(models_to_test)}")
    print(f"📊 Sample texts: {len(sample_texts)}")
    if test_texts:
        print(f"📊 Real test texts: {len(test_texts)}")
    
    print(f"\n💡 Để chạy training test: python scripts/test_cpu_quick.py")

if __name__ == "__main__":
    main() 