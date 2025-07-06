#!/usr/bin/env python3
"""
Evaluate Vietnamese Clickbait Classification Models
Đánh giá các mô hình phân loại clickbait tiếng Việt
"""

import os
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import underthesea
from datasets import load_dataset

# Add configs to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.vietnamese_models import VIETNAMESE_BERT_CONFIGS

def load_trained_vietnamese_model(model_path):
    """Load trained Vietnamese model"""
    print(f"📂 Loading trained Vietnamese model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        print(f"✅ Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def preprocess_vietnamese_text(text, model_name, preprocessing_method="word_segmentation"):
    """Preprocess Vietnamese text for inference"""
    
    if "phobert" in model_name.lower() and preprocessing_method == "word_segmentation":
        try:
            segmented = underthesea.word_tokenize(text, format="text")
            return segmented
        except Exception as e:
            print(f"⚠️ Word segmentation failed: {e}")
            return text
    else:
        return text

def predict_vietnamese_clickbait(model, tokenizer, text, model_name, max_length=256):
    """Predict clickbait for Vietnamese text"""
    
    # Preprocess text
    processed_text = preprocess_vietnamese_text(text, model_name)
    
    # Tokenize
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    )
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def evaluate_vietnamese_model_on_dataset(model_path, data_dir="data_vietnamese", test_file="test/data.jsonl"):
    """Evaluate Vietnamese model on test dataset"""
    
    print(f"\n{'='*70}")
    print(f"🇻🇳 EVALUATING VIETNAMESE MODEL: {Path(model_path).name}")
    print(f"{'='*70}")
    
    # Load model
    model, tokenizer = load_trained_vietnamese_model(model_path)
    if not model:
        return None
    
    # Load test data
    test_path = Path(data_dir) / test_file
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return None
    
    # Load test dataset
    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    print(f"📊 Loaded {len(test_data)} test samples")
    
    # Make predictions
    predictions = []
    true_labels = []
    probabilities = []
    confidence_scores = []
    
    model_name = Path(model_path).name
    
    print("🔮 Making predictions...")
    for i, item in enumerate(test_data):
        if i % 100 == 0:
            print(f"   Processed {i}/{len(test_data)} samples")
        
        text = item['text']
        true_label = item['label']
        
        pred_class, confidence, probs = predict_vietnamese_clickbait(
            model, tokenizer, text, model_name
        )
        
        predictions.append(pred_class)
        true_labels.append(true_label)
        probabilities.append(probs)
        confidence_scores.append(confidence)
    
    # Calculate metrics
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    probabilities = np.array(probabilities)
    
    # Basic metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_binary = f1_score(true_labels, predictions, average='binary')
    
    # Per-class metrics
    f1_clickbait = f1_score(true_labels, predictions, pos_label=1)
    f1_no_clickbait = f1_score(true_labels, predictions, pos_label=0)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
    except:
        roc_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Classification report
    class_names = ["Không clickbait", "Clickbait"]
    class_report = classification_report(
        true_labels, predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Results
    results = {
        "model_path": str(model_path),
        "model_name": model_name,
        "test_samples": len(test_data),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_weighted": float(f1),
            "f1_macro": float(f1_macro),
            "f1_binary": float(f1_binary),
            "f1_clickbait": float(f1_clickbait),
            "f1_no_clickbait": float(f1_no_clickbait),
            "roc_auc": float(roc_auc)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "mean_confidence": float(np.mean(confidence_scores)),
        "evaluation_timestamp": datetime.now().isoformat()
    }
    
    # Print results
    print(f"\n📊 VIETNAMESE MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Test samples: {len(test_data)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"F1 Clickbait: {f1_clickbait:.4f}")
    print(f"F1 No-clickbait: {f1_no_clickbait:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Mean confidence: {np.mean(confidence_scores):.4f}")
    
    # Show confusion matrix
    print(f"\n📈 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 No-CB  CB")
    print(f"Actual No-CB:    {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"Actual CB:       {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    return results

def analyze_vietnamese_prediction_patterns(results, output_dir="evaluation_output"):
    """Analyze patterns in Vietnamese clickbait predictions"""
    
    print(f"\n🔬 ANALYZING VIETNAMESE PREDICTION PATTERNS")
    print("=" * 50)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create visualizations
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Vietnamese Clickbait Classification Analysis', fontsize=16, fontweight='bold')
    
    # 1. Model performance comparison
    ax1 = axes[0, 0]
    if isinstance(results, list) and len(results) > 1:
        models = [r['model_name'] for r in results]
        f1_scores = [r['metrics']['f1_weighted'] for r in results]
        accuracy_scores = [r['metrics']['accuracy'] for r in results]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        bars2 = ax1.bar(x + width/2, accuracy_scores, width, label='Accuracy', alpha=0.8)
        
        ax1.set_xlabel('Vietnamese Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m[:15] + '...' if len(m) > 15 else m for m in models], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'Single Model\nEvaluation', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Model Performance')
    
    # 2. Confusion Matrix
    ax2 = axes[0, 1]
    result = results[0] if isinstance(results, list) else results
    cm = np.array(result['confusion_matrix'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['No-clickbait', 'Clickbait'],
                yticklabels=['No-clickbait', 'Clickbait'])
    ax2.set_title('Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # 3. Per-class F1 scores
    ax3 = axes[1, 0]
    if isinstance(results, list):
        model_names = [r['model_name'][:10] + '...' if len(r['model_name']) > 10 else r['model_name'] for r in results]
        clickbait_f1 = [r['metrics']['f1_clickbait'] for r in results]
        no_clickbait_f1 = [r['metrics']['f1_no_clickbait'] for r in results]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax3.bar(x - width/2, clickbait_f1, width, label='Clickbait F1', alpha=0.8, color='red')
        ax3.bar(x + width/2, no_clickbait_f1, width, label='No-clickbait F1', alpha=0.8, color='blue')
        
        ax3.set_xlabel('Vietnamese Models')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Per-class F1 Scores')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        single_result = results
        classes = ['Clickbait', 'No-clickbait']
        f1_scores = [single_result['metrics']['f1_clickbait'], single_result['metrics']['f1_no_clickbait']]
        colors = ['red', 'blue']
        
        bars = ax3.bar(classes, f1_scores, color=colors, alpha=0.8)
        ax3.set_title('Per-class F1 Scores')
        ax3.set_ylabel('F1 Score')
        ax3.grid(True, alpha=0.3)
        
        for bar, score in zip(bars, f1_scores):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
    
    # 4. Metrics radar chart or distribution
    ax4 = axes[1, 1]
    if isinstance(results, list) and len(results) > 1:
        # Show metric distribution across models
        metrics_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC AUC']
        all_metrics = []
        
        for result in results:
            model_metrics = [
                result['metrics']['accuracy'],
                result['metrics']['f1_weighted'],
                result['metrics']['precision'],
                result['metrics']['recall'],
                result['metrics']['roc_auc']
            ]
            all_metrics.append(model_metrics)
        
        all_metrics = np.array(all_metrics)
        
        # Box plot of metrics
        ax4.boxplot(all_metrics, labels=[m[:8] for m in metrics_names])
        ax4.set_title('Metrics Distribution')
        ax4.set_ylabel('Score')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
    else:
        # Single model metrics
        single_result = results if not isinstance(results, list) else results[0]
        metrics_names = ['Accuracy', 'F1', 'Precision', 'Recall', 'ROC AUC']
        metrics_values = [
            single_result['metrics']['accuracy'],
            single_result['metrics']['f1_weighted'],
            single_result['metrics']['precision'],
            single_result['metrics']['recall'],
            single_result['metrics']['roc_auc']
        ]
        
        bars = ax4.bar(metrics_names, metrics_values, alpha=0.8, color='green')
        ax4.set_title('Model Metrics')
        ax4.set_ylabel('Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, metrics_values):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = Path(output_dir) / "vietnamese_evaluation_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {plot_file}")
    
    plt.show()

def test_vietnamese_samples(model_path):
    """Test model on Vietnamese sample headlines"""
    
    print(f"\n🧪 TESTING ON VIETNAMESE SAMPLE HEADLINES")
    print("=" * 60)
    
    model, tokenizer = load_trained_vietnamese_model(model_path)
    if not model:
        return
    
    vietnamese_test_samples = [
        {"text": "Chính phủ thông qua nghị định mới về thuế môi trường từ 1/2024", "expected": 0},
        {"text": "SỐCKKKK! Bạn sẽ không tin được điều mà cô gái này vừa làm!!!", "expected": 1},
        {"text": "BIDV tăng lãi suất tiết kiệm lên 7.5% từ tuần tới", "expected": 0},
        {"text": "7 bí mật làm giàu mà các tỷ phú không muốn bạn biết - Số 3 gây sốc!", "expected": 1},
        {"text": "Đội tuyển Việt Nam thắng 2-1 trước Thái Lan tại vòng loại World Cup", "expected": 0},
        {"text": "Cách kiếm 50 triệu/tháng mà 99% người Việt chưa biết - Bạn có dám thử?", "expected": 1},
        {"text": "Giá vàng tăng mạnh 2% trong phiên giao dịch hôm nay", "expected": 0},
        {"text": "Điều này sẽ thay đổi cuộc sống bạn mãi mãi - Đọc ngay kẻo hối hận!", "expected": 1},
        {"text": "Thống đốc NHNN họp báo về chính sách tiền tệ quý IV/2024", "expected": 0},
        {"text": "Phát hiện chấn động: Loại quả này ăn hàng ngày có thể...", "expected": 1}
    ]
    
    model_name = Path(model_path).name
    correct_predictions = 0
    
    for i, sample in enumerate(vietnamese_test_samples, 1):
        text = sample["text"]
        expected = sample["expected"]
        
        pred_class, confidence, probs = predict_vietnamese_clickbait(
            model, tokenizer, text, model_name
        )
        
        # Results
        pred_text = "Clickbait" if pred_class == 1 else "Không clickbait"
        expected_text = "Clickbait" if expected == 1 else "Không clickbait"
        correct = pred_class == expected
        if correct:
            correct_predictions += 1
        
        icon = "✅" if correct else "❌"
        print(f"{i:2d}. {icon} {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"     Dự đoán: {pred_text} (tin cậy: {confidence:.3f})")
        print(f"     Thực tế:  {expected_text}")
        print()
    
    accuracy = correct_predictions / len(vietnamese_test_samples)
    print(f"🎯 Sample Test Accuracy: {accuracy:.3f} ({correct_predictions}/{len(vietnamese_test_samples)})")

def compare_vietnamese_models(models_dir="outputs_vietnamese", output_dir="evaluation_output"):
    """Compare multiple Vietnamese models"""
    
    print(f"\n🇻🇳 COMPARING VIETNAMESE MODELS")
    print("=" * 70)
    
    models_dir = Path(models_dir)
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    # Find all trained models
    model_paths = []
    for model_folder in models_dir.iterdir():
        if model_folder.is_dir() and (model_folder / "config.json").exists():
            model_paths.append(model_folder)
    
    if not model_paths:
        print(f"❌ No trained models found in {models_dir}")
        return
    
    print(f"📂 Found {len(model_paths)} Vietnamese models:")
    for path in model_paths:
        print(f"   - {path.name}")
    
    # Evaluate each model
    all_results = []
    for model_path in model_paths:
        result = evaluate_vietnamese_model_on_dataset(model_path)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("❌ No models could be evaluated")
        return
    
    # Save comparison results
    Path(output_dir).mkdir(exist_ok=True)
    comparison_file = Path(output_dir) / "vietnamese_models_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Comparison results saved to: {comparison_file}")
    
    # Create comparison table
    print(f"\n📊 VIETNAMESE MODELS COMPARISON TABLE")
    print("=" * 90)
    
    df_data = []
    for result in all_results:
        df_data.append({
            'Model': result['model_name'][:20],
            'Accuracy': f"{result['metrics']['accuracy']:.3f}",
            'F1': f"{result['metrics']['f1_weighted']:.3f}",
            'F1-CB': f"{result['metrics']['f1_clickbait']:.3f}",
            'F1-NCB': f"{result['metrics']['f1_no_clickbait']:.3f}",
            'ROC AUC': f"{result['metrics']['roc_auc']:.3f}",
            'Samples': result['test_samples']
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    # Find best model
    best_f1_idx = max(range(len(all_results)), key=lambda i: all_results[i]['metrics']['f1_weighted'])
    best_model = all_results[best_f1_idx]
    
    print(f"\n🏆 BEST VIETNAMESE MODEL:")
    print(f"   Model: {best_model['model_name']}")
    print(f"   F1-score: {best_model['metrics']['f1_weighted']:.4f}")
    print(f"   Accuracy: {best_model['metrics']['accuracy']:.4f}")
    
    # Create visualizations
    analyze_vietnamese_prediction_patterns(all_results, output_dir)
    
    return all_results

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese clickbait classification models")
    parser.add_argument("--model_path", help="Path to specific model to evaluate")
    parser.add_argument("--models_dir", default="outputs_vietnamese", help="Directory containing multiple models")
    parser.add_argument("--data_dir", default="data_vietnamese", help="Vietnamese data directory")
    parser.add_argument("--output_dir", default="evaluation_output", help="Output directory for results")
    parser.add_argument("--compare_all", action="store_true", help="Compare all models in directory")
    parser.add_argument("--test_samples", action="store_true", help="Test on sample headlines")
    
    args = parser.parse_args()
    
    print("🇻🇳 VIETNAMESE CLICKBAIT CLASSIFICATION - MODEL EVALUATION")
    print("=" * 70)
    
    if args.compare_all:
        # Compare all models
        compare_vietnamese_models(args.models_dir, args.output_dir)
        
    elif args.model_path:
        # Evaluate specific model
        if not Path(args.model_path).exists():
            print(f"❌ Model path not found: {args.model_path}")
            return
        
        result = evaluate_vietnamese_model_on_dataset(args.model_path, args.data_dir)
        if result:
            # Save single model result
            Path(args.output_dir).mkdir(exist_ok=True)
            result_file = Path(args.output_dir) / f"{Path(args.model_path).name}_evaluation.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"💾 Results saved to: {result_file}")
            
            # Analyze single model
            analyze_vietnamese_prediction_patterns(result, args.output_dir)
        
        if args.test_samples:
            test_vietnamese_samples(args.model_path)
    else:
        print("❌ Please specify --model_path or --compare_all")
        print("Examples:")
        print("  python evaluate_vietnamese_models.py --model_path outputs_vietnamese/phobert-base-vietnamese")
        print("  python evaluate_vietnamese_models.py --compare_all")

if __name__ == "__main__":
    main() 