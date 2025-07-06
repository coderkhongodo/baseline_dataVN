#!/usr/bin/env python3
"""
Master Pipeline for Vietnamese Clickbait Classification
Chạy toàn bộ pipeline từ preprocessing đến evaluation cho tiếng Việt
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

def run_command(command, description, show_output=True):
    """Execute command with error handling"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=not show_output,
            text=True,
            check=True
        )
        
        if not show_output:
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        
        print(f"✅ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_vietnamese_data_exists(data_path):
    """Check if Vietnamese dataset exists"""
    if not os.path.exists(data_path):
        print(f"❌ Vietnamese dataset not found: {data_path}")
        print("Please provide your Vietnamese dataset using --input_data argument")
        return False
    return True

def vietnamese_environment_setup():
    """Setup Vietnamese environment"""
    print("🇻🇳 STEP 1: VIETNAMESE ENVIRONMENT SETUP")
    print("=" * 70)
    
    success = run_command(
        "python scripts/setup_vietnamese_environment.py",
        "Setting up Vietnamese environment",
        show_output=True
    )
    
    return success

def vietnamese_data_preprocessing(input_data_path, segment_method="underthesea"):
    """Preprocess Vietnamese data"""
    print("\n🇻🇳 STEP 2: VIETNAMESE DATA PREPROCESSING")
    print("=" * 70)
    
    if not check_vietnamese_data_exists(input_data_path):
        return False
    
    command = f"""python scripts/preprocess_vietnamese_data.py \
        --input "{input_data_path}" \
        --output data_vietnamese \
        --segment {segment_method} \
        --test_size 0.2 \
        --val_size 0.1"""
    
    success = run_command(
        command,
        "Preprocessing Vietnamese dataset",
        show_output=True
    )
    
    return success

def train_vietnamese_bert_models(models_to_train, hardware="rtx_4090"):
    """Train Vietnamese BERT models"""
    print("\n🇻🇳 STEP 3: TRAINING VIETNAMESE BERT MODELS")
    print("=" * 70)
    
    success = True
    for model in models_to_train:
        command = f"""python scripts/train_vietnamese_bert.py \
            --model {model} \
            --hardware {hardware} \
            --data_dir data_vietnamese \
            --output_dir outputs_vietnamese"""
        
        model_success = run_command(
            command,
            f"Training Vietnamese BERT model: {model}",
            show_output=True
        )
        
        success = success and model_success
    
    return success

def run_vietnamese_prompting_evaluation():
    """Run Vietnamese prompting evaluation"""
    print("\n🇻🇳 STEP 4: VIETNAMESE PROMPTING EVALUATION")
    print("=" * 70)
    
    # Check if .env file exists for LLM API
    if not os.path.exists('.env'):
        print("⚠️ .env file not found. Creating template...")
        env_template = """# LLM API Configuration for Vietnamese Prompting
SHUBI_API_KEY=your_api_key_here
SHUBI_URL=your_base_url_here

# Example for OpenAI-compatible APIs:
# OPENAI_API_KEY=your_openai_key
# OPENAI_BASE_URL=https://api.openai.com/v1
"""
        with open('.env', 'w') as f:
            f.write(env_template)
        
        print("📝 Please update .env file with your LLM API credentials")
        print("Skipping prompting evaluation for now...")
        return True
    
    success = run_command(
        "python scripts/vietnamese_prompting.py",
        "Running Vietnamese prompting evaluation",
        show_output=True
    )
    
    return success

def evaluate_vietnamese_models():
    """Evaluate all Vietnamese models"""
    print("\n🇻🇳 STEP 5: EVALUATING VIETNAMESE MODELS")
    print("=" * 70)
    
    success = run_command(
        "python scripts/evaluate_vietnamese_models.py --compare_all --test_samples",
        "Evaluating Vietnamese models",
        show_output=True
    )
    
    return success

def generate_vietnamese_report(pipeline_start_time):
    """Generate final Vietnamese pipeline report"""
    print("\n🇻🇳 GENERATING FINAL VIETNAMESE REPORT")
    print("=" * 70)
    
    total_time = time.time() - pipeline_start_time
    
    report = {
        "pipeline_type": "Vietnamese Clickbait Classification",
        "completion_time": datetime.now().isoformat(),
        "total_runtime_seconds": total_time,
        "total_runtime_formatted": f"{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s",
        "steps_completed": [],
        "output_directories": [
            "data_vietnamese/",
            "outputs_vietnamese/", 
            "evaluation_output/",
            "logs_vietnamese/"
        ],
        "next_steps": [
            "Review evaluation results in evaluation_output/",
            "Test best model with new Vietnamese headlines",
            "Fine-tune hyperparameters based on results",
            "Deploy best model for production use"
        ]
    }
    
    # Check which steps were completed
    steps_status = {
        "environment_setup": os.path.exists("data_vietnamese/"),
        "data_preprocessing": os.path.exists("data_vietnamese/train/data.jsonl"),
        "model_training": os.path.exists("outputs_vietnamese/"),
        "prompting_evaluation": os.path.exists("vietnamese_prompting_results.json"),
        "model_evaluation": os.path.exists("evaluation_output/")
    }
    
    for step, completed in steps_status.items():
        report["steps_completed"].append({
            "step": step,
            "completed": completed,
            "status": "✅ Completed" if completed else "❌ Not completed"
        })
    
    # Find best model if evaluation was completed
    comparison_file = Path("evaluation_output/vietnamese_models_comparison.json")
    if comparison_file.exists():
        try:
            with open(comparison_file, 'r', encoding='utf-8') as f:
                evaluation_results = json.load(f)
            
            if evaluation_results:
                best_model = max(evaluation_results, key=lambda x: x['metrics']['f1_weighted'])
                report["best_model"] = {
                    "name": best_model['model_name'],
                    "f1_score": best_model['metrics']['f1_weighted'],
                    "accuracy": best_model['metrics']['accuracy'],
                    "path": best_model['model_path']
                }
        except Exception as e:
            print(f"⚠️ Could not load evaluation results: {e}")
    
    # Save report
    report_file = "vietnamese_pipeline_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"📊 VIETNAMESE PIPELINE SUMMARY")
    print("=" * 50)
    print(f"⏱️  Total runtime: {report['total_runtime_formatted']}")
    print(f"📁 Output directories created:")
    for directory in report['output_directories']:
        exists = "✅" if os.path.exists(directory) else "❌"
        print(f"   {exists} {directory}")
    
    print(f"\n📋 Steps completed:")
    for step_info in report['steps_completed']:
        print(f"   {step_info['status']} {step_info['step']}")
    
    if 'best_model' in report:
        print(f"\n🏆 Best Vietnamese model:")
        print(f"   Model: {report['best_model']['name']}")
        print(f"   F1-score: {report['best_model']['f1_score']:.4f}")
        print(f"   Accuracy: {report['best_model']['accuracy']:.4f}")
    
    print(f"\n📝 Full report saved to: {report_file}")
    
    return True

def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(description="Vietnamese Clickbait Classification Pipeline")
    parser.add_argument("--input_data", required=True, help="Path to Vietnamese dataset file")
    parser.add_argument("--models", nargs="+", default=["phobert-base"], 
                       help="Models to train (phobert-base, xlm-roberta-base, etc.)")
    parser.add_argument("--hardware", default="rtx_4090", choices=["rtx_4090", "rtx_3080", "rtx_3060", "cpu_only"],
                       help="Hardware configuration")
    parser.add_argument("--segment_method", default="underthesea", choices=["underthesea", "pyvi", "none"],
                       help="Vietnamese text segmentation method")
    parser.add_argument("--skip_setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip data preprocessing")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training")
    parser.add_argument("--skip_prompting", action="store_true", help="Skip prompting evaluation")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip model evaluation")
    
    args = parser.parse_args()
    
    print("🇻🇳 VIETNAMESE CLICKBAIT CLASSIFICATION - FULL PIPELINE")
    print("=" * 70)
    print(f"📊 Input data: {args.input_data}")
    print(f"🤖 Models to train: {args.models}")
    print(f"🖥️  Hardware: {args.hardware}")
    print(f"✂️  Segmentation: {args.segment_method}")
    print("=" * 70)
    
    pipeline_start_time = time.time()
    overall_success = True
    
    # Step 1: Environment Setup
    if not args.skip_setup:
        success = vietnamese_environment_setup()
        overall_success = overall_success and success
        if not success:
            print("❌ Environment setup failed. Stopping pipeline.")
            return
    else:
        print("⏭️  Skipping environment setup")
    
    # Step 2: Data Preprocessing
    if not args.skip_preprocessing:
        success = vietnamese_data_preprocessing(args.input_data, args.segment_method)
        overall_success = overall_success and success
        if not success:
            print("❌ Data preprocessing failed. Stopping pipeline.")
            return
    else:
        print("⏭️  Skipping data preprocessing")
    
    # Step 3: Model Training
    if not args.skip_training:
        success = train_vietnamese_bert_models(args.models, args.hardware)
        overall_success = overall_success and success
        if not success:
            print("⚠️ Some models failed to train, but continuing...")
    else:
        print("⏭️  Skipping model training")
    
    # Step 4: Prompting Evaluation
    if not args.skip_prompting:
        success = run_vietnamese_prompting_evaluation()
        overall_success = overall_success and success
        if not success:
            print("⚠️ Prompting evaluation failed, but continuing...")
    else:
        print("⏭️  Skipping prompting evaluation")
    
    # Step 5: Model Evaluation
    if not args.skip_evaluation:
        success = evaluate_vietnamese_models()
        overall_success = overall_success and success
        if not success:
            print("⚠️ Model evaluation failed, but continuing...")
    else:
        print("⏭️  Skipping model evaluation")
    
    # Final Report
    generate_vietnamese_report(pipeline_start_time)
    
    # Final message
    print(f"\n{'='*70}")
    if overall_success:
        print("🎉 VIETNAMESE PIPELINE COMPLETED SUCCESSFULLY!")
        print("🇻🇳 Your Vietnamese clickbait classification system is ready!")
    else:
        print("⚠️ VIETNAMESE PIPELINE COMPLETED WITH SOME ISSUES")
        print("📝 Check the logs above for details")
    
    print(f"\n📁 Check these directories for outputs:")
    print(f"   📊 data_vietnamese/ - Processed Vietnamese dataset")
    print(f"   🤖 outputs_vietnamese/ - Trained Vietnamese models")
    print(f"   📈 evaluation_output/ - Evaluation results and visualizations")
    print(f"   📋 vietnamese_pipeline_report.json - Full pipeline report")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Review best model in evaluation results")
    print(f"   2. Test with your own Vietnamese headlines")
    print(f"   3. Deploy the best model for production use")
    print(f"   4. Collect more Vietnamese data for further improvement")

if __name__ == "__main__":
    main() 