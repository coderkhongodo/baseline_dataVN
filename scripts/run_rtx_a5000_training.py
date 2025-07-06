#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master script cho huấn luyện Vietnamese Clickbait Classification trên 2x RTX A5000
Pipeline hoàn chỉnh: Setup -> Prepare Data -> Train -> Evaluate
"""

import os
import sys
import argparse
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rtx_a5000_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import local configs để hiển thị thông tin
sys.path.append(str(Path(__file__).parent.parent))
from configs.rtx_a5000_configs import (
    HARDWARE_SPECS, TRAINING_STRATEGIES, 
    MEMORY_OPTIMIZATION_CONFIGS, print_hardware_recommendations
)

class RTXTrainingPipeline:
    """Pipeline training cho RTX A5000"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "models/vietnamese_clickbait"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🚀 RTX A5000 Training Pipeline initialized")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def check_system_requirements(self) -> bool:
        """
        Kiểm tra system requirements
        """
        logger.info("🔍 Checking system requirements...")
        
        issues = []
        
        # Check CUDA
        try:
            import torch
            if not torch.cuda.is_available():
                issues.append("CUDA not available")
            else:
                gpu_count = torch.cuda.device_count()
                logger.info(f"✅ Found {gpu_count} GPU(s)")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                
                if gpu_count < 2:
                    logger.warning(f"⚠️  Expected 2 GPUs for RTX A5000 setup, found {gpu_count}")
        except ImportError:
            issues.append("PyTorch not installed")
        
        # Check required packages
        required_packages = [
            'transformers', 'datasets', 'peft', 'accelerate', 
            'scikit-learn', 'numpy', 'pandas'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} available")
            except ImportError:
                issues.append(f"Missing package: {package}")
        
        if issues:
            logger.error("❌ System requirements not met:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("✅ All system requirements met!")
        return True
    
    def prepare_data(self, force: bool = False) -> bool:
        """
        Chuẩn bị dữ liệu training
        """
        logger.info("📊 Preparing training data...")
        
        # Check if training data already exists
        train_data = self.data_dir / "train_dtVN" / "training_data.jsonl"
        val_data = self.data_dir / "val_dtVN" / "training_data.jsonl"
        test_data = self.data_dir / "test_dtVN" / "training_data.jsonl"
        
        if all(f.exists() for f in [train_data, val_data, test_data]) and not force:
            logger.info("✅ Training data already exists")
            return True
        
        # Run data preparation script
        try:
            cmd = [sys.executable, "scripts/prepare_training_data.py", "--data-dir", str(self.data_dir)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Data preparation completed")
            logger.info(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Data preparation failed: {e}")
            logger.error(e.stderr)
            return False
    
    def run_training(
        self, 
        model_type: str, 
        strategy: str = "balanced",
        test_samples: bool = True
    ) -> Optional[str]:
        """
        Chạy training với model type và strategy chỉ định
        """
        logger.info(f"🎯 Starting training: {model_type} with {strategy} strategy")
        
        # Check if this is a multi-GPU setup
        env = os.environ.copy()
        
        # Setup training command
        cmd = [
            sys.executable, "scripts/train_vietnamese_rtx_a5000.py",
            "--model-type", model_type,
            "--strategy", strategy,
            "--train-data", str(self.data_dir / "train_dtVN" / "training_data.jsonl"),
            "--val-data", str(self.data_dir / "val_dtVN" / "training_data.jsonl"),
            "--output-dir", str(self.output_dir)
        ]
        
        if test_samples:
            cmd.append("--test-samples")
        
        # For multi-GPU, use torchrun
        import torch
        if torch.cuda.device_count() > 1:
            logger.info("🖥️  Using multi-GPU training with torchrun")
            cmd = [
                "torchrun", 
                "--nproc_per_node", str(torch.cuda.device_count()),
                "--nnodes", "1", 
                "--node_rank", "0",
                "--master_addr", "localhost",
                "--master_port", "12355"
            ] + cmd[1:]  # Skip python executable
        
        try:
            # Run training
            logger.info(f"🚀 Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, env=env, text=True)
            
            logger.info("✅ Training completed successfully!")
            
            # Try to extract model directory from logs
            # This is a simple heuristic - in production you'd want better tracking
            return f"{self.output_dir}/{model_type}_{self.timestamp}_{strategy}"
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Training failed: {e}")
            return None
    
    def run_evaluation(self, model_dir: str) -> Dict:
        """
        Chạy evaluation chi tiết trên test set
        """
        logger.info(f"📊 Running evaluation on: {model_dir}")
        
        try:
            cmd = [
                sys.executable, "scripts/evaluate_vietnamese_models.py",
                "--model-dir", model_dir,
                "--test-data", str(self.data_dir / "test_dtVN" / "training_data.jsonl"),
                "--output-dir", f"{model_dir}/evaluation"
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Evaluation completed")
            
            # Try to parse results
            eval_results = {}
            try:
                eval_file = Path(model_dir) / "evaluation" / "detailed_results.json"
                if eval_file.exists():
                    with open(eval_file, 'r') as f:
                        eval_results = json.load(f)
            except:
                pass
            
            return eval_results
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Evaluation failed: {e}")
            logger.error(e.stderr)
            return {}
    
    def run_complete_pipeline(
        self, 
        model_types: List[str],
        strategy: str = "balanced",
        prepare_data_force: bool = False
    ) -> Dict[str, Dict]:
        """
        Chạy pipeline hoàn chỉnh cho nhiều models
        """
        logger.info("🚀 STARTING COMPLETE RTX A5000 TRAINING PIPELINE")
        logger.info("=" * 80)
        
        results = {}
        
        # 1. System check
        if not self.check_system_requirements():
            logger.error("❌ System requirements not met. Exiting.")
            return results
        
        # 2. Data preparation
        if not self.prepare_data(force=prepare_data_force):
            logger.error("❌ Data preparation failed. Exiting.")
            return results
        
        # 3. Train each model
        for model_type in model_types:
            logger.info(f"\n🎯 TRAINING MODEL: {model_type.upper()}")
            logger.info("-" * 60)
            
            try:
                # Show strategy info
                if model_type.replace('-', '_') in TRAINING_STRATEGIES:
                    strategy_info = TRAINING_STRATEGIES[model_type.replace('-', '_')]
                    logger.info(f"📋 Strategy: {strategy_info['description']}")
                    logger.info(f"⏱️  Estimated time: {strategy_info['estimated_time']}")
                    logger.info(f"🎯 Expected accuracy: {strategy_info['expected_accuracy']}")
                
                # Run training
                model_dir = self.run_training(model_type, strategy)
                
                if model_dir:
                    # Run evaluation
                    eval_results = self.run_evaluation(model_dir)
                    
                    results[model_type] = {
                        'model_dir': model_dir,
                        'training_status': 'completed',
                        'evaluation_results': eval_results
                    }
                    
                    logger.info(f"✅ {model_type} completed successfully!")
                else:
                    results[model_type] = {
                        'training_status': 'failed',
                        'evaluation_results': {}
                    }
                    logger.error(f"❌ {model_type} training failed!")
                
            except Exception as e:
                logger.error(f"❌ Error training {model_type}: {e}")
                results[model_type] = {
                    'training_status': 'error',
                    'error': str(e),
                    'evaluation_results': {}
                }
        
        # 4. Summary
        self.print_final_summary(results)
        
        # 5. Save results
        summary_file = self.output_dir / f"training_summary_{self.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📋 Summary saved to: {summary_file}")
        
        return results
    
    def print_final_summary(self, results: Dict[str, Dict]):
        """
        In tổng kết cuối cùng
        """
        logger.info("\n" + "=" * 80)
        logger.info("🎉 TRAINING PIPELINE COMPLETED")
        logger.info("=" * 80)
        
        successful_models = []
        failed_models = []
        
        for model_type, result in results.items():
            if result.get('training_status') == 'completed':
                successful_models.append(model_type)
                eval_results = result.get('evaluation_results', {})
                accuracy = eval_results.get('accuracy', 'N/A')
                f1 = eval_results.get('f1', 'N/A')
                
                logger.info(f"✅ {model_type}: Accuracy={accuracy}, F1={f1}")
            else:
                failed_models.append(model_type)
                logger.info(f"❌ {model_type}: {result.get('training_status', 'unknown')}")
        
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"  Successful: {len(successful_models)}/{len(results)}")
        logger.info(f"  Failed: {len(failed_models)}/{len(results)}")
        
        if successful_models:
            logger.info(f"\n🏆 BEST PERFORMING MODELS:")
            # Sort by F1 score if available
            best_models = []
            for model_type in successful_models:
                eval_results = results[model_type].get('evaluation_results', {})
                f1 = eval_results.get('f1', 0)
                if isinstance(f1, (int, float)):
                    best_models.append((model_type, f1))
            
            best_models.sort(key=lambda x: x[1], reverse=True)
            for i, (model_type, f1) in enumerate(best_models[:3], 1):
                logger.info(f"  {i}. {model_type}: F1={f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description="RTX A5000 Vietnamese Clickbait Training Pipeline")
    parser.add_argument("--models", "-m", nargs="+", 
                       choices=["phobert-base", "phobert-large", "xlm-roberta-base",
                               "xlm-roberta-large", "vistral-7b", "vinallama-7b", "seallm-7b"],
                       default=["phobert-base"],
                       help="Models to train")
    parser.add_argument("--strategy", "-s", default="balanced",
                       choices=["fast", "balanced", "thorough"],
                       help="Training strategy")
    parser.add_argument("--data-dir", "-d", default="data",
                       help="Data directory")
    parser.add_argument("--output-dir", "-o", default="models/vietnamese_clickbait",
                       help="Output directory")
    parser.add_argument("--force-prepare-data", action="store_true",
                       help="Force re-preparation of training data")
    parser.add_argument("--show-hardware", action="store_true",
                       help="Show hardware recommendations and exit")
    parser.add_argument("--show-strategies", action="store_true",
                       help="Show available training strategies and exit")
    
    args = parser.parse_args()
    
    # Show info and exit if requested
    if args.show_hardware:
        print_hardware_recommendations()
        return
    
    if args.show_strategies:
        print("\n🚀 AVAILABLE TRAINING STRATEGIES:")
        print("=" * 60)
        for name, strategy in TRAINING_STRATEGIES.items():
            config = strategy["config"]
            print(f"📋 {name.upper()}:")
            print(f"  Description: {strategy['description']}")
            print(f"  Model: {config.model_name}")
            print(f"  Time: {strategy['estimated_time']}")
            print(f"  Expected: {strategy['expected_accuracy']}")
            print("")
        return
    
    # Initialize pipeline
    pipeline = RTXTrainingPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Show initial info
    print_hardware_recommendations()
    
    print(f"\n📋 TRAINING PLAN:")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    
    # Confirm start
    if len(args.models) > 1:
        estimated_time = len(args.models) * 3  # Rough estimate
        print(f"\n⏱️  Estimated total time: ~{estimated_time} hours")
        
        confirm = input("\n🚀 Start training pipeline? [y/N]: ")
        if confirm.lower() != 'y':
            print("❌ Training cancelled by user")
            return
    
    # Run pipeline
    try:
        results = pipeline.run_complete_pipeline(
            model_types=args.models,
            strategy=args.strategy,
            prepare_data_force=args.force_prepare_data
        )
        
        print("\n🎉 Pipeline completed! Check logs for details.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 