#!/usr/bin/env python3
"""
Compare Multiple Models - NO UNDERTHESEA
So sánh performance của nhiều models Vietnamese clickbait classification
"""

import subprocess
import json
import time
from datetime import datetime
import pandas as pd

def run_training(model_name, use_subset=True, epochs=2):
    """Run training for a specific model"""
    print(f"\n🚀 Training {model_name}...")
    
    # Create temporary config
    config_code = f'''
import sys
sys.path.append('scripts')
from run_simple_training import main, CONFIG

CONFIG['model_name'] = '{model_name}'
CONFIG['use_subset'] = {use_subset}
CONFIG['subset_size'] = 500
CONFIG['epochs'] = {epochs}
CONFIG['save_model'] = True

if __name__ == "__main__":
    main()
'''
    
    # Write temp script
    with open('temp_train.py', 'w') as f:
        f.write(config_code)
    
    try:
        # Run training
        result = subprocess.run(['python', 'temp_train.py'], 
                              capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            # Extract accuracy from output
            lines = result.stdout.split('\n')
            accuracy = None
            f1 = None
            training_time = None
            
            for line in lines:
                if 'Accuracy:' in line:
                    accuracy = float(line.split('Accuracy:')[1].strip())
                elif 'F1-Score:' in line:
                    f1 = float(line.split('F1-Score:')[1].strip())
                elif 'Time:' in line and 'Training' not in line:
                    training_time = line.split('Time:')[1].strip()
            
            return {
                'model': model_name,
                'accuracy': accuracy,
                'f1': f1,
                'training_time': training_time,
                'status': 'success',
                'error': None
            }
        else:
            return {
                'model': model_name,
                'accuracy': None,
                'f1': None,
                'training_time': None,
                'status': 'failed',
                'error': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {
            'model': model_name,
            'accuracy': None,
            'f1': None,
            'training_time': None,
            'status': 'timeout',
            'error': 'Training timeout (30 min)'
        }
    except Exception as e:
        return {
            'model': model_name,
            'accuracy': None,
            'f1': None,
            'training_time': None,
            'status': 'error',
            'error': str(e)
        }
    finally:
        # Cleanup
        import os
        if os.path.exists('temp_train.py'):
            os.remove('temp_train.py')

def main():
    print("🔄 VIETNAMESE CLICKBAIT MODEL COMPARISON (NO UNDERTHESEA)")
    print("=" * 70)
    
    # Models to compare
    models = [
        'vinai/phobert-base',
        'xlm-roberta-base',
        # 'vinai/phobert-large',  # Uncomment nếu muốn test
        # 'xlm-roberta-large',    # Uncomment nếu muốn test
    ]
    
    print(f"📋 Models to compare: {len(models)}")
    for i, model in enumerate(models):
        print(f"   {i+1}. {model}")
    
    print(f"\n⚙️ Configuration:")
    print(f"   Use subset: True (500 samples)")
    print(f"   Epochs: 2 (quick test)")
    print(f"   Method: NO underthesea")
    
    # Run comparisons
    results = []
    total_start = datetime.now()
    
    for i, model in enumerate(models):
        print(f"\n{'='*50}")
        print(f"🔄 [{i+1}/{len(models)}] {model}")
        print(f"{'='*50}")
        
        start_time = time.time()
        result = run_training(model, use_subset=True, epochs=2)
        end_time = time.time()
        
        result['wall_time'] = f"{end_time - start_time:.1f}s"
        results.append(result)
        
        # Print result
        if result['status'] == 'success':
            print(f"✅ Success: Accuracy={result['accuracy']:.4f}, F1={result['f1']:.4f}")
        else:
            print(f"❌ Failed: {result['status']} - {result['error']}")
    
    total_time = datetime.now() - total_start
    
    # Create comparison table
    print(f"\n" + "="*70)
    print(f"📊 COMPARISON RESULTS")
    print(f"="*70)
    
    df_data = []
    for r in results:
        df_data.append({
            'Model': r['model'].split('/')[-1],  # Short name
            'Accuracy': f"{r['accuracy']:.4f}" if r['accuracy'] else 'FAILED',
            'F1-Score': f"{r['f1']:.4f}" if r['f1'] else 'FAILED',
            'Time': r['wall_time'],
            'Status': r['status']
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    # Find best model
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        best = max(successful_results, key=lambda x: x['accuracy'])
        print(f"\n🏆 Best Model: {best['model']}")
        print(f"   Accuracy: {best['accuracy']:.4f}")
        print(f"   F1-Score: {best['f1']:.4f}")
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"   Total time: {total_time}")
    print(f"   Successful: {len(successful_results)}/{len(models)}")
    print(f"   Method: NO underthesea")
    print(f"   Dataset: Vietnamese clickbait (subset)")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"model_comparison_no_underthesea_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_time': str(total_time),
            'method': 'NO_UNDERTHESEA',
            'results': results
        }, f, indent=2)
    
    print(f"💾 Results saved: {results_file}")

if __name__ == "__main__":
    main() 