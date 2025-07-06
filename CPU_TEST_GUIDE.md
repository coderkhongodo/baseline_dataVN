# 🚀 CPU Test Guide - Vietnamese Clickbait Classification

Guide này hướng dẫn test nhanh Vietnamese clickbait classification trên CPU khi chưa có GPU.

## 📋 Chuẩn bị

### 1. Cài đặt dependencies
```bash
# Cài đặt dependencies cho CPU
pip install -r requirements_cpu.txt

# Hoặc cài đặt minimal
pip install torch transformers pandas scikit-learn numpy tqdm
```

### 2. Chuẩn bị dữ liệu
```bash
# Tạo training data (chỉ cần chạy 1 lần)
python scripts/prepare_training_data.py
```

## 🧪 Các loại test

### 1. Test Inference Only (Nhanh nhất - 1-2 phút)
Test với pre-trained PhoBERT mà không cần training:

```bash
python scripts/test_inference_cpu.py
```

**Kết quả mong đợi:**
- Load PhoBERT-base (~135M parameters)
- Test với 8 câu mẫu tiếng Việt
- Test với 20 câu từ dataset thật
- Tốc độ: ~5-10 texts/second trên CPU
- Accuracy trên sample: ~75-85%

### 2. Test Training (5-10 phút)
Training nhỏ với subset dữ liệu:

```bash
python scripts/test_cpu_quick.py
```

**Cấu hình:**
- Model: PhoBERT-base
- Training samples: 200
- Validation samples: 50
- Test samples: 50
- Epochs: 2
- Batch size: 8
- Max length: 64 tokens

**Kết quả mong đợi:**
- Training time: 5-10 phút
- Test accuracy: 70-85%
- Memory usage: ~2-4GB RAM

## 📊 Performance trên CPU

### Inference Speed
| Model | Parameters | Speed (texts/sec) | Memory |
|-------|------------|------------------|---------|
| PhoBERT-base | 135M | 5-10 | ~2GB |
| PhoBERT-large | 370M | 2-5 | ~4GB |

### Training Time (CPU)
| Samples | Epochs | Batch Size | Time | Accuracy |
|---------|--------|------------|------|----------|
| 200 | 2 | 8 | 5-10 min | 70-85% |
| 500 | 3 | 4 | 15-25 min | 75-88% |
| 1000 | 5 | 2 | 30-60 min | 80-90% |

## 🔧 Troubleshooting

### Lỗi thường gặp:

**1. Out of Memory**
```bash
# Giảm batch size
# Trong script, sửa CONFIG['batch_size'] = 4 hoặc 2
```

**2. Model download lỗi**
```bash
# Set proxy nếu cần
export HF_ENDPOINT=https://hf-mirror.com
```

**3. Slow performance**
```bash
# Set CPU threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

**4. Unicode errors**
```bash
# Ensure UTF-8 encoding
export PYTHONIOENCODING=utf-8
```

## 📈 Interpreting Results

### Accuracy Guidelines:
- **60-70%**: Baseline, model đang học
- **70-80%**: Acceptable cho test nhỏ
- **80-85%**: Good performance
- **85-90%**: Excellent (cần data nhiều hơn)
- **>90%**: Có thể overfitting (kiểm tra lại)

### Confusion Matrix:
```
Non-Clickbait predicted: [TN, FP]
Clickbait predicted:     [FN, TP]
```

- **High TN, TP**: Model hoạt động tốt
- **High FP**: Model dự đoán clickbait quá nhiều
- **High FN**: Model miss clickbait thật

## 🎯 Sample Test Cases

Một số câu test mẫu:

**Clickbait (label=1):**
- "Bạn sẽ không tin nổi điều gì xảy ra tiếp theo!"
- "10 bí mật mà chỉ có 1% dân số biết"
- "Cách kiếm tiền online mà 99% người không biết"

**Non-Clickbait (label=0):**
- "Thủ tướng chủ trì họp Chính phủ thường kỳ"
- "Giá xăng tăng 500 đồng từ 15h chiều nay"
- "Đại học Quốc gia Hà Nội tuyển sinh 2024"

## ⚡ Quick Start Commands

```bash
# Setup (1 lần)
pip install -r requirements_cpu.txt
python scripts/prepare_training_data.py

# Test inference (1-2 phút)
python scripts/test_inference_cpu.py

# Test training (5-10 phút)
python scripts/test_cpu_quick.py
```

## 🔜 Nâng cấp lên GPU

Khi có GPU, chuyển sang:
```bash
# Setup cho GPU
pip install -r requirements.txt
python scripts/run_rtx_a5000_training.py --strategy QUICK_TEST
```

## 📝 Notes

- CPU test chỉ để verify code và hiểu model
- Accuracy trên CPU test thấp hơn training full trên GPU
- Để có kết quả tốt nhất, cần training trên GPU với full dataset
- PhoBERT-base là choice tốt nhất cho CPU testing 