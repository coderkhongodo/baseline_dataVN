#!/usr/bin/env python3
"""
Vietnamese Prompting for Clickbait Classification
Phân loại clickbait tiếng Việt bằng các phương pháp prompting
Methods: Zero-shot, Few-shot, Chain-of-Thought (CoT)
"""

import os
import json
import time
import pandas as pd
import jsonlines
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

load_dotenv()

def initialize_vietnamese_llm():
    """Initialize ChatOpenAI model for Vietnamese"""
    llm = ChatOpenAI(
        model="deepseek-v3",  # Tested model that works well with Vietnamese
        temperature=0,
        api_key=os.environ.get("SHUBI_API_KEY"), 
        base_url=os.environ.get("SHUBI_URL")
    )
    return llm

def zero_shot_vietnamese_prompting(llm):
    """Zero-shot prompting tiếng Việt - không có ví dụ"""
    
    zero_shot_prompt = """Bạn là chuyên gia phân tích nội dung tin tức tiếng Việt. Hãy phân loại tiêu đề tin tức sau:

Định nghĩa nhãn:
- 0: Không phải clickbait (thông tin rõ ràng, khách quan, có thực chất, truyền đạt tin tức một cách trung thực)
- 1: Clickbait (câu nói giật tít, gây tò mò, thông tin mơ hồ, phóng đại, dùng cảm xúc để thu hút clicks)

Tiêu đề cần phân loại: "{title}"

Chỉ trả lời bằng số 0 hoặc 1:"""

    # Test examples tiếng Việt
    test_titles = [
        "Bạn sẽ không tin được điều xảy ra khi cô gái này làm việc này!",
        "Chính phủ thông qua nghị định về tăng lương tối thiểu từ 1/1/2024",
        "7 bí mật mà bác sĩ không muốn bạn biết về sức khỏe!",
        "BIDV tăng lãi suất tiết kiệm lên 7.2% từ tháng tới"
    ]
    
    print("=== ZERO-SHOT PROMPTING TIẾNG VIỆT ===\n")
    
    for title in test_titles:
        prompt = zero_shot_prompt.format(title=title)
        print(f"Tiêu đề: {title}")
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()
            print(f"Kết quả zero-shot: {result}")
            
            # Interpret result
            if result == '1':
                print("Dự đoán: Clickbait")
            elif result == '0':
                print("Dự đoán: Không phải clickbait")
            else:
                print(f"Phản hồi không rõ ràng: {result}")
                
        except Exception as e:
            print(f"Lỗi: {e}")
        
        print("-" * 50)

def few_shot_vietnamese_prompting(llm):
    """Few-shot prompting với ví dụ tiếng Việt"""
    
    few_shot_prompt = """Bạn là chuyên gia phân loại clickbait tiếng Việt. Dưới đây là một số ví dụ từ dữ liệu training:

Ví dụ 1:
Tiêu đề: "Thủ tướng ký nghị định về thuế xuất khẩu gạo mới"
Nhãn: 0 (Không phải clickbait)
Lý do: Thông tin rõ ràng, khách quan về chính sách cụ thể, có nội dung thực chất

Ví dụ 2:
Tiêu đề: "Bạn sẽ sốc khi biết mức lương của nghề này! :))"
Nhãn: 1 (Clickbait)
Lý do: Gây tò mò, không nói rõ nghề gì và mức lương cụ thể, dùng emoji và cảm xúc để thu hút

Ví dụ 3:
Tiêu đề: "Giá vàng tăng 2% trong phiên giao dịch hôm nay"
Nhãn: 0 (Không phải clickbait)
Lý do: Thông tin cụ thể, có số liệu rõ ràng, nội dung tin tức thật

Ví dụ 4:
Tiêu đề: "Cách làm giàu mà 99% người Việt không biết - Bạn có tin không?"
Nhãn: 1 (Clickbait)
Lý do: Phóng đại (99%), hứa hẹn bí quyết nhưng không nói rõ, dùng câu hỏi gây tò mò

Ví dụ 5:
Tiêu đề: "Đội tuyển Việt Nam thắng 2-1 trước Thái Lan tại AFF Cup"
Nhãn: 0 (Không phải clickbait)
Lý do: Thông tin thể thao cụ thể, có kết quả rõ ràng, nội dung tin tức thật

Ví dụ 6:
Tiêu đề: "Điều xảy ra khi bạn ăn cái này mỗi ngày sẽ khiến bạn bất ngờ!"
Nhãn: 1 (Clickbait)
Lý do: Mơ hồ về "cái này", không nói rõ thực phẩm gì, tạo sự tò mò và bất ngờ

Bây giờ hãy phân loại tiêu đề sau: "{title}"

Định dạng trả lời:
Nhãn: [0/1]
Lý do: [giải thích ngắn gọn bằng tiếng Việt]"""

    test_titles = [
        "Cách kiếm tiền online mà sinh viên nào cũng nên biết",
        "Ngân hàng Nhà nước tăng lãi suất cơ bản lên 6% từ tuần tới",
        "Bí mật làm đẹp của sao Việt - Số 3 sẽ khiến bạn ngỡ ngàng!",
        "Dự báo thời tiết tuần tới: miền Bắc mưa rét, miền Nam nắng ấm"
    ]
    
    print("=== FEW-SHOT PROMPTING TIẾNG VIỆT ===\n")
    
    for title in test_titles:
        prompt = few_shot_prompt.format(title=title)
        print(f"Tiêu đề: {title}")
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()
            print("Kết quả few-shot:")
            print(result)
            
        except Exception as e:
            print(f"Lỗi: {e}")
        
        print("-" * 60)

def chain_of_thought_vietnamese(llm):
    """Chain of Thought prompting tiếng Việt"""
    
    cot_prompt = """Phân loại tiêu đề clickbait tiếng Việt theo từng bước cụ thể:

Tiêu đề: "{title}"

Hãy phân tích theo các bước sau:
1. Từ khóa cảm xúc - Tìm từ ngữ gây chú ý, phóng đại, tạo cảm xúc mạnh
2. Tính cụ thể của thông tin - Thông tin có rõ ràng, đầy đủ hay mơ hồ, thiếu sót
3. Khoảng trống tò mò - Có tạo ra câu hỏi mà không trả lời để gây tò mò không?
4. Giọng điệu và cấu trúc - Formal (trang trọng) hay clickbait style (giật tít)
5. Số liệu và phóng đại - Có sử dụng số liệu phóng đại (99%, 100%) hay không?
6. Phân loại cuối cùng với lý do đầy đủ

Định dạng trả lời:
Bước 1 - Từ khóa cảm xúc: [phân tích chi tiết]
Bước 2 - Tính cụ thể: [phân tích chi tiết]
Bước 3 - Khoảng trống tò mò: [phân tích chi tiết]
Bước 4 - Giọng điệu: [phân tích chi tiết]
Bước 5 - Số liệu phóng đại: [phân tích chi tiết]
Bước 6 - Phân loại: [0 (Không clickbait) / 1 (Clickbait)] - Lý do: [giải thích tổng hợp]"""

    test_titles = [
        "Wow, cách này giúp bạn giảm 10kg trong 1 tuần!",
        "Thống đốc NHNN họp báo về chính sách tiền tệ quý 4",
        "Bí quyết thành công mà tỷ phú Việt không bao giờ tiết lộ",
        "Giá xăng tăng 500 đồng/lít từ 15h hôm nay"
    ]
    
    print("=== CHAIN OF THOUGHT PROMPTING TIẾNG VIỆT ===\n")
    
    for title in test_titles:
        prompt = cot_prompt.format(title=title)
        print(f"Tiêu đề: {title}")
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()
            print("Kết quả Chain of Thought:")
            print(result)
            
        except Exception as e:
            print(f"Lỗi: {e}")
        
        print("=" * 80)

def load_vietnamese_evaluation_data(file_path, limit=20):
    """Load Vietnamese test data"""
    data = []
    try:
        if file_path.endswith('.jsonl'):
            with jsonlines.open(file_path) as reader:
                for i, obj in enumerate(reader):
                    if i >= limit:
                        break
                    data.append(obj)
        else:
            # Assume JSON format
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                data = raw_data[:limit]
                
        print(f"✅ Loaded {len(data)} Vietnamese samples for evaluation")
        return data
        
    except Exception as e:
        print(f"❌ Error loading Vietnamese data: {e}")
        return []

def classify_vietnamese_with_method(llm, title, method="zero_shot"):
    """Classify Vietnamese title với method cụ thể"""
    
    if method == "zero_shot":
        prompt = f"""Phân loại tiêu đề tin tức tiếng Việt:
- 0: Không phải clickbait (thông tin rõ ràng, khách quan)
- 1: Clickbait (giật tít, gây tò mò, phóng đại)

Tiêu đề: "{title}"

Chỉ trả lời 0 hoặc 1:"""

    elif method == "few_shot":
        prompt = f"""Dựa trên các ví dụ sau, phân loại tiêu đề:

Ví dụ:
"Thủ tướng ký nghị định mới" → 0 (tin tức thật)
"Bạn sẽ sốc khi biết điều này!" → 1 (clickbait)
"Giá vàng tăng 2% hôm nay" → 0 (tin tức thật)
"Cách làm giàu 99% người không biết" → 1 (clickbait)

Tiêu đề: "{title}"
Trả lời: [0/1]"""

    elif method == "cot":
        prompt = f"""Phân tích từng bước:
Tiêu đề: "{title}"

1. Từ khóa cảm xúc: 
2. Tính cụ thể: 
3. Tạo tò mò: 
4. Kết luận: [0/1]

Chỉ trả lời 0 hoặc 1 ở cuối:"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        # Extract label from response
        if method == "cot":
            # Look for final 0 or 1
            lines = response_text.split('\n')
            for line in reversed(lines):
                if '0' in line and '1' not in line:
                    return 0, response_text
                elif '1' in line and '0' not in line:
                    return 1, response_text
        else:
            # Direct classification
            if response_text == '1':
                return 1, response_text
            elif response_text == '0':
                return 0, response_text
            elif 'Trả lời: 1' in response_text or '[1]' in response_text:
                return 1, response_text
            elif 'Trả lời: 0' in response_text or '[0]' in response_text:
                return 0, response_text
        
        return -1, response_text  # Unable to parse
        
    except Exception as e:
        print(f"Lỗi phân loại: {e}")
        return -1, str(e)

def evaluate_vietnamese_prompting_method(llm, data, method="zero_shot", delay=1.0):
    """Evaluate Vietnamese prompting method"""
    
    print(f"\n🇻🇳 Đánh giá phương pháp {method.upper()} trên dữ liệu tiếng Việt")
    print("-" * 60)
    
    predictions = []
    true_labels = []
    valid_samples = 0
    
    for i, item in enumerate(data):
        title = item.get('text', item.get('title', ''))
        true_label = item.get('label', item.get('class', -1))
        
        if not title or true_label == -1:
            continue
            
        print(f"\nSample {i+1}: {title[:60]}...")
        
        pred_label, response = classify_vietnamese_with_method(llm, title, method)
        
        if pred_label != -1:
            predictions.append(pred_label)
            true_labels.append(true_label)
            valid_samples += 1
            
            # Show result
            pred_text = "Clickbait" if pred_label == 1 else "Không clickbait"
            true_text = "Clickbait" if true_label == 1 else "Không clickbait"
            correct = "✅" if pred_label == true_label else "❌"
            
            print(f"Dự đoán: {pred_text} | Thực tế: {true_text} {correct}")
            
            if method == "cot" and len(response) < 500:  # Show CoT reasoning if not too long
                print(f"Lý luận: {response[:200]}...")
        else:
            print("❌ Không thể phân loại")
        
        time.sleep(delay)  # Rate limiting
    
    # Calculate metrics
    if valid_samples > 0:
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ - {method.upper()}")
        print("=" * 50)
        print(f"Mẫu hợp lệ: {valid_samples}/{len(data)}")
        print(f"Độ chính xác: {accuracy:.3f}")
        print(f"Precision:    {precision:.3f}")
        print(f"Recall:       {recall:.3f}")
        print(f"F1-score:     {f1:.3f}")
        
        return {
            'method': method,
            'valid_samples': valid_samples,
            'total_samples': len(data),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    else:
        print("❌ Không có mẫu hợp lệ nào để đánh giá")
        return None

def run_vietnamese_evaluation_comparison(llm, data_limit=15):
    """So sánh các phương pháp prompting tiếng Việt"""
    
    print("🇻🇳 SO SÁNH CÁC PHƯƠNG PHÁP PROMPTING TIẾNG VIỆT")
    print("=" * 70)
    
    # Load Vietnamese test data
    test_data_path = "data_vietnamese/test/data.jsonl"
    if os.path.exists(test_data_path):
        data = load_vietnamese_evaluation_data(test_data_path, data_limit)
    else:
        # Use sample data if test file not found
        print("⚠️ Không tìm thấy file test Vietnamese, sử dụng dữ liệu mẫu")
        data = [
            {"text": "Chính phủ thông qua nghị định mới về thuế", "label": 0},
            {"text": "Bạn sẽ không tin được điều này!", "label": 1},
            {"text": "BIDV tăng lãi suất tiết kiệm lên 7.2%", "label": 0},
            {"text": "7 bí mật mà bác sĩ không muốn bạn biết", "label": 1},
            {"text": "Đội tuyển Việt Nam thắng 2-1 trước Thái Lan", "label": 0},
            {"text": "Cách làm giàu 90% người Việt chưa biết", "label": 1},
        ]
    
    if not data:
        print("❌ Không có dữ liệu để đánh giá")
        return
    
    methods = ["zero_shot", "few_shot", "cot"]
    results = []
    
    for method in methods:
        result = evaluate_vietnamese_prompting_method(llm, data, method, delay=1.5)
        if result:
            results.append(result)
    
    # Create comparison table
    if results:
        print(f"\n📈 BẢNG SO SÁNH KẾT QUẢ CUỐI CÙNG")
        print("=" * 70)
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format='%.3f'))
        
        # Find best method
        best_method = df.loc[df['f1_score'].idxmax()]
        print(f"\n🏆 PHƯƠNG PHÁP TỐT NHẤT: {best_method['method'].upper()}")
        print(f"   📊 F1-Score: {best_method['f1_score']:.3f}")
        print(f"   🎯 Accuracy: {best_method['accuracy']:.3f}")
        
        # Save results
        results_file = "vietnamese_prompting_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Kết quả đã lưu vào: {results_file}")

def main():
    """Main function"""
    print("🇻🇳 VIETNAMESE CLICKBAIT CLASSIFICATION - PROMPTING METHODS")
    print("=" * 70)
    
    # Initialize LLM
    llm = initialize_vietnamese_llm()
    
    try:
        # Test connection
        test_response = llm.invoke([HumanMessage(content="Xin chào")])
        print(f"✅ Kết nối thành công với LLM: {test_response.content}")
    except Exception as e:
        print(f"❌ Lỗi kết nối LLM: {e}")
        return
    
    print("\n" + "="*70)
    
    # Run demonstrations
    zero_shot_vietnamese_prompting(llm)
    print("\n" + "="*70)
    
    few_shot_vietnamese_prompting(llm)
    print("\n" + "="*70)
    
    chain_of_thought_vietnamese(llm)
    print("\n" + "="*70)
    
    # Run evaluation comparison
    run_vietnamese_evaluation_comparison(llm, data_limit=10)

if __name__ == "__main__":
    main() 