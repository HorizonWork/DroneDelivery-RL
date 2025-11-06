# HƯỚNG DẪN SỬ DỤNG TIỆN ÍCH
## DroneDelivery-RL Utilities Guide

---

## 🎯 **MỤC TIÊU**

Hướng dẫn sử dụng các tiện ích đi kèm hệ thống:
- Phân tích năng lượng
- Trực quan hóa kết quả
- Xuất/nhập dữ liệu
- Công cụ debug và phân tích

---

## 📋 **DANH SÁCH TIỆN ÍCH**

### 1. Phân tích năng lượng
```bash
# scripts/utilities/analyze_energy.py
python scripts/utilities/analyze_energy.py \
    --evaluation-results results/model_evaluation.json \
    --output results/energy_analysis.csv
```

### 2. Trực quan hóa kết quả
```bash
# scripts/utilities/visualize_results.py
python scripts/utilities/visualize_results.py \
    --evaluation-results results/model_evaluation.json \
    --output-dir results/figures
```

### 3. Xuất dữ liệu quỹ đạo
```bash
# scripts/utilities/export_trajectories.py
python scripts/utilities/export_trajectories.py \
    --trajectories results/trajectories/ppo_trajectories.pkl \
    --format csv
```

### 4. Thu thập dữ liệu
```bash
# scripts/utilities/collect_data.py
python scripts/utilities/collect_data.py \
    --model models/checkpoints/final_model.pt \
    --episodes 100 \
    --output data/collected_trajectories.pkl
```

---

## ⚡ **PHÂN TÍCH NĂNG LƯỢNG**

### 1. Phân tích chi tiết năng lượng
```bash
# Phân tích chi tiết năng lượng tiêu thụ
python scripts/utilities/analyze_energy.py \
    --evaluation-results results/model_evaluation.json \
    --detailed-analysis true \
    --output results/detailed_energy_report.json

# Output bao gồm:
# - Energy per episode
# - Energy per distance traveled
# - Thrust energy vs hover energy
# - Power consumption patterns
# - Energy efficiency ratios
```

### 2. So sánh hiệu quả năng lượng
```bash
# So sánh giữa các phương pháp
python scripts/utilities/analyze_energy.py \
    --comparison true \
    --ppo-results results/ppo_evaluation.json \
    --baseline-results results/baseline_evaluation.json \
    --output results/energy_comparison.json
```

### 3. Biểu đồ năng lượng
```bash
# Tạo biểu đồ phân tích năng lượng
python scripts/utilities/visualize_results.py \
    --plot-type energy_consumption \
    --data results/energy_analysis.csv \
    --output results/figures/energy_consumption.png
```

---

## 📊 **TRỰC QUAN HÓA KẾT QUẢ**

### 1. Biểu đồ hiệu suất cơ bản
```bash
# Tạo tất cả các biểu đồ cơ bản
python scripts/utilities/visualize_results.py \
    --evaluation-results results/model_evaluation.json \
    --create-all-plots true \
    --output-dir results/figures

# Các loại biểu đồ được tạo:
# - Success rate over episodes
# - Energy consumption distribution
# - Flight time histogram
# - Collision rate timeline
# - ATE error analysis
```

### 2. Biểu đồ so sánh
```bash
# So sánh giữa các phương pháp
python scripts/utilities/visualize_results.py \
    --comparison true \
    --ppo-results results/ppo_evaluation.json \
    --baseline-results results/baseline_evaluation.json \
    --output-dir results/figures/comparison
```

### 3. Trực quan hóa quỹ đạo bay
```bash
# Visualize 3D trajectories
python scripts/utilities/visualize_results.py \
    --plot-type 3d_trajectory \
    --trajectories results/trajectories/ppo_trajectories.pkl \
    --output results/figures/3d_trajectories.html
```

---

## 📈 **PHÂN TÍCH KẾT QUẢ NÂNG CAO**

### 1. Phân tích thống kê
```bash
# Phân tích thống kê chi tiết
python scripts/utilities/analyze_results.py \
    --results results/model_evaluation.json \
    --statistical-analysis true \
    --confidence-level 0.95 \
    --output results/statistical_analysis.json

# Bao gồm:
# - Confidence intervals
# - Statistical significance tests
# - Variance analysis
# - Outlier detection
```

### 2. Phân tích nhạy cảm
```bash
# Phân tích nhạy cảm với tham số
python scripts/utilities/analyze_results.py \
    --sensitivity-analysis true \
    --parameter learning_rate \
    --range 1e-5:1e-3 \
    --output results/sensitivity_analysis.json
```

---

## 📁 **QUẢN LÝ DỮ LIỆU**

### 1. Xuất dữ liệu quỹ đạo
```bash
# Xuất quỹ đạo dưới nhiều định dạng
python scripts/utilities/export_trajectories.py \
    --input results/trajectories/ppo_trajectories.pkl \
    --format all \
    --output-dir data/exported_trajectories

# Định dạng hỗ trợ: CSV, JSON, TXT, PKL, MAT
```

### 2. Nhập dữ liệu quỹ đạo
```bash
# Nhập quỹ đạo từ file
python scripts/utilities/collect_data.py \
    --import-trajectories data/custom_trajectories.csv \
    --validate true \
    --output results/imported_trajectories.json
```

### 3. Chuyển đổi định dạng dữ liệu
```bash
# Chuyển đổi giữa các định dạng
python scripts/utilities/convert_data_format.py \
    --input data/trajectories.json \
    --output data/trajectories.csv \
    --format csv
```

---

## 🔧 **CÔNG CỤ DEBUG**

### 1. Phân tích hành động
```bash
# Phân tích phân phối hành động
python scripts/utilities/debug_analyzer.py \
    --model models/checkpoints/final_model.pt \
    --analyze-actions true \
    --episodes 10 \
    --output results/action_analysis.json
```

### 2. Phân tích observation
```bash
# Phân tích observation space
python scripts/utilities/debug_analyzer.py \
    --analyze-observations true \
    --episodes 5 \
    --output results/observation_analysis.json
```

### 3. Performance profiler
```bash
# Phân tích hiệu năng
python scripts/utilities/performance_profiler.py \
    --model models/checkpoints/final_model.pt \
    --episodes 20 \
    --profile-memory true \
    --profile-time true \
    --output results/performance_profile.json
```

---

## 🎯 **TIỆN ÍCH CHUYÊN NGHIỆP**

### 1. Hyperparameter analysis
```bash
# Phân tích ảnh hưởng siêu tham số
python scripts/utilities/hyperparameter_analyzer.py \
    --training-logs logs/training/ \
    --analyze-learning-curves true \
    --output results/hyperparameter_analysis.json
```

### 2. Model comparison
```bash
# So sánh nhiều mô hình
python scripts/utilities/model_comparator.py \
    --models-dir models/checkpoints/ \
    --evaluation-episodes 50 \
    --output results/model_comparison.json
```

### 3. Ablation study
```bash
# Phân tích thành phần (Ablation study)
python scripts/utilities/ablation_analyzer.py \
    --ablation-results results/ablation_study/ \
    --output results/ablation_analysis.json
```

---

## 📊 **BÁO CÁO TỰ ĐỘNG**

### 1. Tạo báo cáo hoàn chỉnh
```bash
# Tạo báo cáo phân tích tự động
python scripts/utilities/generate_report.py \
    --data-dir results/ \
    --output reports/analysis_report.pdf \
    --include-plots true \
    --include-statistics true
```

### 2. Tạo bảng kết quả
```bash
# Tạo bảng kết quả cho bài báo
python scripts/utilities/generate_tables.py \
    --results results/model_evaluation.json \
    --format latex \
    --output results/latex_tables/
```

---

## 🚀 **TỐI ƯU TIỆN ÍCH**

### 1. Batch processing
```bash
# Xử lý hàng loạt nhiều file
python scripts/utilities/batch_processor.py \
    --input-dir results/batch_input/ \
    --output-dir results/batch_output/ \
    --operation analyze_energy
```

### 2. Parallel utilities
```bash
# Chạy tiện ích song
python scripts/utilities/parallel_analyzer.py \
    --input results/large_dataset.json \
    --num-processes 8 \
    --output results/parallel_analysis.json
```

---

## 🛠️ **CÔNG CỤ PHÁT TRIỂN**

### 1. Configuration validator
```bash
# Kiểm tra cấu hình
python scripts/utilities/config_validator.py \
    --config config/training/ppo_hyperparameters.yaml \
    --validate-all true
```

### 2. Data validator
```bash
# Kiểm tra dữ liệu đầu vào
python scripts/utilities/data_validator.py \
    --data results/model_evaluation.json \
    --validate-schema true \
    --validate-values true
```

---

## ⚙️ **CẤU HÌNH TIỆN ÍCH**

### 1. Cấu hình mặc định
```yaml
# config/utilities/default_config.yaml
utilities:
  default_output_dir: "results/"
  default_input_dir: "data/"
  default_model_dir: "models/checkpoints/"
  
  visualization:
    dpi: 300
    format: "png"
    style: "seaborn"
    
  analysis:
    confidence_level: 0.95
    statistical_tests: true
    outlier_detection: true
    
  export:
    default_format: "json"
    include_metadata: true
    compression: true
```

---

## 🚨 **LƯU Ý KHI SỬ DỤNG**

### 1. Yêu cầu tài nguyên
```bash
# Một số tiện ích yêu cầu nhiều tài nguyên:
# - visualize_results: 4GB+ RAM, GPU recommended
# - analyze_energy: 2GB+ RAM
# - batch_processor: 8GB+ RAM for large datasets
# - parallel_analyzer: Multiple CPU cores
```

### 2. Định dạng đầu vào
```bash
# Các định dạng hỗ trợ:
# - JSON: {results: [...], metrics: {...}}
# - CSV: episode,success_rate,energy,time
# - PKL: Python pickle files
# - MAT: MATLAB format
```

---

## 📞 **HỖ TRỢ & TÀI NGUYÊN**

### Script tiện ích:
- `scripts/utilities/validate_results.py` - Kiểm tra kết quả hợp lệ
- `scripts/utilities/merge_results.py` - Gộp nhiều kết quả
- `scripts/utilities/filter_results.py` - Lọc kết quả theo tiêu chí
- `scripts/utilities/normalize_data.py` -Chuẩn hóa dữ liệu

### Tài liệu liên quan:
- **API Reference**: docs/UTILITIES_API.md
- **Data Format**: docs/DATA_FORMATS.md
- **Performance Tips**: docs/PERFORMANCE_GUIDE.md

---

## ✅ **KẾT LUẬN**

Các tiện ích trong thư mục `scripts/utilities/` cung cấp công cụ mạnh mẽ để:
- ✅ Phân tích kết quả huấn luyện và đánh giá
- ✅ Trực quan hóa dữ liệu phức tạp
- ✅ So sánh giữa các phương pháp
- ✅ Debug và tối ưu hệ thống
- ✅ Tạo báo cáo chuyên nghiệp
- ✅ Chuyển đổi định dạng dữ liệu

**🎉 Hệ thống tiện ích đã sẵn sàng hỗ trợ quá trình nghiên cứu và phát triển!**
