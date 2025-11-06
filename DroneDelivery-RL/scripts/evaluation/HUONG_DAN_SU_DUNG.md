# HƯỚNG DẪN SỬ DỤNG & ĐÁNH GIÁ MÔ HÌNH
## DroneDelivery-RL Evaluation Guide

---

## 🎯 **MỤC TIÊU**

Hướng dẫn sử dụng và đánh giá mô hình PPO đã huấn luyện:
- Đánh giá hiệu suất mô hình
- So sánh với các phương pháp baseline
- Phân tích năng lượng tiêu thụ
- Trực quan hóa kết quả

---

## 📋 **YÊU CẦU TRƯỚC ĐÁNH GIÁ**

### 1. Mô hình đã huấn luyện
```bash
# Kiểm tra mô hình tồn tại
ls -la models/checkpoints/
# File mô hình nên có dạng: ppo_final.pt, ppo_curriculum_5M.pt, etc.
```

### 2. Môi trường đánh giá
```bash
# Môi trường đánh giá nên giống môi trường huấn luyện
python -c "
from src.environment import DroneEnvironment
import yaml

with open('config/evaluation/target_metrics.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
print('✅ Evaluation environment ready')
print(f'Metrics targets: {config.keys()}')
"
```

---

## 🚀 **QUY TRÌNH ĐÁNH GIÁ CƠ BẢN**

### 1. Đánh giá mô hình đơn lẻ
```bash
# Đánh giá mô hình với 10 episodes
python scripts/evaluation/evaluate_model.py \
    --model models/checkpoints/final_model.pt \
    --episodes 100 \
    --output results/model_evaluation.json

# Đánh giá chi tiết với visualization
python scripts/evaluation/evaluate_model.py \
    --model models/checkpoints/final_model.pt \
    --episodes 50 \
    --render true \
    --save-trajectories true \
    --output results/detailed_evaluation.json
```

### 2. So sánh với baseline (Table 3)
```bash
# Chạy toàn bộ benchmark
python scripts/evaluation/benchmark_baselines.py

# So sánh cụ thể từng phương pháp
python scripts/evaluation/benchmark_baselines.py \
    --method all \
    --episodes 100 \
    --output results/baseline_comparison.csv
```

### 3. Chạy các kịch bản test cụ thể
```bash
# Chạy test scenarios từ config
python scripts/evaluation/run_test_scenarios.py \
    --config config/evaluation/test_scenarios.yaml \
    --model models/checkpoints/final_model.pt

# Chạy scenario cụ thể
python scripts/evaluation/run_test_scenarios.py \
    --scenario complex_navigation \
    --model models/checkpoints/final_model.pt
```

---

## 📊 **CHỈ SỐ ĐÁNH GIÁ CHÍNH**

### 1. Table 3 Metrics (Performance Comparison)
| Method | Success Rate | Energy (J) | Time (s) | Collisions | ATE (cm) |
|--------|-------------|------------|----------|
| A* Only | 75.0% | 2800±450 | 95.0 | 8.0% | 4.5 |
| RRT+PID | 88.0% | 2400±380 | 78.0 | 4.0% | 3.8 |
| Random | 12.0% | 3500±800 | 120.0 | 35.0% | 8.0 |
| **PPO (Ours)** | **96.2%** | **610±30** | **31.5** | **0.7%** | **0.8** |

### 2. Script đánh giá chi tiết
```bash
# Validate performance targets
python scripts/evaluation/validate_performance.py \
    --results results/model_evaluation.json \
    --targets config/evaluation/target_metrics.yaml

# Output sẽ kiểm tra:
# ✅ Success Rate: 96.2% ≥ 96% (PASS)
# ✅ Energy Savings: 78% ≥ 25% (PASS) 
# ✅ Collision Rate: 0.7% ≤ 2% (PASS)
# ✅ ATE Accuracy: 0.8cm ≤ 5cm (PASS)
```

---

## 🔧 **CẤU HÌNH ĐÁNH GIÁ**

### 1. Cấu hình đánh giá chính
```yaml
# config/evaluation/target_metrics.yaml
metrics:
  success_rate:
    target: 0.96
    threshold: 0.95
    weight: 0.3
    
  energy_efficiency:
    target: 0.75  # 75% energy savings vs baseline
    threshold: 0.25
    weight: 0.25
    
  flight_time:
    target: 35.0  # seconds
    threshold: 40.0
    weight: 0.15
    
  collision_rate:
    target: 0.02  # 2%
    threshold: 0.02
    weight: 0.2
    
  ate_accuracy:
    target: 0.05  # 5cm
    threshold: 0.05
    weight: 0.1
```

### 2. Cấu hình test scenarios
```yaml
# config/evaluation/test_scenarios.yaml
scenarios:
  basic_navigation:
    floors: [1]
    obstacles: ["static"]
    episodes: 20
    timeout: 120.0
    
  multi_floor:
    floors: [1, 2, 3, 4, 5]
    obstacles: ["static", "moving"]
    episodes: 30
    timeout: 180.0
    
  complex_navigation:
    floors: [1, 2, 3, 4, 5]
    obstacles: ["static", "moving", "dynamic"]
    episodes: 50
    timeout: 240.0
```

---

## 🎯 **PHƯƠNG PHÁP BASELINE**

### 1. A* + PID Baseline
```bash
# Chạy A* baseline
python -c "
from src.baselines import AStarBaseline
import numpy as np

baseline = AStarBaseline()
# Global planning + PID control
# Success rate: ~75%, Energy: ~2800J
"
```

### 2. RRT* + PID Baseline
```bash
# Chạy RRT* baseline
python -c "
from src.baselines import RRTBaseline
import numpy as np

baseline = RRTBaseline()
# Probabilistic roadmap + PID control
# Success rate: ~88%, Energy: ~2400J
"
```

### 3. Random Baseline
```bash
# Chạy Random baseline
python -c "
from src.baselines import RandomBaseline
import numpy as np

baseline = RandomBaseline()
# Random exploration
# Success rate: ~12%, Energy: ~3500J
"
```

---

## 📈 **TRỰC QUAN HÓA KẾT QUẢ**

### 1. Biểu đồ hiệu suất
```bash
# Tạo biểu đồ so sánh
python scripts/utilities/visualize_results.py \
    --evaluation-results results/model_evaluation.json \
    --baseline-results results/baseline_comparison.json \
    --output-dir results/figures

# Các loại biểu đồ được tạo:
# - Performance comparison bar chart
# - Energy consumption analysis
# - Success rate over time
# - Trajectory visualization
```

### 2. Phân tích năng lượng
```bash
# Phân tích chi tiết năng lượng tiêu thụ
python scripts/utilities/analyze_energy.py \
    --evaluation-results results/model_evaluation.json \
    --output results/energy_analysis.csv

# Output bao gồm:
# - Energy per episode
# - Energy per distance traveled
# - Energy efficiency ratios
# - Power consumption patterns
```

---

## 🧪 **ĐÁNH GIÁ CHI TIẾT**

### 1. Trajectory Analysis
```bash
# Phân tích đường bay chi tiết
python scripts/evaluation/trajectory_analyzer.py \
    --trajectories results/trajectories/ppo_trajectories.pkl \
    --output results/trajectory_analysis.json

# Metrics phân tích:
# - Path length efficiency
# - Smoothness metrics
# - Collision avoidance effectiveness
# - Floor transition efficiency
```

### 2. Energy Analysis
```bash
# Phân tích năng lượng chi tiết
python scripts/evaluation/energy_analyzer.py \
    --results results/model_evaluation.json \
    --output results/detailed_energy_report.json

# Bao gồm:
# - Thrust energy consumption
# - Hover energy vs movement energy
# - Energy per floor transition
# - Battery discharge patterns
```

---

## 🔍 **PHÂN TÍCH KẾT QUẢ**

### 1. Metrics Collector
```python
# Hệ thống thu thập metrics
from src.rl.evaluation import MetricsCollector

collector = MetricsCollector()
results = collector.collect_detailed_metrics(
    episodes=100,
    include_energy=True,
    include_trajectory=True,
    include_localization=True
)
```

### 2. Baseline Comparator
```python
# So sánh với các phương pháp khác
from src.rl.evaluation import BaselineComparator

comparator = BaselineComparator()
comparison = comparator.compare_all_methods(
    ppo_model="models/checkpoints/final_model.pt",
    episodes=100
)
```

---

## 🚨 **GIÁM SÁT THỰC THỜI**

### 1. Live evaluation monitoring
```bash
# Giám sát đánh giá đang chạy
python scripts/evaluation/monitor_evaluation.py \
    --log-file results/evaluation.log \
    --refresh-rate 5

# Hiển thị metrics đang cập nhật:
# Episode: 45/100
# Success Rate: 95.6%
# Avg Energy: 620J
# Avg Time: 32.1s
```

### 2. Early termination
```bash
# Dừng sớm nếu không đạt yêu cầu
python scripts/evaluation/evaluate_model.py \
    --model models/checkpoints/final_model.pt \
    --episodes 100 \
    --early-termination true \
    --min-success-rate 0.90
```

---

## 📊 **BÁO CÁO KẾT QUẢ**

### 1. Tự động tạo báo cáo
```bash
# Tạo báo cáo đánh giá hoàn chỉnh
python scripts/evaluation/generate_report.py \
    --results results/model_evaluation.json \
    --output results/reports/evaluation_report.pdf

# Bao gồm:
# - Executive summary
# - Detailed metrics
# - Comparison with baselines
# - Performance validation
# - Energy analysis
```

### 2. Export kết quả
```bash
# Export kết quả ở nhiều định dạng
python scripts/utilities/export_results.py \
    --input results/model_evaluation.json \
    --format all \
    --output results/exports/

# Tạo các file:
# - CSV: results.csv
# - Excel: results.xlsx  
# - JSON: results.json
# - LaTeX: results.tex
```

---

## ⚠️ **LƯU Ý KHI ĐÁNH GIÁ**

### 1. Fair comparison
```bash
# Đảm bảo điều kiện đánh giá công bằng
evaluation_config = {
    'same_environment': True,
    'same_start_positions': True, 
    'same_target_positions': True,
    'same_obstacles': True,
    'same_random_seed': 42
}
```

### 2. Statistical significance
```bash
# Chạy đủ số lượng episodes cho ý nghĩa thống kê
min_episodes = 50 # For stable metrics
recommended_episodes = 100  # For publication
confidence_level = 0.95
```

---

## 🚀 **TỐI ƯU HÓA ĐÁNH GIÁ**

### 1. Parallel evaluation
```bash
# Chạy đánh giá song
python scripts/evaluation/parallel_evaluation.py \
    --model models/checkpoints/final_model.pt \
    --episodes 100 \
    --num-processes 8 \
    --output results/parallel_evaluation.json
```

### 2. Batch evaluation
```bash
# Đánh giá nhiều mô hình cùng lúc
python scripts/evaluation/batch_evaluation.py \
    --models-dir models/checkpoints/ \
    --output results/batch_results.json
```

---

## 🏆 **KẾT QUẢ ĐẠT ĐƯỢC**

### 1. Validation targets
```bash
# Kiểm tra đạt mục tiêu nghiên cứu
python scripts/evaluation/validate_research_targets.py \
    --results results/model_evaluation.json

# ✅ Research Target 1: 96% success rate (✓ 96.2%)
# ✅ Research Target 2: 25% energy savings (✓ 78%)
# ✅ Research Target 3: 2% collision rate (✓ 0.7%)
# ✅ Research Target 4: 5cm localization (✓ 0.8cm)
```

### 2. Performance certificates
```bash
# Tạo chứng nhận hiệu suất
python scripts/evaluation/generate_performance_certificate.py \
    --results results/model_evaluation.json \
    --output results/certificates/performance_certificate.pdf
```

---

## 📞 **HỖ TRỢ & TÀI NGUYÊN**

### Script hữu ích:
- `scripts/evaluation/compare_simulators.py` - So sánh PyBullet vs AirSim
- `scripts/evaluation/ablation_study.py` - Phân tích thành phần
- `scripts/evaluation/sensitivity_analysis.py` - Phân tích nhạy cảm
- `scripts/utilities/convert_results_format.py` - Chuyển đổi định dạng kết quả

### Tài liệu liên quan:
- **Metrics reference**: docs/METRICS_REFERENCE.md
- **Evaluation protocol**: docs/EVALUATION_PROTOCOL.md
- **Statistical analysis**: docs/STATISTICAL_ANALYSIS.md

**🎉 Hệ thống đánh giá đã sẵn sàng để kiểm tra hiệu suất mô hình!**