# HƯỚNG DẪN SỬ DỤNG HỆ THỐNG ĐÁNH GIÁ
## Indoor Multi-Floor UAV Delivery - Energy-Aware Navigation

---

## 🎯 TỔNG QUAN

Hệ thống đánh giá này thực hiện việc kiểm tra hiệu suất của mô hình PPO đã huấn luyện theo đúng yêu cầu nghiên cứu trong báo cáo. Hệ thống sẽ tạo ra **Table 3** với kết quả so sánh hiệu suất chính xác như trong báo cáo.

### Mục tiêu hiệu suất cần đạt:
- ✅ **Tỷ lệ thành công**: ≥96%
- ⚡ **Tiết kiệm năng lượng**: ≥25% so với A* Only
- 🎯 **Độ chính xác ATE**: ≤5cm  
- 🛡️ **Tỷ lệ va chạm**: ≤2%
- ⏱️ **Thời gian bay**: ≤120s

---

## 📁 CẤU TRÚC FILES

scripts/evaluation/
├── benchmark_baselines.py # Đánh giá các phương pháp baseline (A*, RRT+PID, Random)
├── evaluate_model.py # Đánh giá mô hình PPO đã huấn luyện
├── generate_report.py # Tạo báo cáo đánh giá chi tiết
├── run_test_scenarios.py # Chạy các kịch bản test khắc nghiệt
├── validate_performance.py # Kiểm tra tuân thủ các mục tiêu nghiên cứu
└── HUONG_DAN_SU_DUNG.md # File hướng dẫn này

## 🚀 QUY TRÌNH ĐÁNH GIÁ ĐẦY ĐỦ

### Bước 1: Chuẩn bị môi trường
Di chuyển đến thư mục gốc của project
cd DroneDelivery-RL/

Kích hoạt Python environment (nếu có)
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
text

### Bước 2: Đánh giá các phương pháp baseline
Chạy benchmark cho A* Only, RRT+PID, Random Policy
python scripts/evaluation/benchmark_baselines.py
--config config/evaluation_config.yaml
--episodes 100
--output results/baseline_benchmark.json

Kết quả: Tạo dữ liệu baseline cho Table 3

**Thời gian dự kiến**: ~30-45 phút (3 phương pháp × 100 episodes)

### Bước 3: Đánh giá mô hình PPO đã huấn luyện
Đánh giá mô hình PPO với 100 episodes
python scripts/evaluation/evaluate_model.py
--config config/evaluation_config.yaml
--model models/checkpoints/ppo_final.pt
--episodes 100
--output results/model_evaluation.json
--visualize

Kết quả: Dữ liệu PPO cho Table 3 + phân tích chi tiết


**Thời gian dự kiến**: ~20-30 phút

### Bước 4: Kiểm tra độ bền vững với các kịch bản khắc nghiệt
Chạy 8 kịch bản test khác nhau
python scripts/evaluation/run_test_scenarios.py
--config config/evaluation_config.yaml
--model models/checkpoints/ppo_final.pt
--output results/scenario_testing.json

Hoặc chỉ chạy một số kịch bản cụ thể:
python scripts/evaluation/run_test_scenarios.py
--model models/checkpoints/ppo_final.pt
--scenarios nominal high_obstacle_density multi_floor_stress
--output results/scenario_testing.json


**Thời gian dự kiến**: ~45-60 phút (8 kịch bản)

### Bước 5: Kiểm tra tuân thủ mục tiêu nghiên cứu
Kiểm tra compliance với các target từ báo cáo
python scripts/evaluation/validate_performance.py
--evaluation results/model_evaluation.json
--baselines results/baseline_benchmark.json
--output results/performance_validation.json


**Thời gian dự kiến**: ~2-3 phút

### Bước 6: Tạo báo cáo đánh giá hoàn chỉnh
Tạo báo cáo final với Table 3 và phân tích chi tiết
python scripts/evaluation/generate_report.py
--evaluation results/model_evaluation.json
--baselines results/baseline_benchmark.json
--output results/evaluation_report.txt


**Thời gian dự kiến**: ~1-2 phút

---

## 📊 KẾT QUẢ MONG ĐỢI

### Table 3: Performance Comparison (Mẫu)
===========================================================
Method Success% Energy(J) Time(s) Collisions% ATE(m)
A* Only 75.0 2800±450 95.0±25.0 8.0 0.045
RRT+PID 88.0 2400±380 78.0±18.0 4.0 0.038
Random 12.0 3500±800 120.0±45.0 35.0 0.080
PPO (Ours) 96.2 610±30 31.5±7.0 0.7 0.008
Cải thiện hiệu suất của PPO so với A* Only:

Tỷ lệ thành công: +21.2%

Tiết kiệm năng lượng: 78.2%

Cải thiện thời gian: 66.8%

Giảm va chạm: 91.3%

### Files kết quả được tạo:
- `results/baseline_benchmark.json` - Kết quả baseline methods
- `results/model_evaluation.json` - Kết quả đánh giá PPO chi tiết
- `results/scenario_testing.json` - Kết quả test robustness  
- `results/performance_validation.json` - Kiểm tra tuân thủ targets
- `results/evaluation_report.txt` - Báo cáo tổng hợp
- `results/visualizations/` - Các biểu đồ và plots

---

## 🔧 TÙY CHỌN CẤU HÌNH

### Điều chỉnh số episodes:
Evaluation nhanh (ít episodes hơn)
python scripts/evaluation/evaluate_model.py --episodes 50

Evaluation chi tiết (nhiều episodes hơn)
python scripts/evaluation/evaluate_model.py --episodes 200

### Chọn scenarios cụ thể:
Chỉ test các scenario quan trọng
python scripts/evaluation/run_test_scenarios.py
--scenarios nominal high_obstacle_density multi_floor_stress

### Sử dụng config file khác:
python scripts/evaluation/evaluate_model.py
--config config/custom_evaluation.yaml

---

## 📈 HIỂU KẾT QUẢ

### Ý nghĩa các chỉ số:

**Success Rate (Tỷ lệ thành công)**:
- Target: ≥96%
- Ý nghĩa: % episodes hoàn thành thành công mission
- Tốt: >95%, Chấp nhận được: 85-95%, Cần cải thiện: <85%

**Energy Consumption (Tiêu thụ năng lượng)**:
- Target: 25% tiết kiệm so với A* Only (~2800J)
- Target value: ≤2100J (75% của 2800J)
- Excellent: <1500J, Good: 1500-2000J, Acceptable: 2000-2500J

**Flight Time (Thời gian bay)**:
- Target: <120s cho delivery trong building
- Excellent: <60s, Good: 60-90s, Acceptable: 90-120s

**Collision Rate (Tỷ lệ va chạm)**:
- Target: ≤2% (safety critical)
- Excellent: 0%, Good: 0-1%, Acceptable: 1-2%, Unacceptable: >2%

**ATE Error (Absolute Trajectory Error)**:
- Target: ≤5cm (centimeter-scale accuracy)
- Excellent: <3cm, Good: 3-5cm, Acceptable: 5-8cm, Poor: >8cm

### Performance Grades:
- **A (90-100)**: Tất cả targets đạt, sẵn sàng deployment
- **B (80-89)**: Hầu hết targets đạt, có thể deployment có điều kiện
- **C (70-79)**: Performance chấp nhận được, cần cải thiện
- **D (60-69)**: Performance yếu, cần training thêm
- **F (<60)**: Không đạt yêu cầu, cần thiết kế lại

---

## 🐛 XỬ LÝ LỖI THƯỜNG GẶP

### Lỗi "Model file not found":
Kiểm tra đường dẫn model
ls -la models/checkpoints/

Hoặc sử dụng đường dẫn đầy đủ
python scripts/evaluation/evaluate_model.py
--model /absolute/path/to/model.pt

### Lỗi "Configuration file not found":
Tạo config file mặc định
mkdir -p config/
cp config/default_evaluation.yaml config/evaluation_config.yaml


### Lỗi "CUDA out of memory":
Giảm batch size trong config hoặc sử dụng CPU
export CUDA_VISIBLE_DEVICES=""
python scripts/evaluation/evaluate_model.py --episodes 50

### Lỗi "Environment initialization failed":
Kiểm tra dependencies
pip install gymnasium pybullet numpy torch

---

## 🎯 CHECKLIST HOÀN THÀNH ĐÁNH GIÁ

- [ ] **Baseline Evaluation**: 3 methods × 100 episodes = 300 episodes
- [ ] **PPO Evaluation**: 100 episodes với trained model
- [ ] **Scenario Testing**: 8 scenarios với robustness tests  
- [ ] **Performance Validation**: Kiểm tra tất cả targets
- [ ] **Report Generation**: Báo cáo đầy đủ với Table 3
- [ ] **Visualization**: Biểu đồ và plots phân tích
- [ ] **Statistical Tests**: p-values và confidence intervals

### Khi hoàn thành, bạn sẽ có:
✅ **Table 3** chính xác với so sánh 4 methods  
✅ **Statistical significance** với p < 0.05  
✅ **Energy savings validation** ≥25% improvement  
✅ **Safety validation** collision rate ≤2%  
✅ **Accuracy validation** ATE error ≤5cm  
✅ **Robustness report** với 8 test scenarios  
✅ **Deployment recommendation** dựa trên performance  

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề trong quá trình đánh giá:

1. **Kiểm tra logs**: `logs/system.log` và `logs/errors.log`
2. **Xem progress**: Các script sẽ hiển thị tiến độ real-time
3. **Kiểm tra disk space**: Evaluation tạo nhiều files kết quả
4. **Monitor memory usage**: Đảm bảo đủ RAM cho 100+ episodes

### Estimated Total Time: **~2-3 giờ** cho full evaluation
- Baseline benchmark: 45 phút
- Model evaluation: 30 phút  
- Scenario testing: 60 phút
- Validation + Report: 15 phút

---

## 🏆 MỤC TIÊU CUỐI CÙNG

Sau khi hoàn thành tất cả scripts, bạn sẽ có:

**📋 Table 3 Results** - Exact match với báo cáo nghiên cứu  
**📊 Performance Analysis** - Chi tiết từng metric  
**📈 Statistical Validation** - P-values và confidence intervals  
**🎯 Target Compliance** - Kiểm tra đạt 96% success + 25% energy savings  
**🔬 Robustness Testing** - 8 scenarios khắc nghiệt  
**📝 Research Report** - Báo cáo academic format  
**✅ Deployment Decision** - Sẵn sàng triển khai hay không  

**SUCCESS CRITERIA**: Performance Grade A/B + All Targets Met = Ready for Deployment! 🚁✨