---

Hướng dẫn sử dụng các tiện ích đi kèm hệ thống:
- Phân tích năng lượng
- Trực quan hóa kết quả
- Xuất/nhập dữ liệu
- Công cụ debug và phân tích

---

bash
python scripts/utilities/analyze_energy.py \
    --evaluation-results results/model_evaluation.json \
    --output results/energy_analysis.csv

bash
python scripts/utilities/visualize_results.py \
    --evaluation-results results/model_evaluation.json \
    --output-dir results/figures

bash
python scripts/utilities/export_trajectories.py \
    --trajectories results/trajectories/ppo_trajectories.pkl \
    --format csv

bash
python scripts/utilities/collect_data.py \
    --model models/checkpoints/final_model.pt \
    --episodes 100 \
    --output data/collected_trajectories.pkl

---

bash
python scripts/utilities/analyze_energy.py \
    --evaluation-results results/model_evaluation.json \
    --detailed-analysis true \
    --output results/detailed_energy_report.json

bash
python scripts/utilities/analyze_energy.py \
    --comparison true \
    --ppo-results results/ppo_evaluation.json \
    --baseline-results results/baseline_evaluation.json \
    --output results/energy_comparison.json

bash
python scripts/utilities/visualize_results.py \
    --plot-type energy_consumption \
    --data results/energy_analysis.csv \
    --output results/figures/energy_consumption.png

---

bash
python scripts/utilities/visualize_results.py \
    --evaluation-results results/model_evaluation.json \
    --create-all-plots true \
    --output-dir results/figures

bash
python scripts/utilities/visualize_results.py \
    --comparison true \
    --ppo-results results/ppo_evaluation.json \
    --baseline-results results/baseline_evaluation.json \
    --output-dir results/figures/comparison

bash
python scripts/utilities/visualize_results.py \
    --plot-type 3d_trajectory \
    --trajectories results/trajectories/ppo_trajectories.pkl \
    --output results/figures/3d_trajectories.html

---

bash
python scripts/utilities/analyze_results.py \
    --results results/model_evaluation.json \
    --statistical-analysis true \
    --confidence-level 0.95 \
    --output results/statistical_analysis.json

bash
python scripts/utilities/analyze_results.py \
    --sensitivity-analysis true \
    --parameter learning_rate \
    --range 1e-5:1e-3 \
    --output results/sensitivity_analysis.json

---

bash
python scripts/utilities/export_trajectories.py \
    --input results/trajectories/ppo_trajectories.pkl \
    --format all \
    --output-dir data/exported_trajectories

bash
python scripts/utilities/collect_data.py \
    --import-trajectories data/custom_trajectories.csv \
    --validate true \
    --output results/imported_trajectories.json

bash
python scripts/utilities/convert_data_format.py \
    --input data/trajectories.json \
    --output data/trajectories.csv \
    --format csv

---

bash
python scripts/utilities/debug_analyzer.py \
    --model models/checkpoints/final_model.pt \
    --analyze-actions true \
    --episodes 10 \
    --output results/action_analysis.json

bash
python scripts/utilities/debug_analyzer.py \
    --analyze-observations true \
    --episodes 5 \
    --output results/observation_analysis.json

bash
python scripts/utilities/performance_profiler.py \
    --model models/checkpoints/final_model.pt \
    --episodes 20 \
    --profile-memory true \
    --profile-time true \
    --output results/performance_profile.json

---

bash
python scripts/utilities/hyperparameter_analyzer.py \
    --training-logs logs/training/ \
    --analyze-learning-curves true \
    --output results/hyperparameter_analysis.json

bash
python scripts/utilities/model_comparator.py \
    --models-dir models/checkpoints/ \
    --evaluation-episodes 50 \
    --output results/model_comparison.json

bash
python scripts/utilities/ablation_analyzer.py \
    --ablation-results results/ablation_study/ \
    --output results/ablation_analysis.json

---

bash
python scripts/utilities/generate_report.py \
    --data-dir results/ \
    --output reports/analysis_report.pdf \
    --include-plots true \
    --include-statistics true

bash
python scripts/utilities/generate_tables.py \
    --results results/model_evaluation.json \
    --format latex \
    --output results/latex_tables/

---

bash
python scripts/utilities/batch_processor.py \
    --input-dir results/batch_input/ \
    --output-dir results/batch_output/ \
    --operation analyze_energy

bash
python scripts/utilities/parallel_analyzer.py \
    --input results/large_dataset.json \
    --num-processes 8 \
    --output results/parallel_analysis.json

---

bash
python scripts/utilities/config_validator.py \
    --config config/training/ppo_hyperparameters.yaml \
    --validate-all true

bash
python scripts/utilities/data_validator.py \
    --data results/model_evaluation.json \
    --validate-schema true \
    --validate-values true

---

yaml
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

---

bash

bash

---

- scripts/utilities/validate_results.py - Kiểm tra kết quả hợp lệ
- scripts/utilities/merge_results.py - Gộp nhiều kết quả
- scripts/utilities/filter_results.py - Lọc kết quả theo tiêu chí
- scripts/utilities/normalize_data.py -Chuẩn hóa dữ liệu

- API Reference: docs/UTILITIES_API.md
- Data Format: docs/DATA_FORMATS.md
- Performance Tips: docs/PERFORMANCE_GUIDE.md

---

Các tiện ích trong thư mục scripts/utilities/ cung cấp công cụ mạnh mẽ để:
-  Phân tích kết quả huấn luyện và đánh giá
-  Trực quan hóa dữ liệu phức tạp
-  So sánh giữa các phương pháp
-  Debug và tối ưu hệ thống
-  Tạo báo cáo chuyên nghiệp
-  Chuyển đổi định dạng dữ liệu

 Hệ thống tiện ích đã sẵn sàng hỗ trợ quá trình nghiên cứu và phát triển!
