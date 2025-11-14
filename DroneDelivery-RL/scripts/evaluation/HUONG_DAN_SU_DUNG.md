---

Hướng dẫn sử dụng và đánh giá mô hình PPO đã huấn luyện:
- Đánh giá hiệu suất mô hình
- So sánh với các phương pháp baseline
- Phân tích năng lượng tiêu thụ
- Trực quan hóa kết quả

---

bash
ls -la models/checkpoints/

bash
python -c "
from src.environment import DroneEnvironment
import yaml

with open('config/evaluation/target_metrics.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(' Evaluation environment ready')
print(f'Metrics targets: {config.keys()}')
"

---

bash
python scripts/evaluation/evaluate_model.py \
    --model models/checkpoints/final_model.pt \
    --episodes 100 \
    --output results/model_evaluation.json

python scripts/evaluation/evaluate_model.py \
    --model models/checkpoints/final_model.pt \
    --episodes 50 \
    --render true \
    --save-trajectories true \
    --output results/detailed_evaluation.json

bash
python scripts/evaluation/benchmark_baselines.py

python scripts/evaluation/benchmark_baselines.py \
    --method all \
    --episodes 100 \
    --output results/baseline_comparison.csv

bash
python scripts/evaluation/run_test_scenarios.py \
    --config config/evaluation/test_scenarios.yaml \
    --model models/checkpoints/final_model.pt

python scripts/evaluation/run_test_scenarios.py \
    --scenario complex_navigation \
    --model models/checkpoints/final_model.pt

---

 Method  Success Rate  Energy (J)  Time (s)  Collisions  ATE (cm)
-------------------------------------------
 A Only  75.0  2800450  95.0  8.0  4.5
 RRT+PID  88.0  2400380  78.0  4.0  3.8
 Random  12.0  3500800  120.0  35.0  8.0
 PPO (Ours)  96.2  61030  31.5  0.7  0.8

bash
python scripts/evaluation/validate_performance.py \
    --results results/model_evaluation.json \
    --targets config/evaluation/target_metrics.yaml

---

yaml
metrics:
  success_rate:
    target: 0.96
    threshold: 0.95
    weight: 0.3

  energy_efficiency:
    target: 0.75
    threshold: 0.25
    weight: 0.25

  flight_time:
    target: 35.0
    threshold: 40.0
    weight: 0.15

  collision_rate:
    target: 0.02
    threshold: 0.02
    weight: 0.2

  ate_accuracy:
    target: 0.05
    threshold: 0.05
    weight: 0.1

yaml
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

---

bash
python -c "
from src.baselines import AStarBaseline
import numpy as np

baseline = AStarBaseline()
"

bash
python -c "
from src.baselines import RRTBaseline
import numpy as np

baseline = RRTBaseline()
"

bash
python -c "
from src.baselines import RandomBaseline
import numpy as np

baseline = RandomBaseline()
"

---

bash
python scripts/utilities/visualize_results.py \
    --evaluation-results results/model_evaluation.json \
    --baseline-results results/baseline_comparison.json \
    --output-dir results/figures

bash
python scripts/utilities/analyze_energy.py \
    --evaluation-results results/model_evaluation.json \
    --output results/energy_analysis.csv

---

bash
python scripts/evaluation/trajectory_analyzer.py \
    --trajectories results/trajectories/ppo_trajectories.pkl \
    --output results/trajectory_analysis.json

bash
python scripts/evaluation/energy_analyzer.py \
    --results results/model_evaluation.json \
    --output results/detailed_energy_report.json

---

python
from src.rl.evaluation import MetricsCollector

collector = MetricsCollector()
results = collector.collect_detailed_metrics(
    episodes=100,
    include_energy=True,
    include_trajectory=True,
    include_localization=True
)

python
from src.rl.evaluation import BaselineComparator

comparator = BaselineComparator()
comparison = comparator.compare_all_methods(
    ppo_model="models/checkpoints/final_model.pt",
    episodes=100
)

---

bash
python scripts/evaluation/monitor_evaluation.py \
    --log-file results/evaluation.log \
    --refresh-rate 5

bash
python scripts/evaluation/evaluate_model.py \
    --model models/checkpoints/final_model.pt \
    --episodes 100 \
    --early-termination true \
    --min-success-rate 0.90

---

bash
python scripts/evaluation/generate_report.py \
    --results results/model_evaluation.json \
    --output results/reports/evaluation_report.pdf

bash
python scripts/utilities/export_results.py \
    --input results/model_evaluation.json \
    --format all \
    --output results/exports/

---

bash
evaluation_config = {
    'same_environment': True,
    'same_start_positions': True,
    'same_target_positions': True,
    'same_obstacles': True,
    'same_random_seed': 42
}

bash
min_episodes = 50
recommended_episodes = 100
confidence_level = 0.95

---

bash
python scripts/evaluation/parallel_evaluation.py \
    --model models/checkpoints/final_model.pt \
    --episodes 100 \
    --num-processes 8 \
    --output results/parallel_evaluation.json

bash
python scripts/evaluation/batch_evaluation.py \
    --models-dir models/checkpoints/ \
    --output results/batch_results.json

---

bash
python scripts/evaluation/validate_research_targets.py \
    --results results/model_evaluation.json

bash
python scripts/evaluation/generate_performance_certificate.py \
    --results results/model_evaluation.json \
    --output results/certificates/performance_certificate.pdf

---

- scripts/evaluation/compare_simulators.py - So sánh PyBullet vs AirSim
- scripts/evaluation/ablation_study.py - Phân tích thành phần
- scripts/evaluation/sensitivity_analysis.py - Phân tích nhạy cảm
- scripts/utilities/convert_results_format.py - Chuyển đổi định dạng kết quả

- Metrics reference: docs/METRICS_REFERENCE.md
- Evaluation protocol: docs/EVALUATION_PROTOCOL.md
- Statistical analysis: docs/STATISTICAL_ANALYSIS.md

 Hệ thống đánh giá đã sẵn sàng để kiểm tra hiệu suất mô hình!