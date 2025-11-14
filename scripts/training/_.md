---

Hướng dẫn chi tiết quá trình huấn luyện mô hình PPO cho hệ thống điều hướng drone:
- Huấn luyện curriculum learning (1 tầng  2 tầng  5 tầng)
- Cấu hình siêu tham số theo Table 2
- Theo dõi quá trình huấn luyện
- Lưu và quản lý checkpoints

---

bash
python scripts/setup/verify_installation.py

pip list  grep -E "(torchgymnasiumstable-baselines3)"

- GPU: RTX 3070+ (khuyến nghị) hoặc CPU 8+ cores
- RAM: 16GB+ (32GB khuyến nghị)
- Storage: 50GB+ cho checkpoints và logs
- Thời gian: 8-12 giờ cho 5 triệu timestep

---

bash
python scripts/training/train_full_curriculum.py

python scripts/training/train_full_curriculum.py \
    --config config/training/ppo_hyperparameters.yaml \
    --output-dir models/checkpoints \
    --log-dir logs/training \
    --total-timesteps 50000 \
    --name curriculum_training

bash
python scripts/training/train_phase.py \
    --phase 1 \
    --timesteps 10000 \
    --output-dir models/checkpoints/phase_1

python scripts/training/train_phase.py \
    --phase 2 \
    --timesteps 200000 \
    --resume models/checkpoints/phase_1/final_model.pt \
    --output-dir models/checkpoints/phase_2

python scripts/training/train_phase.py \
    --phase 3 \
    --timesteps 200000 \
    --resume models/checkpoints/phase_2/final_model.pt \
    --output-dir models/checkpoints/phase_3

---

yaml
ppo:
  learning_rate: 3.0e-4
  rollout_length: 2048
  batch_size: 64
  epochs_per_update: 10
  clip_range: 0.2
  discount_factor: 0.99
  gae_lambda: 0.95
  entropy_coefficient: 0.01
  value_loss_coefficient: 0.5
  max_grad_norm: 0.5

model:
  hidden_sizes: [256, 128, 64]
  activation: "tanh"

yaml
environment:
  curriculum:
    phases:
      - name: "single_floor"
        floors: 1
        timesteps: 1000000
        obstacles: ["static"]

      - name: "two_floors"
        floors: 2
        timesteps: 200000
        obstacles: ["static", "moving"]

      - name: "five_floors"
        floors: 5
        timesteps: 2000000
        obstacles: ["static", "moving", "dynamic"]

  reward:
    energy_efficiency_weight: 0.3
    success_weight: 0.5
    collision_penalty: 1000.0
    time_penalty: 0.1

  observation:
    normalize: true
    stack_frames: 1

  action:
    clip_actions: true
    scale_actions: true

---

bash
tensorboard --logdir logs/training

wandb login
wandb init --project drone-delivery-rl

 Metric  Mục tiêu  Ý nghĩa
---------------------------
 Policy Loss  Giảm dần  Mô hình học tốt
 Value Loss  Giảm dần  Giá trị ước lượng chính xác
 Entropy  Ổn định  Cân bằng exploration/exploitation
 Episode Reward  Tăng dần  Hiệu suất cải thiện
 Success Rate  95  Nhiệm vụ hoàn thành tốt

bash
tail -f logs/training/training.log

ls -la models/checkpoints/
watch -n 1 'ls -la models/checkpoints/'

---

bash
python -c "import torch; print(torch.cuda.is_available())"

export CUDA_VISIBLE_DEVICES=0
python scripts/training/train_ppo.py

python
from src.environment import ParallelDroneEnvironment

parallel_env = ParallelDroneEnvironment(
    num_envs=8,
    config=training_config
)

yaml
ppo:
  batch_size: 128
 rollout_length: 4096

---

bash
python scripts/training/resume_training.py \
    --checkpoint models/checkpoints/ppo_checkpoint_1000000.pt \
    --config config/training/ppo_hyperparameters.yaml \
    --additional-timesteps 4000000

bash
python scripts/training/train_ppo.py \
    --resume models/checkpoints/final_model.pt \
    --config config/training/new_environment_config.yaml \
    --learning-rate 1.0e-5

---

python
from src.rl.utils import CheckpointManager

checkpoint_manager = CheckpointManager(
    save_dir="models/checkpoints",
    save_freq=100000,
    max_checkpoints=5,
    metric_to_track="success_rate"
)

yaml
checkpoint:
  save_frequency: 100000
  save_best_only: false
  save_best_metric: "success_rate"
  keep_checkpoints: 10
 save_optimizer_state: true
  save_training_state: true

---

bash
python scripts/evaluation/evaluate_during_training.py \
    --checkpoint-dir models/checkpoints \
    --eval-freq 50000 \
    --num-episodes 20

python
early_stopping = {
    'patience': 10,
    'min_delta': 0.01,
    'metric': 'success_rate'
}

---

bash
python scripts/utilities/visualize_training.py \
    --log-dir logs/training \
    --output-dir results/training_curves

bash
python scripts/utilities/analyze_training_performance.py \
    --checkpoint models/checkpoints/final_model.pt \
    --metrics-dir logs/training/metrics

---

yaml
training:
  ppo:
    entropy_coefficient: 0.01
    clip_range: 0.2
    max_grad_norm: 0.5
    learning_rate_schedule: "linear"

bash
watch -n 1 'nvidia-smi'
htop

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

---

bash
python scripts/training/train_ppo.py --learning-rate 1.0e-5

python scripts/training/train_ppo.py --entropy-coeff 0.05

bash
ppo:
  rollout_length: 1024
  batch_size: 32

bash
export CUDA_VISIBLE_DEVICES=""

sudo swapon --show

---

bash
python scripts/evaluation/evaluate_model.py \
    --model models/checkpoints/final_model.pt \
    --episodes 100 \
    --render false

 Metric  Target  Expected
--------------------------
 Success Rate  96  96.2
 Energy Consumption  -  610J
 Flight Time  -  31.5s
 Collision Rate  2  0.7
 ATE Error  5cm  0.8cm

bash
python scripts/utilities/optimize_model.py \
    --input models/checkpoints/final_model.pt \
    --output models/optimized/final_model_optimized.pt

---

- Config reference: config/training/README.md
- Hyperparameter guide: docs/HYPERPARAMETER_GUIDE.md
- Troubleshooting: docs/ERROR_HANDLING.md

- scripts/training/hyperparameter_search.py - Tìm kiếm siêu tham số
- scripts/training/monitor_training.py - Giám sát huấn luyện
- scripts/utilities/export_model.py - Xuất mô hình cho deployment

 Mô hình PPO đã sẵn sàng cho quá trình huấn luyện!