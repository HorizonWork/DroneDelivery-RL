# HƯỚNG DẪN HUẤN LUYỆN MÔ HÌNH
## DroneDelivery-RL PPO Training Guide

---

## 🎯 **MỤC TIÊU**

Hướng dẫn chi tiết quá trình huấn luyện mô hình PPO cho hệ thống điều hướng drone:
- Huấn luyện curriculum learning (1 tầng → 2 tầng → 5 tầng)
- Cấu hình siêu tham số theo Table 2
- Theo dõi quá trình huấn luyện
- Lưu và quản lý checkpoints

---

## 📋 **YÊU CẦU TRƯỚC HUẤN LUYỆN**

### 1. Môi trường đã cài đặt
```bash
# Kiểm tra cài đặt
python scripts/setup/verify_installation.py

# Đảm bảo các thư viện cần thiết
pip list | grep -E "(torch|gymnasium|stable-baselines3)"
```

### 2. Tài nguyên hệ thống
- **GPU**: RTX 3070+ (khuyến nghị) hoặc CPU 8+ cores
- **RAM**: 16GB+ (32GB khuyến nghị)
- **Storage**: 50GB+ cho checkpoints và logs
- **Thời gian**: 8-12 giờ cho 5 triệu timestep

---

## 🚀 **QUY TRÌNH HUẤN LUYỆN CƠ BẢN**

### 1. Huấn luyện toàn bộ (Full curriculum)
```bash
# Huấn luyện PPO với curriculum learning hoàn chỉnh
python scripts/training/train_full_curriculum.py

# Với các tùy chọn cụ thể
python scripts/training/train_full_curriculum.py \
    --config config/training/ppo_hyperparameters.yaml \
    --output-dir models/checkpoints \
    --log-dir logs/training \
    --total-timesteps 50000 \
    --name curriculum_training
```

### 2. Huấn luyện từng giai đoạn
```bash
# Giai đoạn 1: 1 tầng (1 triệu timestep)
python scripts/training/train_phase.py \
    --phase 1 \
    --timesteps 10000 \
    --output-dir models/checkpoints/phase_1

# Giai đoạn 2: 2 tầng (2 triệu timestep)  
# Resume từ phase 1
python scripts/training/train_phase.py \
    --phase 2 \
    --timesteps 200000 \
    --resume models/checkpoints/phase_1/final_model.pt \
    --output-dir models/checkpoints/phase_2

# Giai đoạn 3: 5 tầng (2 triệu timestep)
# Resume từ phase 2
python scripts/training/train_phase.py \
    --phase 3 \
    --timesteps 200000 \
    --resume models/checkpoints/phase_2/final_model.pt \
    --output-dir models/checkpoints/phase_3
```

---

## ⚙️ **CẤU HÌNH HUẤN LUYỆN**

### 1. Siêu tham số chính (Table 2)
```yaml
# config/training/ppo_hyperparameters.yaml
ppo:
  learning_rate: 3.0e-4           # Adam optimizer step size
  rollout_length: 2048            # Environment steps per update
  batch_size: 64                  # Size of mini-batches
  epochs_per_update: 10           # Number of passes over batch
  clip_range: 0.2                 # PPO clipping parameter
  discount_factor: 0.99           # Future rewards weighting
  gae_lambda: 0.95                # GAE parameter
  entropy_coefficient: 0.01       # Exploration encouragement
  value_loss_coefficient: 0.5     # Value loss weight
  max_grad_norm: 0.5              # Gradient clipping

model:
  hidden_sizes: [256, 128, 64]    # Network architecture
  activation: "tanh"              # Activation function
```

### 2. Cấu hình môi trường huấn luyện
```yaml
# config/training/environment_config.yaml
environment:
  # Curriculum configuration
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

  # Training-specific settings
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
```

---

## 📊 **THEO DÕI HUẤN LUYỆN**

### 1. TensorBoard monitoring
```bash
# Mở TensorBoard để theo dõi
tensorboard --logdir logs/training

# Hoặc dùng Weights & Biases
wandb login
wandb init --project drone-delivery-rl
```

### 2. Các metrics chính cần theo dõi
| Metric | Mục tiêu | Ý nghĩa |
|--------|----------|---------|
| **Policy Loss** | Giảm dần | Mô hình học tốt |
| **Value Loss** | Giảm dần | Giá trị ước lượng chính xác |
| **Entropy** | Ổn định | Cân bằng exploration/exploitation |
| **Episode Reward** | Tăng dần | Hiệu suất cải thiện |
| **Success Rate** | >95% | Nhiệm vụ hoàn thành tốt |

### 3. Giám sát thực thời
```bash
# Kiểm tra log huấn luyện
tail -f logs/training/training.log

# Kiểm tra checkpoints
ls -la models/checkpoints/
watch -n 1 'ls -la models/checkpoints/'
```

---

## 🔥 **TĂNG TỐC HUẤN LUYỆN**

### 1. Sử dụng GPU
```bash
# Đảm bảo CUDA available
python -c "import torch; print(torch.cuda.is_available())"

# Huấn luyện với GPU (mặc định)
export CUDA_VISIBLE_DEVICES=0
python scripts/training/train_ppo.py
```

### 2. Tăng số lượng môi trường song song
```python
# Trong training script
from src.environment import ParallelDroneEnvironment

# Tạo nhiều môi trường song song
parallel_env = ParallelDroneEnvironment(
    num_envs=8,  # Tăng số lượng môi trường
    config=training_config
)
```

### 3. Điều chỉnh batch size
```yaml
# Tăng batch size nếu có đủ RAM/GPU
ppo:
  batch_size: 128    # Thay vì 64
 rollout_length: 4096  # Thay vì 2048
```

---

## 🔄 **HUẤN LUYỆN TIẾP TỤC (RESUME)**

### 1. Resume từ checkpoint
```bash
# Resume huấn luyện từ checkpoint
python scripts/training/resume_training.py \
    --checkpoint models/checkpoints/ppo_checkpoint_1000000.pt \
    --config config/training/ppo_hyperparameters.yaml \
    --additional-timesteps 4000000
```

### 2. Fine-tuning từ mô hình đã huấn luyện
```bash
# Fine-tuning với môi trường mới
python scripts/training/train_ppo.py \
    --resume models/checkpoints/final_model.pt \
    --config config/training/new_environment_config.yaml \
    --learning-rate 1.0e-5  # Giảm learning rate
```

---

## 🛡️ **QUẢN LÝ CHECKPOINTS**

### 1. Tự động lưu checkpoints
```python
# CheckpointManager trong training
from src.rl.utils import CheckpointManager

checkpoint_manager = CheckpointManager(
    save_dir="models/checkpoints",
    save_freq=100000,      # Lưu mỗi 100k timesteps
    max_checkpoints=5,     # Giữ tối đa 5 checkpoints
    metric_to_track="success_rate"  # Theo dõi metric
)
```

### 2. Cấu hình lưu checkpoint
```yaml
# config/training/checkpoint_config.yaml
checkpoint:
  save_frequency: 100000     # Timesteps giữa các lần lưu
  save_best_only: false      # Lưu tất cả hoặc chỉ tốt nhất
  save_best_metric: "success_rate"  # Metric để đánh giá tốt nhất
  keep_checkpoints: 10       # Số lượng checkpoints giữ lại
 save_optimizer_state: true # Có lưu optimizer không
  save_training_state: true  # Có lưu trạng thái huấn luyện không
```

---

## 🧪 **KIỂM TRA TRONG HUẤN LUYỆN**

### 1. Đánh giá định kỳ
```bash
# Chạy evaluation trong quá trình training
python scripts/evaluation/evaluate_during_training.py \
    --checkpoint-dir models/checkpoints \
    --eval-freq 50000 \
    --num-episodes 20
```

### 2. Early stopping
```python
# Cấu hình early stopping
early_stopping = {
    'patience': 10,           # Số lượng evaluation không cải thiện
    'min_delta': 0.01,       # Cải thiện tối thiểu
    'metric': 'success_rate'  # Metric để theo dõi
}
```

---

## 📈 **PHÂN TÍCH KẾT QUẢ HUẤN LUYỆN**

### 1. Phân tích learning curves
```bash
# Tạo biểu đồ học tập
python scripts/utilities/visualize_training.py \
    --log-dir logs/training \
    --output-dir results/training_curves
```

### 2. Phân tích hiệu suất
```bash
# Phân tích chi tiết hiệu suất
python scripts/utilities/analyze_training_performance.py \
    --checkpoint models/checkpoints/final_model.pt \
    --metrics-dir logs/training/metrics
```

---

## ⚠️ **LƯU Ý QUAN TRỌNG**

### 1. Overfitting prevention
```yaml
# Cấu hình regularization
training:
  ppo:
    entropy_coefficient: 0.01    # Giữ exploration
    clip_range: 0.2             # Tránh cập nhật quá lớn
    max_grad_norm: 0.5          # Gradient clipping
    learning_rate_schedule: "linear"  # Giảm learning rate
```

### 2. Memory management
```bash
# Giám sát memory usage
watch -n 1 'nvidia-smi'  # GPU memory
htop # CPU & RAM usage

# Điều chỉnh nếu memory không đủ
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

---

## 🚨 **XỬ LÝ SỰ CỐ**

### 1. Huấn luyện không hội tụ
```bash
# Giảm learning rate
python scripts/training/train_ppo.py --learning-rate 1.0e-5

# Tăng entropy coefficient
python scripts/training/train_ppo.py --entropy-coeff 0.05
```

### 2. Memory overflow
```bash
# Giảm rollout length
# config/training/ppo_hyperparameters.yaml:
ppo:
  rollout_length: 1024  # Thay vì 2048
  batch_size: 32        # Thay vì 64
```

### 3. Out of memory (CUDA)
```bash
# Dùng CPU thay vì GPU
export CUDA_VISIBLE_DEVICES=""

# Hoặc tăng virtual memory
sudo swapon --show
```

---

## 🏆 **HOÀN THÀNH HUẤN LUYỆN**

### 1. Kiểm tra mô hình cuối cùng
```bash
# Đánh giá mô hình cuối cùng
python scripts/evaluation/evaluate_model.py \
    --model models/checkpoints/final_model.pt \
    --episodes 100 \
    --render false
```

### 2. Kết quả mong đợi (Table 3)
| Metric | Target | Expected |
|--------|--------|----------|
| **Success Rate** | ≥96% | 96.2% |
| **Energy Consumption** | - | 610J |
| **Flight Time** | - | 31.5s |
| **Collision Rate** | ≤2% | 0.7% |
| **ATE Error** | ≤5cm | 0.8cm |

### 3. Tối ưu mô hình
```bash
# Tối ưu mô hình cho inference
python scripts/utilities/optimize_model.py \
    --input models/checkpoints/final_model.pt \
    --output models/optimized/final_model_optimized.pt
```

---

## 📞 **HỖ TRỢ & TÀI NGUYÊN**

### Tài liệu liên quan:
- **Config reference**: config/training/README.md
- **Hyperparameter guide**: docs/HYPERPARAMETER_GUIDE.md  
- **Troubleshooting**: docs/ERROR_HANDLING.md

### Các script hữu ích:
- `scripts/training/hyperparameter_search.py` - Tìm kiếm siêu tham số
- `scripts/training/monitor_training.py` - Giám sát huấn luyện
- `scripts/utilities/export_model.py` - Xuất mô hình cho deployment

**🎉 Mô hình PPO đã sẵn sàng cho quá trình huấn luyện!**