# HƯỚNG DẪN HUẤN LUYỆN PPO
## Energy-Aware Indoor Drone Navigation System

---

## 🎯 TỔNG QUAN

Hệ thống huấn luyện PPO này thực hiện việc training agent để đạt được **96% success rate** và **25% energy savings** như trong báo cáo nghiên cứu. Sử dụng **curriculum learning** với 3 phases và tổng cộng **5 million timesteps**.

### Mục tiêu huấn luyện:
- 🎯 **Success Rate**: ≥96% navigation success
- ⚡ **Energy Savings**: ≥25% so với A* Only baseline  
- 🛡️ **Safety**: ≤2% collision rate
- 🎯 **Precision**: ≤5cm ATE localization error
- ⏱️ **Efficiency**: <120s flight time

---

## 📁 CẤU TRÚC TRAINING SCRIPTS

scripts/training/
├── train_ppo.py # 🚁 Main PPO training (5M timesteps)

├── train_phase.py # 🎯 Individual phase training
├── hyperparameter_search.py # 🔍 Auto hyperparameter optimization
├── monitor_training.py # 📊 Real-time monitoring
├── resume_training.py # 🔄 Resume từ checkpoint

Windows: Không cần (Python packages sẽ handle)

---

## 🚀 QUY TRÌNH HUẤN LUYỆN ĐẦY ĐỦ

### Bước 1: Chuẩn bị môi trường
Activate conda environment
```bash
conda activate drone_delivery_rl
```

Kiểm tra installation
```bash
python scripts/setup/verify_installation.py
```

Tạo thư mục kết quả
```bash
mkdir -p models/checkpoints results/training logs
```

Windows: Không cần (Python packages sẽ handle)

### Bước 2: (Tùy chọn) Tìm hyperparameters tối ưu
Chạy hyperparameter search (50 trials, ~24 giờ)
```bash
python scripts/training/hyperparameter_search.py \
--config config/main_config.yaml \
--trials 50 \
--timeout 24 \
--output results/hyperparameter_search
```

Kết quả: Tìm được best parameters cho PPO
Windows: Không cần (Python packages sẽ handle)

**Thời gian**: 12-24 giờ (có thể skip nếu dùng default parameters)

### Bước 3: Huấn luyện chính - Có 3 phương pháp

#### Phương pháp A: Main PPO Training (KHUYẾN NGHỊ) ⭐
Huấn luyện complete với curriculum learning
```bash
python scripts/training/train_ppo.py \
--config config/main_config.yaml \
--name ppo_energy_aware_5floors
```

Với best hyperparameters (nếu đã search)
```bash
python scripts/training/train_ppo.py \
--config results/hyperparameter_search/best_hyperparameters.json \
--name ppo_optimized
```

Windows: Không cần (Python packages sẽ handle)

#### Phương pháp B: Full Curriculum Training
Training từng phase tuần tự (3 phases)
```bash
python scripts/training/train_full_curriculum.py \
--config config/main_config.yaml \
--output-dir models/curriculum_training
```

Windows: Không cần (Python packages will handle)

#### Phương pháp C: Individual Phase Training
Train từng phase riêng biệt (debug mode)
```bash
python scripts/training/train_phase.py --phase single_floor
python scripts/training/train_phase.py --phase two_floor
python scripts/training/train_phase.py --phase five_floor
```

Windows: Không cần (Python packages will handle)

**Thời gian**: 8-15 giờ cho complete training

### Bước 4: Monitoring (chạy parallel)
Mở terminal thứ 2 để monitor training
```bash
python scripts/training/monitor_training.py \
--experiment ppo_energy_aware_5floors \
--interval 30
```

Sẽ hiển thị live dashboard với:
- Progress bar và remaining time
- Success rate, energy consumption
- System resources (CPU, RAM, GPU)
- Alerts nếu có vấn đề
Windows: Không cần (Python packages will handle)

### Bước 5: Resume nếu bị gián đoạn
Resume từ checkpoint mới nhất
```bash
python scripts/training/resume_training.py \
--checkpoint models/checkpoints/ppo_step_03000000_timestamp.pt \
--config config/main_config.yaml
```

Windows: Không cần (Python packages will handle)

---

## 📊 CURRICULUM LEARNING PHASES

### Phase 1: Single Floor Static (1M timesteps)
**Mục tiêu**: Học basic navigation và obstacle avoidance
- 🏢 **Environment**: 1 floor, static obstacles only
- 📊 **Target**: 85% success rate
- ⚡ **Energy**: ~800J per episode
- ⏱️ **Time**: ~2-3 giờ training

### Phase 2: Two Floor Dynamic (2M timesteps)  
**Mục tiêu**: Học multi-floor navigation và dynamic obstacles
- 🏢 **Environment**: 2 floors, 3 dynamic obstacles
- 📊 **Target**: 90% success rate  
- ⚡ **Energy**: ~700J per episode
- ⏱️ **Time**: ~4-5 giờ training

### Phase 3: Five Floor Complex (2M timesteps)
**Mục tiêu**: Mastery complex 5-floor navigation
- 🏢 **Environment**: 5 floors, 5 dynamic obstacles + humans
- 📊 **Target**: 96% success rate
- ⚡ **Energy**: ~610J per episode  
- ⏱️ **Time**: ~4-6 giờ training

### Automatic Phase Transition:
- **Success criteria met**: Auto advance to next phase
- **Early completion**: Phase có thể complete trước timestep limit
- **Failure handling**: Retry phase nếu không đạt target

---

## 📈 MONITORING VÀ TRACKING

### Live Training Dashboard:
==================================================
🚁
📊 TRAINING PROGRESS
Timestep: 2,347,891 / 5,000,000
Progress: [████████████░░░░░░] 46.9%

🎯 PERFORMANCE METRICS
Success Rate: 91.2%
Energy Consumption: 687J

💻 SYSTEM RESOURCES
CPU Usage: 78.4%
Memory Usage: 65.2%
Disk Free: 45.2GB
GPU Memory

💾 LATEST CHECKPOINT
File: ppo_step_02300000_20251106_201530.pt

⏱️ Training Time: 8.3 hours

Last Update: 20:26:15
Windows: Không cần (Python packages will handle)

### Alert System:
- ⚠️ **Low Success Rate**: <20% success rate alert
- ⚠️ **High Memory Usage**: >90% RAM usage
- ⚠️ **Training Stalled**: No checkpoint >5 phút  
- ⚠️ **Loss Explosion**: Policy loss >100

---

## 🔧 CẤU HÌNH TRAINING

### Default PPO Configuration:
```yaml
config/main_config.yaml
rl:
  ppo:
    learning_rate: 3e-4 # Optimized for drone control
    rollout_length: 2048 # Long episodes cho exploration
    epochs: 10 # PPO update epochs
    clip_range: 0.2 # PPO clipping parameter
    entropy_coef: 0.01 # Exploration bonus
    value_loss_coef: 0.5 # Value function weight
    gamma: 0.99
```

```yaml
training:
  total_timesteps: 5_000_000 # Research paper target
 eval_frequency: 50_000 # Evaluate mỗi 50k timesteps
```

Windows: Không cần (Python packages will handle)

### Customization Options:
Training với custom parameters
```bash
python scripts/training/train_ppo.py \
--config config/main_config.yaml \
--name custom_experiment \
--timesteps 3000000 # Override total timesteps
```

Training với GPU (nếu có)
```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/training/train_ppo.py
```

Training với reduced memory
```bash
python scripts/training/train_ppo.py \
--config config/low_memory_config.yaml
```

Windows: Không cần (Python packages will handle)

---

## 📊 TRACKING RESULTS

### Training Outputs:
- **Checkpoints**: `models/checkpoints/ppo_step_XXXXXXXX_timestamp.pt`
- **Final model**: `models/checkpoints/ppo_final.pt`  
- **Training logs**: `logs/training.log`
- **TensorBoard**: `runs/ppo_drone_YYYYMMDD_HHMMSS/`
- **Phase results**: `models/curriculum_training/phase_results.json`

### TensorBoard Visualization:
View training curves
```bash
tensorboard --logdir runs/
```

Browser: http://localhost:6006
Metrics available:
- Episode rewards, success rates
- Energy consumption trends
- Policy/value losses
- Learning curves per phase
Windows: Không cần (Python packages will handle)

### Key Metrics để theo dõi:
1. **Episode Reward**: Should increase from -100 to 500+
2. **Success Rate**: Target 85% → 90% → 96% across phases
3. **Energy Consumption**: Should decrease to ~610J final
4. **Policy Loss**: Should converge to <0.1
5. **Value Loss**: Should stabilize <1.0

---

## ⏱️ TIMELINE VÀ EXPECTATIONS

### Complete Training Timeline:
| Phase | Timesteps | Duration | Success Target | Energy Target |
|-------|-----------|----------|----------------|---------------|
| Phase 1 | 1M | 2-3 giờ | 85% | ~800J |
| Phase 2 | 2M | 4-5 giờ | 90% | ~700J |  
| Phase 3 | 2M | 4-6 giờ | 96% | ~610J |
| **TOTAL** | **5M** | **10-14 giờ** | **96%** | **610J** |

### Checkpoints Schedule:
- **Every 100k timesteps**: Automatic checkpoint save
- **Every 500k timesteps**: Full evaluation + best model save
- **Phase completions**: Phase-specific checkpoints
- **Final completion**: `ppo_final.pt` production-ready model

---

## 🔄 RESUME VÀ RECOVERY

### Training bị gián đoạn:
Tìm checkpoint mới nhất
```bash
ls -la models/checkpoints/ | grep ppo_step | tail -1
```

Resume từ checkpoint
```bash
python scripts/training/resume_training.py \
--checkpoint models/checkpoints/ppo_step_025000_20251106.pt \
--config config/main_config.yaml
```

Windows: Không cần (Python packages will handle)

### Recovery từ failed training:
Check logs để identify issue
```bash
tail -100 logs/training.log
```

Resume từ checkpoint stable trước đó
```bash
python scripts/training/resume_training.py \
--checkpoint models/checkpoints/ppo_step_02000000_20251106.pt \
--timesteps 50000 # Continue to full target
```

Windows: Không cần (Python packages will handle)

### Backup strategy:
Regular backup của important checkpoints
cp models/checkpoints/ppo_step_01000000_.pt backup/
```bash
cp models/checkpoints/ppo_step_0200000_.pt backup/
cp models/checkpoints/ppo_step_03000000_.pt backup/
```

Windows: Không cần (Python packages will handle)

---

## 🐛 TROUBLESHOOTING

### Issue 1: "CUDA out of memory"
Giảm batch size
```yaml
config/main_config.yaml:
rl:
  ppo:
    batch_size: 128 # Từ 256
```

Hoặc dùng CPU
```bash
export CUDA_VISIBLE_DEVICES=""
```

Windows: Không cần (Python packages will handle)

### Issue 2: "Training too slow"
Dùng GPU nếu có
```bash
export CUDA_VISIBLE_DEVICES=0
```
Tăng batch size (nếu có RAM)
```yaml
batch_size: 512
```
Dùng multiple processes (advanced)
```yaml
num_workers: 4
```

Windows: Không cần (Python packages will handle)

### Issue 3: "Success rate không improve"
Check hyperparameters
```bash
python scripts/training/hyperparameter_search.py --trials 20
```
Hoặc adjust learning rate
```yaml
learning_rate: 1e-4 # Giảm từ 3e-4
```
Tăng exploration
```yaml
entropy_coef: 0.02 # Tăng từ 0.01
```

Windows: Không cần (Python packages will handle)

### Issue 4: "Loss exploding"
Giảm learning rate
learning_rate: 1e-4
Adjust clip range
clip_range: 0.1 # Giảm từ 0.2
Check gradient clipping
max_grad_norm: 0.5
Windows: Không cần (Python packages will handle)

### Issue 5: "Training stalled"
Restart từ earlier checkpoint
```bash
python scripts/training/resume_training.py \
--checkpoint models/checkpoints/ppo_step_015000_*.pt
```

Hoặc adjust curriculum thresholds
Windows: Không cần (Python packages will handle)

---

## 📊 SUCCESS INDICATORS

### Phase 1 Success (1M timesteps):
- ✅ Success rate: 85%+ 
- ✅ Stable navigation trong single floor
- ✅ Basic obstacle avoidance  
- ✅ Energy consumption: ~800J

### Phase 2 Success (3M timesteps):
- ✅ Success rate: 90%+
- ✅ Multi-floor navigation
- ✅ Dynamic obstacle handling
- ✅ Energy optimization: ~700J  

### Phase 3 Success (5M timesteps):
- ✅ Success rate: 96%+
- ✅ Complex 5-floor navigation
- ✅ Human obstacle avoidance
- ✅ Energy efficiency: ~610J
- ✅ **READY FOR TABLE 3 EVALUATION**

### Final Success Criteria:
🎉 TRAINING HOÀN THÀNH KHI:
```yaml
✅ 5,000,000 timesteps completed
✅ Success rate: 96%+ achieved
✅ Energy consumption ≤700J average
✅ Collision rate ≤2%
✅ All 3 curriculum phases passed
```

Windows: Không cần (Python packages will handle)

---

## 💻 COMMANDS REFERENCE

### Huấn luyện cơ bản:
Standard training
```bash
python scripts/training/train_ppo.py
```

Với custom config
```bash
python scripts/training/train_ppo.py --config config/custom.yaml
```

Với experiment name
```bash
python scripts/training/train_ppo.py --name experiment_v2
```

Windows: Không cần (Python packages will handle)

### Curriculum training:
Complete curriculum
```bash
python scripts/training/train_full_curriculum.py
```

Individual phases
```bash
python scripts/training/train_phase.py --phase single_floor
python scripts/training/train_phase.py --phase two_floor
```

Windows: Không cần (Python packages will handle)

### Monitoring:
Real-time monitor
```bash
python scripts/training/monitor_training.py
```

Monitor specific experiment
```bash
python scripts/training/monitor_training.py --experiment ppo_v3
```

Windows: Không cần (Python packages will handle)

### Resume và recovery:
Resume training
```bash
python scripts/training/resume_training.py \
--checkpoint models/checkpoints/ppo_step_XXXXXXXX.pt
```

Resume với different target
```bash
python scripts/training/resume_training.py \
--checkpoint ppo_step_30000.pt \
--timesteps 6000000 # Extend training
```

Windows: Không cần (Python packages will handle)

### Hyperparameter optimization:
Quick search (10 trials)
```bash
python scripts/training/hyperparameter_search.py --trials 10 --timeout 6
```

Full search (50 trials)
```bash
python scripts/training/hyperparameter_search.py --trials 50 --timeout 24
```

Use found parameters
```bash
python scripts/training/train_ppo.py \
--config results/hyperparameter_search/best_hyperparameters.json
```

Windows: Không cần (Python packages will handle)

---

## 📋 TRAINING CHECKLIST

### Pre-Training:
- [ ] **Environment setup** completed
- [ ] **Config files** prepared
- [ ] **Disk space** ≥15GB available
- [ ] **Memory** ≥8GB available  
- [ ] **Time allocation** 10-15 giờ

### During Training:
- [ ] **Monitor progress** live dashboard
- [ ] **Check alerts** memory, success rate warnings
- [ ] **Backup checkpoints** important milestones
- [ ] **Log issues** any errors or stalls

### Post-Training:
- [ ] **Final model** `ppo_final.pt` created
- [ ] **Success rate** ≥96% achieved
- [ ] **Energy target** ≤700J achieved  
- [ ] **Training completed** 5M timesteps
- [ ] **Ready for evaluation** Table 3 generation

---

## 🎯 EXPECTED RESULTS

### Training Curves (TensorBoard):
- **Episode Rewards**: -100 → 500+ increasing trend
- **Success Rates**: 0% → 85% → 90% → 96% phase progression
- **Energy Consumption**: 1000J → 800J → 700J → 610J decreasing
- **Policy Loss**: High → Converge to <0.1
- **Value Loss**: Unstable → Stable <1.0

### Final Model Performance:
🏆 TRAINING SUCCESS METRICS:
✅ Success Rate: 96.2% (Target: ≥96%)
✅ Energy Consumption: ~610J average (Target: ≤700J)
✅ Energy Savings: 78% vs A* Only (Target: ≥25%)
✅ Flight Time: 31.5s average
✅ Collision Rate: 0.7% (Target: ≤2%)

🎯 READY FOR TABLE 3 EVALUATION!

Windows: Không cần (Python packages will handle)

---

## 🔍 MONITORING METRICS

### Key metrics để track:
1. **Episode Reward Trend**: Should show clear improvement
2. **Success Rate Progress**: Must reach 96% final
3. **Energy Efficiency**: Must show decreasing trend to 610J
4. **Policy Stability**: Loss convergence indicates learning
5. **System Health**: No memory leaks or resource issues

### Warning signs:
- 🚨 Success rate giảm hoặc stagnant
- 🚨 Energy consumption tăng
- 🚨 Loss exploding (>100)
- 🚨 Training stalled (no checkpoints >10 phút)
- 🚨 Memory usage >90%

---

## 🏆 COMPLETION CRITERIA

**Training считается thành công khi:**
1. ✅ **5,000,000 timesteps** hoàn thành
2. ✅ **96%+ success rate** stable trong 100+ episodes
3. ✅ **610J energy consumption** average achieved
4. ✅ **All curriculum phases** passed successfully
5. ✅ **Final model** `ppo_final.pt` saved
6. ✅ **Evaluation ready** cho Table 3 generation

**Estimated Total Time**: 10-15 giờ (depending on hardware)

**🎉 Success → Ready for comprehensive evaluation và Table 3 results generation!** 🚁📊✨