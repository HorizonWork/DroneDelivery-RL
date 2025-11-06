# HƯỚNG DẪN SỬ DỤNG UTILITIES
## DroneDelivery-RL Utility Scripts

---

## 🎯 TỔNG QUAN

Utilities package cung cấp các công cụ hỗ trợ phân tích, thu thập dữ liệu, và visualization cho hệ thống DroneDelivery-RL. Các scripts này giúp:
- ⚡ **Phân tích năng lượng** chi tiết và optimization
- 📊 **Thu thập dataset** comprehensive cho research
- 🚀 **Export trajectories** nhiều format khác nhau
- 🎨 **Visualization** publication-quality plots

---

## 📁 CẤU TRÚC UTILITIES

scripts/utilities/
├── analyze_energy.py # ⚡ Energy consumption analysis
├── collect_data.py # 📊 Comprehensive data collection
├── export_trajectories.py # 🚀 Multi-format trajectory export
├── visualize_results.py # 🎨 Results visualization
└── HUONG_DAN_UTILITIES.md # 📖 File hướng dẫn này

Windows: Không cần (Python packages sẽ handle)

---

## ⚡ ANALYZE_ENERGY.PY

### Mục đích:
Phân tích chi tiết năng lượng tiêu thụ, so sánh với baselines, và identify optimization opportunities.

### Sử dụng cơ bản:
Phân tích energy từ training results
```bash
python scripts/utilities/analyze_energy.py \
--training-results results/training_results.json \
--output results/energy_analysis \
--visualize
```
Phân tích energy từ evaluation results
```bash
python scripts/utilities/analyze_energy.py \
--evaluation-results results/model_evaluation.json \
--output results/energy_analysis \
--visualize
```

Windows: Không cần (Python packages sẽ handle)


### Features chính:
- **Phase progression analysis**: Energy improvement qua 3 curriculum phases
- **Baseline comparison**: So sánh với A* Only (2800J), RRT+PID (2400J), Random (3500J)
- **Energy breakdown**: 70% thrust, 20% avionics, 5% communication, 5% other
- **Efficiency grading**: A (<500J), B (500-700J), C (700-1000J), D (1000-2000J), F (>2000J)
- **Optimization opportunities**: Thrust optimization, hovering reduction, consistency improvement
- **Battery life impact**: Missions per charge, degradation consideration

### Kết quả output:
results/energy_analysis/
├── energy_analysis_results.json # Complete analysis data
├── training_analysis/ # Training-specific plots
│ ├── energy_trends.png # Energy progression over training
│ ├── energy_distribution.png # Consumption distribution
│ ├── baseline_comparison.png # vs A*/RRT+PID/Random
│ └── component_breakdown.png # Energy component analysis
└── evaluation_analysis/ # Evaluation-specific plots
├── efficiency_grade.png # A/B/C grade visualization
└── battery_impact.png # Missions per charge analysis
Windows: Không cần (Python packages sẽ handle)


### Example results:
{
"efficiency_analysis": {
"mean_energy_consumption": 610,
"efficiency_grade": "A",
"baseline_comparison": {
"A_star_only": {
"energy_savings_percent": 78.2,
"target_met": true
}
},
"battery_life_impact": {
"missions_per_charge": 59,
"battery_efficiency": "excellent"
}
}
}
Windows: Không cần (Python packages sẽ handle)


---

## 📊 COLLECT_DATA.PY

### Mục đích:
Thu thập comprehensive dataset từ hệ thống drone để phục vụ research analysis và validation.

### Sử dụng cơ bản:
Thu thập data với trained model
```bash
python scripts/utilities/collect_data.py \
--model models/checkpoints/ppo_final.pt \
--episodes 200 \
--scenarios nominal high_obstacles multi_floor dynamic_environment \
--output data/collected_dataset
```
Thu thập data không có model (random policy)
```bash
python scripts/utilities/collect_data.py \
--episodes 100 \
--scenarios nominal \
--output data/baseline_dataset
```

Windows: Không cần (Python packages sẽ handle)


### Scenarios available:
- **nominal**: 3 floors, 15% obstacles, 3 dynamic obstacles
- **high_obstacles**: 3 floors, 30% obstacles, 6 dynamic obstacles
- **multi_floor**: 5 floors, 20% obstacles, 5 dynamic obstacles, complex layout
- **dynamic_environment**: 3 floors, 8 dynamic obstacles + humans

### Data collection rates:
- **Trajectory**: 20Hz (position, velocity)
- **Energy**: 20Hz (instantaneous consumption)  
- **SLAM**: 10Hz (pose estimates)
- **Sensors**: 100Hz (IMU, camera, lidar)
- **Control**: 20Hz (commands, thrust)

### Kết quả output:
data/collected_dataset/
├── complete_dataset.json # Full dataset JSON
├── trajectories.csv # Trajectory data
├── energy_profiles.csv # Energy consumption profiles
├── performance_metrics.csv # Episode performance data
└── metadata.json # Collection metadata
Windows: Không cần (Python packages will handle)


### Dataset structure:
{
"metadata": {
"total_episodes": 200,
"scenarios_collected": ["nominal", "high_obstacles", "multi_floor"],
"collection_time_hours": 2.3
},
"trajectories": {
"count": 200,
"data": [[episode_id, scenario, trajectory_points], ...]
},
"performance_metrics": {
"count": 200,
"data": [[episode_id, scenario, success, collision, reward, energy], ...]
}
}

Windows: Không cần (Python packages will handle)

---

## 🚀 EXPORT_TRAJECTORIES.PY

### Mục đích:
Export trajectory data sang nhiều formats khác nhau để sử dụng với different analysis tools.

### Sử dụng cơ bản:
Export từ evaluation results
```bash
python scripts/utilities/export_trajectories.py \
--evaluation-results results/model_evaluation.json \
--formats csv json numpy visualization \
--output data/exported_trajectories
```

Export từ training results
```bash
python scripts/utilities/export_trajectories.py \
--training-results results/training_results.json \
--formats matlab ros_bag \
--output data/training_trajectories
```

Windows: Không cần (Python packages will handle)

### Export formats support:
- **csv**: Structured CSV files cho data analysis
- **json**: JSON format cho web applications
- **numpy**: .npy arrays cho Python/NumPy analysis
- **matlab**: JSON format compatible với MATLAB
- **ros_bag**: ROS-compatible format cho robotics tools
- **visualization**: PNG plots và interactive visualizations

### Kết quả output structure:
data/exported_trajectories/
├── csv/
│ └── all_trajectories.csv # Combined CSV data
├── json/
│ └── trajectories.json # JSON format
├── numpy/
│ ├── trajectory_episode_0001.npy # Individual episodes
│ └── all_trajectories.npy # Combined array
├── matlab/
│ └── trajectories_matlab.json # MATLAB-compatible
├── ros/
│ └── trajectories_ros.json # ROS bag format
└── visualizations/
├── trajectories_3d.png # 3D trajectory plot
├── floor_1_trajectories.png # Floor plan views
├── floor_2_trajectories.png
├── ...
└── energy_vs_path_length.png # Energy correlation plot

Windows: Không cần (Python packages will handle)

### CSV format example:
episode_id,scenario,step,timestamp,x,y,z,energy_consumption,episode_reward
0,nominal,0,0.0,2.1,2.3,1.0,0.45,0
0,nominal,1,0.05,2.2,2.4,1.1,0.52,0.1
...

Windows: Không cần (Python packages will handle)

---

## 🎨 VISUALIZE_RESULTS.PY

### Mục đích:
Tạo publication-quality visualizations cho training và evaluation results, including Table 3 comparison.

### Sử dụng cơ bản:
Visualize training results
```bash
python scripts/utilities/visualize_results.py \
--training-results results/training_results.json \
--output results/visualizations
```

Visualize evaluation với baseline comparison
```bash
python scripts/utilities/visualize_results.py \
--evaluation-results results/model_evaluation.json \
--baseline-results results/baseline_benchmark.json \
--output results/visualizations
```

Visualize both training và evaluation
```bash
python scripts/utilities/visualize_results.py \
--training-results results/training_results.json \
--evaluation-results results/model_evaluation.json \
--baseline-results results/baseline_benchmark.json \
--output results/visualizations
```

Windows: Không cần (Python packages will handle)

### Visualization types:

#### Training Visualizations:
- **Training curves**: Episode rewards, success rate progression, energy consumption
- **Phase progression**: Performance improvement across curriculum phases
- **Energy analysis**: Energy trends, efficiency improvement over time
- **Performance distributions**: Statistical distributions of metrics

#### Evaluation Visualizations:
- **Table 3 comparison**: 4-method performance bar charts
- **Target achievements**: Visual indicators cho research targets
- **Statistical analysis**: Confidence intervals, significance tests
- **Energy efficiency**: Baseline comparisons, savings analysis

### Kết quả output:
results/visualizations/
├── training_curves.png # Training progress plots
├── phase_progression.png # Curriculum learning phases
├── energy_analysis.png # Energy trends analysis
├── performance_comparison_table3.png # Table 3 visualization
├── target_achievements.png # Research targets status
├── statistical_distributions.png # Performance distributions
└── energy_efficiency_analysis.png # Energy savings analysis

Windows: Không cần (Python packages will handle)

### Table 3 visualization example:
Performance Comparison (Table 3 Visualization)
Method Success% Energy(J) Time(s) Collisions%
A* Only [75.0%] [2800±450] [95.0] [8.0%]
RRT+PID [88.0%] [2400±380] [78.0] [4.0%]
Random [12.0%] [3500±800] [120.0] [35.0%]
PPO (Ours) [96.2%] [610±30] [31.5] [0.7%]
✅ All targets achieved with significant improvements

Windows: Không cần (Python packages will handle)

---

## 🔄 WORKFLOW INTEGRATION

### Complete analysis workflow:
1. Train model đầy đủ
```bash
python scripts/training/train_ppo.py --config config/main_config.yaml
```

2. Evaluate model
```bash
python scripts/evaluation/evaluate_model.py \
--model models/checkpoints/ppo_final.pt
```

3. Collect comprehensive data
```bash
python scripts/utilities/collect_data.py \
--model models/checkpoints/ppo_final.pt \
--episodes 200 \
--scenarios nominal high_obstacles multi_floor dynamic_environment
```

4. Analyze energy patterns
```bash
python scripts/utilities/analyze_energy.py \
--evaluation-results results/model_evaluation.json \
--training-results results/training_results.json \
--visualize
```

5. Export trajectories
```bash
python scripts/utilities/export_trajectories.py \
--evaluation-results results/model_evaluation.json \
--formats csv json visualization
```
6. Generate visualizations
```bash
python scripts/utilities/visualize_results.py \
--evaluation-results results/model_evaluation.json \
--baseline-results results/baseline_benchmark.json
```

Windows: Không cần (Python packages will handle)


### Research paper workflow:
For Table 3 generation
```bash
python scripts/evaluation/benchmark_baselines.py # Generate baselines
python scripts/evaluation/evaluate_model.py # Evaluate PPO
python scripts/utilities/visualize_results.py \ # Create Table 3 visualization
--evaluation-results results/model_evaluation.json \
--baseline-results results/baseline_benchmark.json
```

For energy analysis section
```bash
python scripts/utilities/analyze_energy.py \
--evaluation-results results/model_evaluation.json \
--visualize
```

For trajectory analysis
```bash
python scripts/utilities/export_trajectories.py \
--evaluation-results results/model_evaluation.json \
--formats visualization
```

Windows: Không cần (Python packages will handle)

---

## 📊 DATA FORMATS VÀ STRUCTURES

### Training Results JSON structure:
{
"training_completed": true,
"total_timesteps": 5000000,
"total_episodes": 15234,
"training_time_hours": 12.3,
"final_evaluation": {
"success_rate": 96.2,
"mean_energy": 610
},
"training_history": {
"episode_rewards": [0, 10, 25, ...],
"episode_energies": [1200, 1150, 1100, ...],
"success_rates": [0, 5.0, 15.0, ...]
}
}
Windows: Không cần (Python packages will handle)


### Evaluation Results JSON structure:
{
"evaluation_completed": true,
"episodes_evaluated": 100,
"performance_metrics": {
"success_rate": 96.2,
"mean_energy": 610.5,
"std_energy": 32.1,
"mean_time": 31.5,
"collision_rate": 0.7,
"mean_ate": 0.008
},
"targets_met": {
"success_rate_96_percent": true,
"energy_savings_25_percent": true,
"collision_rate_2_percent": true,
"ate_error_5cm": true
}
}
Windows: Không cần (Python packages will handle)


---

## 🛠️ CUSTOMIZATION OPTIONS

### Energy Analysis Customization:
Analyze specific energy components
```bash
python scripts/utilities/analyze_energy.py \
--evaluation-results results/model_evaluation.json \
--config config/energy_analysis_config.yaml \
--visualize
```
Custom baseline values
```bash
python scripts/utilities/analyze_energy.py \
--training-results results/training_results.json \
--baseline-astar 2500 \ # Custom A* baseline
--baseline-rrt 2200 \ # Custom RRT baseline
--visualize
```

Windows: Không cần (Python packages will handle)


### Data Collection Customization:
High-frequency data collection
```bash
python scripts/utilities/collect_data.py \
--model models/checkpoints/ppo_final.pt \
--episodes 500 \
--scenarios nominal high_obstacles multi_floor dynamic_environment \
--sampling-rate 50 # 50Hz instead of 20Hz
```
Specific scenario focus
```bash
python scripts/utilities/collect_data.py \
--model models/checkpoints/ppo_final.pt \
--episodes 100 \
--scenarios multi_floor \ # Focus on 5-floor scenarios only
--detailed-sensors # Include detailed sensor data
```

Windows: Không cần (Python packages will handle)


### Trajectory Export Customization:
High-precision export
```bash
python scripts/utilities/export_trajectories.py \
--evaluation-results results/model_evaluation.json \
--formats csv numpy \
--precision 8 \ # 8 decimal places
--coordinate-system ENU # East-North-Up instead of NED
```
Custom time base
```bash
python scripts/utilities/export_trajectories.py \
--training-results results/training_results.json \
--formats json visualization \
--time-base 50 # 50Hz instead of 20Hz
```

Windows: Không cần (Python packages will handle)


### Visualization Customization:
High-resolution plots
```bash
python scripts/utilities/visualize_results.py \
--evaluation-results results/model_evaluation.json \
--baseline-results results/baseline_benchmark.json \
--output results/high_res_visualizations \
--dpi 600 \ # High DPI for publications
--figure-size 16 12 # Large figures
```
Custom color scheme
```bash
python scripts/utilities/visualize_results.py \
--evaluation-results results/model_evaluation.json \
--color-scheme publication \ # Publication-friendly colors
--style ieee # IEEE paper style
```

Windows: Không cần (Python packages will handle)


---

## 📈 EXPECTED RESULTS

### Energy Analysis Results:
⚡ ENERGY ANALYSIS SUMMARY
Mean Energy: 610J
Efficiency Grade: A
Energy Savings vs A*: 78.2%
Target Met: ✅ YES

Optimization Opportunities:

Thrust optimization: 15-25% potential savings

Consistency improvement: 5-15% potential savings

Hovering reduction: 10-20% potential savings

Battery Impact:

Missions per charge: 59

Missions per day: ~20

Battery efficiency: Excellent
Windows: Không cần (Python packages will handle)


### Data Collection Results:
📊 DATA COLLECTION SUMMARY
Episodes collected: 200
Scenarios: nominal, high_obstacles, multi_floor, dynamic_environment
Collection time: 1.8 hours
Trajectories: 200
Energy profiles: 200
Performance records: 200
Dataset saved to: data/collected_dataset

Data Breakdown:

Total trajectory points: 45,678

Total energy samples: 45,678

Success episodes: 193 (96.5%)

Average path length: 28.4m

Windows: Không cần (Python packages will handle)

### Trajectory Export Results:
📁 TRAJECTORY EXPORT SUMMARY
EVALUATION_EXPORT:
Trajectories: 100
Formats: csv, json, numpy, visualization
Files created: 12

TRAINING_EXPORT:
Trajectories: 50
Formats: matlab, ros_bag
Files created: 8

📂 Output directory: data/exported_trajectories

Files Generated:

CSV: all_trajectories.csv (structured data)

NumPy: 100 individual .npy + combined array

Visualizations: 3D plots + 5 floor plans

MATLAB: trajectories_matlab.json
Windows: Không cần (Python packages will handle)


### Visualization Results:
🎨 VISUALIZATION SUMMARY
Training visualizations: 8 plots created
Evaluation visualizations: 12 plots created

Key Plots Generated:
✅ training_curves.png - Learning progress
✅ performance_comparison_table3.png - Main results
✅ target_achievements.png - Research targets status
✅ energy_efficiency_analysis.png - Energy savings
✅ statistical_distributions.png - Performance distributions

🎯 All visualizations saved to: results/visualizations
Windows: Không cần (Python packages will handle)


---

## 🔍 ANALYSIS USE CASES

### Use Case 1: Research Paper Figures
Generate Figure 1: Training curves
```bash
python scripts/utilities/visualize_results.py \
--training-results results/training_results.json \
--output figures/paper_figures
```
Generate Table 3: Performance comparison
```bash
python scripts/utilities/visualize_results.py \
--evaluation-results results/model_evaluation.json \
--baseline-results results/baseline_benchmark.json \
--output figures/paper_figures
```

Generate Figure 2: Energy analysis
```bash
python scripts/utilities/analyze_energy.py \
--evaluation-results results/model_evaluation.json \
--visualize \
--output figures/energy_figures
```

Windows: Không cần (Python packages will handle)


### Use Case 2: Dataset Preparation
Collect large dataset for analysis
```bash
python scripts/utilities/collect_data.py \
--model models/checkpoints/ppo_final.pt \
--episodes 1000 \
--scenarios nominal high_obstacles multi_floor dynamic_environment \
--output data/research_dataset
```
Export for external tools
```bash
python scripts/utilities/export_trajectories.py \
--training-results results/training_results.json \
--formats csv matlab ros_bag \
--output data/external_analysis
```

Windows: Không cần (Python packages will handle)


### Use Case 3: Performance Analysis
Complete energy analysis
```bash
python scripts/utilities/analyze_energy.py \
--training-results results/training_results.json \
--evaluation-results results/model_evaluation.json \
--visualize
```
Comprehensive visualization
```bash
python scripts/utilities/visualize_results.py \
--training-results results/training_results.json \
--evaluation-results results/model_evaluation.json \
--baseline-results results/baseline_benchmark.json
```

Windows: Không cần (Python packages will handle)


---

## 🐛 TROUBLESHOOTING

### Issue 1: "No training data found"
Kiểm tra file path
```bash
ls -la results/training_results.json
```
Hoặc dùng relative path
```bash
python scripts/utilities/analyze_energy.py \
--training-results ./results/training_results.json
```

Windows: Không cần (Python packages will handle)


### Issue 2: "Visualization generation failed"
Cài thêm visualization packages
```bash
conda activate drone_delivery_rl
pip install seaborn plotly
```
Hoặc dùng basic plotting
```bash
python scripts/utilities/visualize_results.py \
--evaluation-results results/model_evaluation.json \
--basic-plots # Simple matplotlib plots only
```

Windows: Không cần (Python packages will handle)


### Issue 3: "Memory error during data collection"
Giảm số episodes
```bash
python scripts/utilities/collect_data.py \
--episodes 50 \ # Thay vì 200
--scenarios nominal # Chỉ 1 scenario
```
Hoặc collect từng batch
```bash
python scripts/utilities/collect_data.py --episodes 100 --output data/batch1
python scripts/utilities/collect_data.py --episodes 100 --output data/batch2
```

Windows: Không cần (Python packages will handle)


### Issue 4: "Export format not supported"
Check supported formats
```bash
python scripts/utilities/export_trajectories.py --help
```
Use supported formats only
```bash
python scripts/utilities/export_trajectories.py \
--evaluation-results results/model_evaluation.json \
--formats csv json numpy # Verified formats
```

Windows: Không cần (Python packages will handle)


---

## 📊 INTEGRATION VỚI WORKFLOW

### Pre-publication workflow:
1. Complete evaluation
```bash
python scripts/evaluation/evaluate_model.py \
--model models/checkpoints/ppo_final.pt
```

2. Baseline comparison
```bash
python scripts/evaluation/benchmark_baselines.py
```

3. Energy analysis
```bash
python scripts/utilities/analyze_energy.py \
--evaluation-results results/model_evaluation.json \
--visualize
```

4. Generate paper figures
```bash
python scripts/utilities/visualize_results.py \
--evaluation-results results/model_evaluation.json \
--baseline-results results/baseline_benchmark.json
```
5. Export data for external validation
```bash
python scripts/utilities/export_trajectories.py \
--evaluation-results results/model_evaluation.json \
--formats csv json matlab
```

Windows: Không cần (Python packages will handle)


### Data sharing workflow:
1. Collect standardized dataset
```bash
python scripts/utilities/collect_data.py \
--model models/checkpoints/ppo_final.pt \
--episodes 500 \
--scenarios nominal high_obstacles multi_floor
```

2. Export multiple formats
```bash
python scripts/utilities/export_trajectories.py \
--training-results results/training_results.json \
--formats csv json numpy matlab ros_bag
```
3. Create documentation plots
```bash
python scripts/utilities/visualize_results.py \
--training-results results/training_results.json \
--output documentation/plots
```

Windows: Không cần (Python packages will handle)


---

## 🎯 SUCCESS METRICS

### Energy Analysis Success:
- ✅ **Energy efficiency grade A**: <700J consumption
- ✅ **25%+ energy savings** vs A* Only baseline
- ✅ **Battery efficiency**: 50+ missions per charge
- ✅ **Optimization opportunities** identified
- ✅ **Component breakdown** detailed analysis

### Data Collection Success:
- ✅ **200+ episodes** collected successfully  
- ✅ **Multiple scenarios** comprehensive coverage
- ✅ **High-frequency data**: 20Hz trajectory, 100Hz sensors
- ✅ **Performance metrics**: Success rate, energy, safety
- ✅ **Export formats** CSV, JSON ready

### Visualization Success:
- ✅ **Publication quality**: 300 DPI, proper formatting
- ✅ **Table 3 visualization**: 4-method comparison
- ✅ **Target achievements**: All research targets visualized
- ✅ **Statistical significance**: Confidence intervals included
- ✅ **Energy analysis**: Comprehensive efficiency plots

---

## 🏆 FINAL DELIVERABLES

Khi hoàn thành tất cả utilities, bạn sẽ có:

### 📊 Analysis Results:
- **Energy analysis report**: Efficiency grades, optimization recommendations
- **Performance datasets**: Structured data trong multiple formats
- **Statistical validation**: Confidence intervals, significance tests

### 🎨 Visualizations:
- **Table 3 comparison**: Publication-ready performance comparison  
- **Training curves**: Learning progress visualization
- **Energy trends**: Efficiency improvement plots
- **Target achievements**: Research objectives status

### 📁 Data Exports:
- **CSV files**: Ready for Excel, Python analysis
- **JSON files**: Ready for web applications
- **NumPy arrays**: Ready for scientific computing
- **MATLAB data**: Ready for MATLAB/Simulink
- **ROS bags**: Ready for robotics analysis

**Estimated total time**: 2-4 giờ cho complete analysis workflow

**🎉 Complete utilities system sẵn sàng cho comprehensive research analysis!** ⚡📊🎨🚁✨
