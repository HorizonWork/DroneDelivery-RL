"""
TensorBoard Logger
Advanced logging for RL training with TensorBoard integration.
Supports scalar metrics, histograms, and custom visualizations.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
import json

class TensorBoardLogger:
    """
    Advanced TensorBoard logging for RL training.
    Provides comprehensive metrics visualization and monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Logger configuration
        self.log_dir = Path(config.get('log_dir', 'runs'))
        self.experiment_name = config.get('experiment_name', 'ppo_drone')
        self.flush_frequency = config.get('flush_frequency', 100)
        
        # Create experiment directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{self.experiment_name}_{timestamp}"
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.experiment_dir))
        
        # Logging state
        self.global_step = 0
        self.episode_count = 0
        self.log_counter = 0
        
        # Metric history for trend analysis
        self.metric_history = {}
        
        self.logger.info("TensorBoard Logger initialized")
        self.logger.info(f"Log directory: {self.experiment_dir}")
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log scalar value."""
        step = step if step is not None else self.global_step
        self.writer.add_scalar(tag, value, step)
        
        # Store in history
        if tag not in self.metric_history:
            self.metric_history[tag] = []
        self.metric_history[tag].append({'step': step, 'value': value})
        
        # Limit history size
        if len(self.metric_history[tag]) > 10000:
            self.metric_history[tag] = self.metric_history[tag][-5000:]
    
    def log_scalars(self, scalar_dict: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalars."""
        step = step if step is not None else self.global_step
        for tag, value in scalar_dict.items():
            self.log_scalar(tag, value, step)
    
    def log_histogram(self, tag: str, values: Union[np.ndarray, torch.Tensor], step: Optional[int] = None):
        """Log histogram of values."""
        step = step if step is not None else self.global_step
        
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        self.writer.add_histogram(tag, values, step)
    
    def log_training_metrics(self, metrics: Dict[str, Any], timestep: int):
        """Log comprehensive training metrics."""
        # Episode metrics
        if 'episode_reward' in metrics:
            self.log_scalar('Training/Episode_Reward', metrics['episode_reward'], timestep)
        
        if 'episode_length' in metrics:
            self.log_scalar('Training/Episode_Length', metrics['episode_length'], timestep)
        
        if 'success_rate' in metrics:
            self.log_scalar('Training/Success_Rate', metrics['success_rate'], timestep)
        
        # Energy metrics
        if 'energy_consumption' in metrics:
            self.log_scalar('Energy/Episode_Consumption', metrics['energy_consumption'], timestep)
        
        if 'energy_efficiency' in metrics:
            self.log_scalar('Energy/Efficiency_J_per_m', metrics['energy_efficiency'], timestep)
        
        # Policy metrics
        if 'policy_loss' in metrics:
            self.log_scalar('Training/Policy_Loss', metrics['policy_loss'], timestep)
        
        if 'value_loss' in metrics:
            self.log_scalar('Training/Value_Loss', metrics['value_loss'], timestep)
        
        if 'entropy' in metrics:
            self.log_scalar('Training/Entropy', metrics['entropy'], timestep)
    
    def log_evaluation_results(self, eval_results: Dict[str, Any], timestep: int):
        """Log evaluation results."""
        self.log_scalar('Evaluation/Success_Rate', eval_results.get('success_rate', 0), timestep)
        self.log_scalar('Evaluation/Mean_Energy', eval_results.get('mean_energy', 0), timestep)
        self.log_scalar('Evaluation/Mean_Time', eval_results.get('mean_time', 0), timestep)
        self.log_scalar('Evaluation/Collision_Rate', eval_results.get('collision_rate', 0), timestep)
        self.log_scalar('Evaluation/ATE_Error', eval_results.get('mean_ate', 0), timestep)
    
    def log_hyperparameters(self, hyperparams: Dict[str, float], timestep: int):
        """Log current hyperparameters."""
        for param_name, value in hyperparams.items():
            self.log_scalar(f'Hyperparameters/{param_name}', value, timestep)
    
    def log_network_weights(self, network: torch.nn.Module, step: Optional[int] = None):
        """Log network weight distributions."""
        step = step if step is not None else self.global_step
        
        for name, param in network.named_parameters():
            if param.requires_grad:
                self.log_histogram(f'Weights/{name}', param.data, step)
                if param.grad is not None:
                    self.log_histogram(f'Gradients/{name}', param.grad.data, step)
    
    def log_action_distribution(self, actions: np.ndarray, timestep: int):
        """Log action distribution analysis."""
        if len(actions.shape) != 2 or actions.shape[1] != 4:
            return
        
        action_names = ['velocity_x', 'velocity_y', 'velocity_z', 'yaw_rate']
        
        for i, name in enumerate(action_names):
            action_component = actions[:, i]
            self.log_histogram(f'Actions/{name}', action_component, timestep)
            self.log_scalar(f'Actions/{name}_mean', np.mean(action_component), timestep)
            self.log_scalar(f'Actions/{name}_std', np.std(action_component), timestep)
        
        # Overall action magnitude
        action_magnitudes = np.linalg.norm(actions[:, :3], axis=1)  # Exclude yaw
        self.log_histogram('Actions/Magnitude', action_magnitudes, timestep)
        self.log_scalar('Actions/Mean_Magnitude', np.mean(action_magnitudes), timestep)
    
    def log_phase_transition(self, phase_name: str, phase_results: Dict[str, Any], timestep: int):
        """Log curriculum phase transition."""
        # Phase completion metrics
        self.log_scalar(f'Curriculum/{phase_name}_Success_Rate', phase_results.get('final_success_rate', 0), timestep)
        self.log_scalar(f'Curriculum/{phase_name}_Energy', phase_results.get('final_energy', 0), timestep)
        self.log_scalar(f'Curriculum/{phase_name}_Training_Time', phase_results.get('training_time', 0), timestep)
        
        # Add text summary
        summary_text = f"""
        Phase: {phase_name}
        Success Rate: {phase_results.get('final_success_rate', 0):.1f}%
        Energy Consumption: {phase_results.get('final_energy', 0):.0f}J
        Training Time: {phase_results.get('training_time', 0)/3600:.1f}h
        Target Achieved: {phase_results.get('target_achieved', False)}
        """
        
        self.writer.add_text(f'Curriculum/{phase_name}_Summary', summary_text, timestep)
    
    def log_trajectory_analysis(self, trajectory_metrics: Dict[str, Any], timestep: int):
        """Log trajectory analysis results."""
        # Path efficiency
        if 'path_efficiency' in trajectory_metrics:
            eff = trajectory_metrics['path_efficiency']
            self.log_scalar('Trajectory/Path_Efficiency_Mean', eff.get('mean', 0), timestep)
            self.log_scalar('Trajectory/Path_Efficiency_Std', eff.get('std', 0), timestep)
        
        # Smoothness
        if 'smoothness' in trajectory_metrics:
            smooth = trajectory_metrics['smoothness']
            self.log_scalar('Trajectory/Smoothness_Score', smooth.get('mean_score', 0), timestep)
        
        # Quality assessment
        if 'quality_assessment' in trajectory_metrics:
            quality = trajectory_metrics['quality_assessment']
            self.log_scalar('Trajectory/Overall_Quality', quality.get('overall_quality_score', 0), timestep)
    
    def log_energy_analysis(self, energy_metrics: Dict[str, Any], timestep: int):
        """Log detailed energy analysis."""
        # Energy consumption
        if 'energy_consumption' in energy_metrics:
            energy = energy_metrics['energy_consumption']
            self.log_scalar('Energy/Mean_Consumption', energy.get('mean_energy', 0), timestep)
            self.log_scalar('Energy/Energy_Range', energy.get('energy_range', 0), timestep)
        
        # Energy efficiency
        if 'energy_efficiency' in energy_metrics:
            eff = energy_metrics['energy_efficiency']
            self.log_scalar('Energy/Energy_Per_Meter', eff.get('mean_energy_per_meter', 0), timestep)
            self.log_scalar('Energy/Efficiency_Target_Met', 1.0 if eff.get('meets_efficiency_target', False) else 0.0, timestep)
        
        # Power analysis
        if 'power_analysis' in energy_metrics:
            power = energy_metrics['power_analysis']
            self.log_scalar('Energy/Mean_Power', power.get('mean_power', 0), timestep)
    
    def create_comparison_plot(self, baseline_results: Dict[str, Any], rl_results: Dict[str, Any], timestep: int):
        """Create comparison visualization with baselines."""
        # Would implement matplotlib integration for custom plots
        # For now, log comparison metrics
        
        if 'A*_Only' in baseline_results and 'success_rate' in rl_results:
            baseline_success = baseline_results['A*_Only'].get('success_rate', 0)
            rl_success = rl_results['success_rate']
            improvement = ((rl_success - baseline_success) / baseline_success) * 100 if baseline_success > 0 else 0
            
            self.log_scalar('Comparison/Success_Rate_Improvement_vs_AStar', improvement, timestep)
        
        if 'A*_Only' in baseline_results and 'mean_energy' in rl_results:
            baseline_energy = baseline_results['A*_Only'].get('mean_energy', 0)
            rl_energy = rl_results['mean_energy']
            savings = ((baseline_energy - rl_energy) / baseline_energy) * 100 if baseline_energy > 0 else 0
            
            self.log_scalar('Comparison/Energy_Savings_vs_AStar', savings, timestep)
    
    def update_global_step(self, step: int):
        """Update global step counter."""
        self.global_step = step
    
    def flush(self):
        """Flush logs to disk."""
        self.writer.flush()
        self.log_counter += 1
        
        if self.log_counter % self.flush_frequency == 0:
            self.logger.debug("TensorBoard logs flushed")
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
        self.logger.info("TensorBoard logger closed")
    
    def export_metrics_json(self, filepath: str):
        """Export logged metrics to JSON."""
        export_data = {
            'experiment_name': self.experiment_name,
            'log_directory': str(self.experiment_dir),
            'total_steps': self.global_step,
            'total_episodes': self.episode_count,
            'metric_history': self.metric_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
