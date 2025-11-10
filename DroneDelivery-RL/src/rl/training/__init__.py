"""
Training Module
PPO training with curriculum learning and multi-phase approach.
Implements exact training procedure from Section 5.2.
"""

from src.rl.training.trainer import PPOTrainer, TrainingState, TrainingConfig
from src.rl.training.curriculum_trainer import CurriculumTrainer
from src.rl.training.hyperparameter_scheduler import HyperparameterScheduler
from src.rl.training.phase_1_trainer import Phase1Trainer
from src.rl.training.phase_2_trainer import Phase2Trainer
from src.rl.training.phase_3_trainer import Phase3Trainer

__all__ = [
    'PPOTrainer',
    'TrainingState',
    'TrainingConfig',
    'CurriculumTrainer',
    'HyperparameterScheduler',
    'Phase1Trainer',
    'Phase2Trainer',
    'Phase3Trainer'
]
