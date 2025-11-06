"""
Training Module
PPO training with curriculum learning and multi-phase approach.
Implements exact training procedure from Section 5.2.
"""

from .trainer import PPOTrainer
from .curriculum_trainer import CurriculumTrainer
from .hyperparameter_scheduler import HyperparameterScheduler
from .phase_1_trainer import Phase1Trainer
from .phase_2_trainer import Phase2Trainer  
from .phase_3_trainer import Phase3Trainer

__all__ = [
    'PPOTrainer',
    'CurriculumTrainer', 
    'HyperparameterScheduler',
    'Phase1Trainer',
    'Phase2Trainer',
    'Phase3Trainer'
]
