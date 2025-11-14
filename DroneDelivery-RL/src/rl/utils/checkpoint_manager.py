import torch
import os
import json
import shutil
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

dataclass
class CheckpointInfo:

    filepath: str
    timestep: int
    episode: int
    success_rate: float
    energy_efficiency: float
    created_at: str
    model_size_mb: float

class CheckpointManager:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.experiment_name = config.get('experiment_name', 'ppo_drone')
        self.max_checkpoints = config.get('max_checkpoints', 5)
        self.save_best_only = config.get('save_best_only', False)

        self.auto_save_frequency = config.get('auto_save_frequency', 100_000)
        self.save_on_improvement = config.get('save_on_improvement', True)

        self.best_success_rate = 0.0
        self.best_energy_efficiency = float('inf')
        self.best_combined_score = 0.0

        self.checkpoint_registry: List[CheckpointInfo] = []

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_models_dir = self.checkpoint_dir / 'best_models'
        self.best_models_dir.mkdir(exist_ok=True)

        self._load_registry()

        self.logger.info("Checkpoint Manager initialized")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.info(f"Max checkpoints: {self.max_checkpoints}")

    def save_checkpoint(self, agent, timestep: int, episode: int,
                       success_rate: float = 0.0, energy_efficiency: float = 0.0,
                       additional_data: Dict[str, Any] = None) - str:

        timestamp = datetime.now().strftime("Ymd_HMS")
        checkpoint_name = f"{self.experiment_name}_step_{timestep:08d}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint_data = {
            'agent_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'training_info': {
                'timestep': timestep,
                'episode': episode,
                'success_rate': success_rate,
                'energy_efficiency': energy_efficiency,
                'global_step': getattr(agent, 'global_step', 0),
                'total_environment_steps': getattr(agent, 'total_environment_steps', 0)
            },
            'agent_config': agent.config.__dict__,
            'checkpoint_metadata': {
                'created_at': timestamp,
                'checkpoint_name': checkpoint_name,
                'pytorch_version': torch.__version__
            }
        }

        if additional_data:
            checkpoint_data['additional_data'] = additional_data

        torch.save(checkpoint_data, checkpoint_path)

        file_size_mb = os.path.getsize(checkpoint_path) / (1024  1024)

        checkpoint_info = CheckpointInfo(
            filepath=str(checkpoint_path),
            timestep=timestep,
            episode=episode,
            success_rate=success_rate,
            energy_efficiency=energy_efficiency,
            created_at=timestamp,
            model_size_mb=file_size_mb
        )

        self.checkpoint_registry.append(checkpoint_info)
        self._update_best_models(checkpoint_info, agent)

        if not self.save_best_only:
            self._cleanup_old_checkpoints()

        self._save_registry()

        self.logger.info(f"Checkpoint saved: {checkpoint_name} ({file_size_mb:.1f}MB)")
        self.logger.info(f"Performance: {success_rate:.1f} success, {energy_efficiency:.0f}J energy")

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str, agent) - Dict[str, Any]:

        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=agent.device)

            agent.policy.load_state_dict(checkpoint_data['agent_state_dict'])
            agent.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

            training_info = checkpoint_data.get('training_info', {})

            if 'global_step' in training_info:
                agent.global_step = training_info['global_step']
            if 'total_environment_steps' in training_info:
                agent.total_environment_steps = training_info['total_environment_steps']

            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            self.logger.info(f"Restored to timestep {training_info.get('timestep', 0):,}, "
                           f"episode {training_info.get('episode', 0):,}")

            return training_info

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise

    def _update_best_models(self, checkpoint_info: CheckpointInfo, agent):

        if checkpoint_info.success_rate  self.best_success_rate:
            self.best_success_rate = checkpoint_info.success_rate
            self._save_best_model(agent, 'best_success_rate', checkpoint_info)

        if 0  checkpoint_info.energy_efficiency  self.best_energy_efficiency:
            self.best_energy_efficiency = checkpoint_info.energy_efficiency
            self._save_best_model(agent, 'best_energy_efficiency', checkpoint_info)

        combined_score = checkpoint_info.success_rate - (checkpoint_info.energy_efficiency / 1000.0)
        if combined_score  self.best_combined_score:
            self.best_combined_score = combined_score
            self._save_best_model(agent, 'best_combined', checkpoint_info)

    def _save_best_model(self, agent, model_type: str, checkpoint_info: CheckpointInfo):

        best_model_path = self.best_models_dir / f"{self.experiment_name}_{model_type}.pt"

        shutil.copy2(checkpoint_info.filepath, best_model_path)

        metadata = {
            'model_type': model_type,
            'timestep': checkpoint_info.timestep,
            'episode': checkpoint_info.episode,
            'success_rate': checkpoint_info.success_rate,
            'energy_efficiency': checkpoint_info.energy_efficiency,
            'created_at': checkpoint_info.created_at,
            'original_checkpoint': checkpoint_info.filepath
        }

        metadata_path = self.best_models_dir / f"{self.experiment_name}_{model_type}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"New best {model_type} model saved: {checkpoint_info.success_rate:.1f} success")

    def _cleanup_old_checkpoints(self):

        if len(self.checkpoint_registry) = self.max_checkpoints:
            return

        sorted_checkpoints = sorted(self.checkpoint_registry, key=lambda x: x.timestep)

        to_remove = sorted_checkpoints[:-self.max_checkpoints]

        for checkpoint_info in to_remove:
            try:
                if os.path.exists(checkpoint_info.filepath):
                    os.remove(checkpoint_info.filepath)
                self.checkpoint_registry.remove(checkpoint_info)
                self.logger.debug(f"Removed old checkpoint: {Path(checkpoint_info.filepath).name}")
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {checkpoint_info.filepath}: {e}")

    def _load_registry(self):

        registry_path = self.checkpoint_dir / 'checkpoint_registry.json'

        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)

                self.checkpoint_registry = [
                    CheckpointInfo(info) for info in registry_data.get('checkpoints', [])
                ]

                metadata = registry_data.get('metadata', {})
                self.best_success_rate = metadata.get('best_success_rate', 0.0)
                self.best_energy_efficiency = metadata.get('best_energy_efficiency', float('inf'))
                self.best_combined_score = metadata.get('best_combined_score', 0.0)

                self.logger.info(f"Loaded {len(self.checkpoint_registry)} checkpoints from registry")

            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint registry: {e}")

    def _save_registry(self):

        registry_data = {
            'checkpoints': [
                {
                    'filepath': info.filepath,
                    'timestep': info.timestep,
                    'episode': info.episode,
                    'success_rate': info.success_rate,
                    'energy_efficiency': info.energy_efficiency,
                    'created_at': info.created_at,
                    'model_size_mb': info.model_size_mb
                }
                for info in self.checkpoint_registry
            ],
            'metadata': {
                'best_success_rate': self.best_success_rate,
                'best_energy_efficiency': self.best_energy_efficiency,
                'best_combined_score': self.best_combined_score,
                'experiment_name': self.experiment_name,
                'last_updated': datetime.now().isoformat()
            }
        }

        registry_path = self.checkpoint_dir / 'checkpoint_registry.json'
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)

    def get_latest_checkpoint(self) - Optional[CheckpointInfo]:

        if not self.checkpoint_registry:
            return None

        return max(self.checkpoint_registry, key=lambda x: x.timestep)

    def get_best_checkpoint(self, criteria: str = 'combined') - Optional[CheckpointInfo]:

        if not self.checkpoint_registry:
            return None

        if criteria == 'success_rate':
            return max(self.checkpoint_registry, key=lambda x: x.success_rate)
        elif criteria == 'energy_efficiency':
            return min(self.checkpoint_registry, key=lambda x: x.energy_efficiency if x.energy_efficiency  0 else float('inf'))
        else:
            def combined_score(info):
                return info.success_rate - (info.energy_efficiency / 1000.0)
            return max(self.checkpoint_registry, key=combined_score)

    def list_checkpoints(self) - List[Dict[str, Any]]:

        return [
            {
                'name': Path(info.filepath).name,
                'timestep': info.timestep,
                'episode': info.episode,
                'success_rate': info.success_rate,
                'energy_efficiency': info.energy_efficiency,
                'size_mb': info.model_size_mb,
                'age_hours': (datetime.now() - datetime.fromisoformat(info.created_at.replace('_', 'T'))).total_seconds() / 3600
            }
            for info in sorted(self.checkpoint_registry, key=lambda x: x.timestep, reverse=True)
        ]
