import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class CurriculumPhase(Enum):

    PHASE_1 = 1
    PHASE_2 = 2
    PHASE_3 = 3

dataclass
class PhaseConfig:

    phase: CurriculumPhase
    name: str
    timesteps: int
    active_floors: int
    dynamic_obstacles: bool
    difficulty: str
    success_threshold: float
    min_episodes: int

dataclass
class PhaseProgress:

    episodes_completed: int = 0
    timesteps_completed: int = 0
    success_count: int = 0
    collision_count: int = 0
    success_rate: float = 0.0
    ready_for_next: bool = False
    phase_start_time: float = 0.0

class CurriculumManager:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.total_timesteps = config.get('total_timesteps', 5000000)
        self.enabled = config.get('curriculum_enabled', True)

        self.phases = {
            CurriculumPhase.PHASE_1: PhaseConfig(
                phase=CurriculumPhase.PHASE_1,
                name="single_floor_static",
                timesteps=1000000,
                active_floors=1,
                dynamic_obstacles=False,
                difficulty="easy",
                success_threshold=0.85,
                min_episodes=1000
            ),
            CurriculumPhase.PHASE_2: PhaseConfig(
                phase=CurriculumPhase.PHASE_2,
                name="two_floors_dynamic",
                timesteps=2000000,
                active_floors=2,
                dynamic_obstacles=True,
                difficulty="medium",
                success_threshold=0.90,
                min_episodes=2000
            ),
            CurriculumPhase.PHASE_3: PhaseConfig(
                phase=CurriculumPhase.PHASE_3,
                name="five_floors_full",
                timesteps=2000000,
                active_floors=5,
                dynamic_obstacles=True,
                difficulty="hard",
                success_threshold=0.96,
                min_episodes=3000
            )
        }

        self.current_phase = CurriculumPhase.PHASE_1
        self.progress = {phase: PhaseProgress() for phase in CurriculumPhase}
        self.total_timesteps_completed = 0

        self.evaluation_episodes = config.get('evaluation_episodes', 100)
        self.evaluation_frequency = config.get('evaluation_frequency', 10000)
        self.patience = config.get('patience', 5)

        self.recent_episodes: List[Dict[str, Any]] = []
        self.max_recent_episodes = 200

        self.progress[self.current_phase].phase_start_time = time.time()

        self.logger.info("Curriculum Manager initialized")
        if self.enabled:
            self.logger.info("3-phase curriculum enabled:")
            for phase, config in self.phases.items():
                self.logger.info(f"  {config.name}: {config.timesteps/1e6:.1f}M timesteps, "
                               f"{config.active_floors} floors, {config.difficulty}")
        else:
            self.logger.info("Curriculum learning disabled - using full environment")

    def get_current_phase(self) - int:

        if not self.enabled:
            return 3

        return self.current_phase.value

    def get_current_config(self) - PhaseConfig:

        if not self.enabled:
            return self.phases[CurriculumPhase.PHASE_3]

        return self.phases[self.current_phase]

    def update_progress(self, success: bool, collision: bool,
                       timesteps_this_episode: int = 1):

        if not self.enabled:
            return

        current_progress = self.progress[self.current_phase]

        current_progress.episodes_completed += 1
        current_progress.timesteps_completed += timesteps_this_episode
        self.total_timesteps_completed += timesteps_this_episode

        if success:
            current_progress.success_count += 1
        if collision:
            current_progress.collision_count += 1

        if current_progress.episodes_completed  0:
            current_progress.success_rate = (
                current_progress.success_count / current_progress.episodes_completed
            )

        episode_info = {
            'phase': self.current_phase.value,
            'success': success,
            'collision': collision,
            'timesteps': timesteps_this_episode,
            'episode_number': current_progress.episodes_completed
        }
        self.recent_episodes.append(episode_info)

        if len(self.recent_episodes)  self.max_recent_episodes:
            self.recent_episodes.pop(0)

        self._check_phase_progression()

        if current_progress.episodes_completed  100 == 0:
            self._log_progress()

    def _check_phase_progression(self):

        if not self.enabled or self.current_phase == CurriculumPhase.PHASE_3:
            return

        current_progress = self.progress[self.current_phase]
        phase_config = self.phases[self.current_phase]

        min_episodes_met = current_progress.episodes_completed = phase_config.min_episodes
        min_timesteps_met = current_progress.timesteps_completed = phase_config.timesteps

        recent_success_rate = self._calculate_recent_success_rate()
        success_threshold_met = recent_success_rate = phase_config.success_threshold

        if min_episodes_met and min_timesteps_met and success_threshold_met:
            if not current_progress.ready_for_next:
                current_progress.ready_for_next = True
                self.logger.info(f"Phase {self.current_phase.value} progression criteria met!")
                self.logger.info(f"  Episodes: {current_progress.episodes_completed}/{phase_config.min_episodes}")
                self.logger.info(f"  Timesteps: {current_progress.timesteps_completed}/{phase_config.timesteps}")
                self.logger.info(f"  Success rate: {recent_success_rate:.3f}/{phase_config.success_threshold}")

                self._advance_to_next_phase()

    def _calculate_recent_success_rate(self, window_size: int = 100) - float:

        if not self.recent_episodes:
            return 0.0

        recent_current_phase = [
            ep for ep in self.recent_episodes[-window_size:]
            if ep['phase'] == self.current_phase.value
        ]

        if not recent_current_phase:
            return 0.0

        success_count = sum(1 for ep in recent_current_phase if ep['success'])
        return success_count / len(recent_current_phase)

    def _advance_to_next_phase(self):

        if self.current_phase == CurriculumPhase.PHASE_1:
            next_phase = CurriculumPhase.PHASE_2
        elif self.current_phase == CurriculumPhase.PHASE_2:
            next_phase = CurriculumPhase.PHASE_3
        else:
            return

        current_config = self.phases[self.current_phase]
        next_config = self.phases[next_phase]

        self.logger.info("="60)
        self.logger.info(f"CURRICULUM PHASE PROGRESSION: {self.current_phase.value}  {next_phase.value}")
        self.logger.info(f"  From: {current_config.name}")
        self.logger.info(f"  To: {next_config.name}")
        self.logger.info(f"  Floors: {current_config.active_floors}  {next_config.active_floors}")
        self.logger.info(f"  Difficulty: {current_config.difficulty}  {next_config.difficulty}")
        self.logger.info("="60)

        self.current_phase = next_phase
        self.progress[next_phase].phase_start_time = time.time()

    def _log_progress(self):

        current_progress = self.progress[self.current_phase]
        phase_config = self.phases[self.current_phase]

        episode_progress = min(100, 100  current_progress.episodes_completed / phase_config.min_episodes)
        timestep_progress = min(100, 100  current_progress.timesteps_completed / phase_config.timesteps)

        self.logger.info(f"Phase {self.current_phase.value} progress:")
        self.logger.info(f"  Episodes: {current_progress.episodes_completed}/{phase_config.min_episodes} ({episode_progress:.1f})")
        self.logger.info(f"  Timesteps: {current_progress.timesteps_completed}/{phase_config.timesteps} ({timestep_progress:.1f})")
        self.logger.info(f"  Success rate: {current_progress.success_rate:.3f} (target: {phase_config.success_threshold})")
        self.logger.info(f"  Recent success rate: {self._calculate_recent_success_rate():.3f}")

    def get_environment_config(self) - Dict[str, Any]:

        phase_config = self.get_current_config()

        return {
            'active_floors': phase_config.active_floors,
            'dynamic_obstacles': phase_config.dynamic_obstacles,
            'difficulty': phase_config.difficulty,
            'phase_name': phase_config.name,
            'success_threshold': phase_config.success_threshold
        }

    def get_progress_info(self) - Dict[str, Any]:

        current_progress = self.progress[self.current_phase]
        phase_config = self.phases[self.current_phase]

        overall_progress = self.total_timesteps_completed / self.total_timesteps

        phase_time = time.time() - current_progress.phase_start_time

        info = {
            'curriculum_enabled': self.enabled,
            'total_progress': {
                'timesteps_completed': self.total_timesteps_completed,
                'total_timesteps': self.total_timesteps,
                'completion_percentage': min(100, 100  overall_progress)
            },
            'current_phase': {
                'phase_number': self.current_phase.value,
                'phase_name': phase_config.name,
                'episodes_completed': current_progress.episodes_completed,
                'timesteps_completed': current_progress.timesteps_completed,
                'success_rate': current_progress.success_rate,
                'recent_success_rate': self._calculate_recent_success_rate(),
                'ready_for_next': current_progress.ready_for_next,
                'phase_duration': phase_time
            },
            'phase_configs': {
                phase.value: {
                    'name': config.name,
                    'timesteps': config.timesteps,
                    'active_floors': config.active_floors,
                    'dynamic_obstacles': config.dynamic_obstacles,
                    'difficulty': config.difficulty,
                    'success_threshold': config.success_threshold
                }
                for phase, config in self.phases.items()
            },
            'all_phases_progress': {
                phase.value: {
                    'episodes_completed': progress.episodes_completed,
                    'timesteps_completed': progress.timesteps_completed,
                    'success_rate': progress.success_rate,
                    'ready_for_next': progress.ready_for_next
                }
                for phase, progress in self.progress.items()
            }
        }

        return info

    def force_phase_transition(self, target_phase: int):

        if not self.enabled:
            self.logger.warning("Cannot force phase transition - curriculum disabled")
            return

        if target_phase not in [1, 2, 3]:
            self.logger.error(f"Invalid phase number: {target_phase}")
            return

        target = CurriculumPhase(target_phase)
        if target != self.current_phase:
            self.logger.warning(f"Force transitioning from phase {self.current_phase.value} to {target_phase}")
            self.current_phase = target
            self.progress[target].phase_start_time = time.time()

    def reset_curriculum(self):

        self.current_phase = CurriculumPhase.PHASE_1
        self.progress = {phase: PhaseProgress() for phase in CurriculumPhase}
        self.total_timesteps_completed = 0
        self.recent_episodes.clear()
        self.progress[self.current_phase].phase_start_time = time.time()

        self.logger.info("Curriculum reset to Phase 1")

    def is_curriculum_complete(self) - bool:

        if not self.enabled:
            return False

        return (self.current_phase == CurriculumPhase.PHASE_3 and
                self.progress[CurriculumPhase.PHASE_3].ready_for_next)

    def get_estimated_time_remaining(self) - Optional[float]:

        current_progress = self.progress[self.current_phase]
        phase_config = self.phases[self.current_phase]

        if current_progress.timesteps_completed == 0:
            return None

        phase_duration = time.time() - current_progress.phase_start_time
        timesteps_per_second = current_progress.timesteps_completed / phase_duration

        remaining_timesteps = max(0, phase_config.timesteps - current_progress.timesteps_completed)
        estimated_seconds = remaining_timesteps / timesteps_per_second if timesteps_per_second  0 else None

        return estimated_seconds
