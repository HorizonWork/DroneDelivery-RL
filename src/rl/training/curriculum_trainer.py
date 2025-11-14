import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.rl.training.trainer import PPOTrainer, TrainingConfig, TrainingState
from src.rl.training.phase_1_trainer import Phase1Trainer
from src.rl.training.phase_2_trainer import Phase2Trainer
from src.rl.training.phase_3_trainer import Phase3Trainer
from src.rl.agents.ppo_agent import PPOAgent

class CurriculumPhase(Enum):

    PHASE_1 = "single_floor_static"
    PHASE_2 = "two_floors_dynamic"
    PHASE_3 = "five_floors_full"

dataclass
class CurriculumConfig:

    phase_1_steps: int = 1_000_000
    phase_2_steps: int = 2_000_000
    phase_3_steps: int = 2_000_000

    phase_1_success_threshold: float = 0.85
    phase_2_success_threshold: float = 0.90

    max_phase_extension: int = 500_000
    min_performance_episodes: int = 50

class CurriculumTrainer:

    def __init__(
        self, agent: PPOAgent, environments: Dict[str, Any], config: Dict[str, Any]
    ):
        self.agent = agent
        self.environments = environments
        self.config = CurriculumConfig(config.get("curriculum", {}))
        self.logger = logging.getLogger(__name__)

        self.phase_trainers = {
            CurriculumPhase.PHASE_1: Phase1Trainer(
                agent, environments["single_floor"], config
            ),
            CurriculumPhase.PHASE_2: Phase2Trainer(
                agent, environments["two_floors"], config
            ),
            CurriculumPhase.PHASE_3: Phase3Trainer(
                agent, environments["five_floors"], config
            ),
        }

        self.current_phase = CurriculumPhase.PHASE_1
        self.phase_start_timestep = 0
        self.total_timesteps = 0

        self.phase_results = {}

        self.training_config = TrainingConfig(config.get("training", {}))
        self.training_config.total_timesteps = (
            self.config.phase_1_steps
            + self.config.phase_2_steps
            + self.config.phase_3_steps
        )

        self.logger.info("Curriculum Trainer initialized")
        self.logger.info(f"Phase 1: {self.config.phase_1_steps:,} steps (single floor)")
        self.logger.info(
            f"Phase 2: {self.config.phase_2_steps:,} steps (two floors + dynamics)"
        )
        self.logger.info(
            f"Phase 3: {self.config.phase_3_steps:,} steps (five floors full)"
        )
        self.logger.info(
            f"Total training: {self.training_config.total_timesteps:,} steps"
        )

    def train_curriculum(self) - Dict[str, Any]:

        self.logger.info("Starting curriculum learning")
        curriculum_start_time = time.time()

        phase_1_result = self._execute_phase(
            CurriculumPhase.PHASE_1, self.config.phase_1_steps
        )
        self.phase_results[CurriculumPhase.PHASE_1] = phase_1_result

        if not self._phase_completion_check(CurriculumPhase.PHASE_1, phase_1_result):
            return self._handle_phase_failure(CurriculumPhase.PHASE_1)

        self.current_phase = CurriculumPhase.PHASE_2
        phase_2_result = self._execute_phase(
            CurriculumPhase.PHASE_2, self.config.phase_2_steps
        )
        self.phase_results[CurriculumPhase.PHASE_2] = phase_2_result

        if not self._phase_completion_check(CurriculumPhase.PHASE_2, phase_2_result):
            return self._handle_phase_failure(CurriculumPhase.PHASE_2)

        self.current_phase = CurriculumPhase.PHASE_3
        phase_3_result = self._execute_phase(
            CurriculumPhase.PHASE_3, self.config.phase_3_steps
        )
        self.phase_results[CurriculumPhase.PHASE_3] = phase_3_result

        curriculum_time = time.time() - curriculum_start_time

        curriculum_results = {
            "curriculum_completed": True,
            "total_training_time": curriculum_time,
            "total_timesteps": self.total_timesteps,
            "phase_results": {
                phase.value: result for phase, result in self.phase_results.items()
            },
            "final_performance": phase_3_result,
            "curriculum_success": self._evaluate_curriculum_success(),
        }

        self.logger.info(
            f"Curriculum learning completed in {curriculum_time/3600:.1f} hours"
        )
        self._log_curriculum_summary(curriculum_results)

        return curriculum_results

    def _execute_phase(
        self, phase: CurriculumPhase, phase_steps: int
    ) - Dict[str, Any]:

        self.logger.info(f"Starting {phase.value} - {phase_steps:,} steps")
        phase_start_time = time.time()
        self.phase_start_timestep = self.total_timesteps

        phase_trainer = self.phase_trainers[phase]

        phase_trainer.set_phase_steps(phase_steps)
        phase_trainer.set_starting_timestep(self.total_timesteps)

        phase_result = phase_trainer.train_phase()

        self.total_timesteps += phase_result.get("timesteps_trained", phase_steps)

        phase_time = time.time() - phase_start_time

        self.logger.info(f"{phase.value} completed in {phase_time/3600:.1f} hours")
        self.logger.info(
            f"  Final success rate: {phase_result.get('final_success_rate', 0):.1f}"
        )
        self.logger.info(f"  Final energy: {phase_result.get('final_energy', 0):.0f}J")

        return phase_result

    def _phase_completion_check(
        self, phase: CurriculumPhase, phase_result: Dict[str, Any]
    ) - bool:

        success_rate = (
            phase_result.get("final_success_rate", 0.0) / 100
        )

        if phase == CurriculumPhase.PHASE_1:
            threshold = self.config.phase_1_success_threshold
            criteria_met = success_rate = threshold

        elif phase == CurriculumPhase.PHASE_2:
            threshold = self.config.phase_2_success_threshold
            criteria_met = success_rate = threshold

        else:
            threshold = self.training_config.target_success_rate
            criteria_met = success_rate = threshold

        if criteria_met:
            self.logger.info(
                f"{phase.value} completion criteria met: {success_rate100:.1f}  {threshold100:.1f}"
            )
        else:
            self.logger.warning(
                f"{phase.value} completion criteria NOT met: {success_rate100:.1f}  {threshold100:.1f}"
            )

        return criteria_met

    def _handle_phase_failure(self, failed_phase: CurriculumPhase) - Dict[str, Any]:

        self.logger.error(f"Phase {failed_phase.value} failed completion criteria")

        return {
            "curriculum_completed": False,
            "failed_at_phase": failed_phase.value,
            "phase_results": {
                phase.value: result for phase, result in self.phase_results.items()
            },
            "failure_reason": f"Failed to meet {failed_phase.value} completion criteria",
        }

    def _evaluate_curriculum_success(self) - bool:

        if CurriculumPhase.PHASE_3 not in self.phase_results:
            return False

        final_results = self.phase_results[CurriculumPhase.PHASE_3]

        success_rate = final_results.get("final_success_rate", 0.0)
        energy_efficiency = final_results.get("energy_efficiency_improvement", 0.0)

        targets_met = (
            success_rate = 96.0
            and energy_efficiency = 25.0
        )

        return targets_met

    def _log_curriculum_summary(self, curriculum_results: Dict[str, Any]):

        self.logger.info("\nCURRICULUM LEARNING SUMMARY")
        self.logger.info("="  50)

        for phase_name, phase_result in curriculum_results["phase_results"].items():
            self.logger.info(f"\n{phase_name.upper()}:")
            self.logger.info(
                f"  Success Rate: {phase_result.get('final_success_rate', 0):.1f}"
            )
            self.logger.info(f"  Energy: {phase_result.get('final_energy', 0):.0f}J")
            self.logger.info(
                f"  Training Time: {phase_result.get('training_time', 0)/3600:.1f}h"
            )

        self.logger.info(f"\nOVERALL CURRICULUM:")
        self.logger.info(
            f"  Total Time: {curriculum_results['total_training_time']/3600:.1f}h"
        )
        self.logger.info(
            f"  Total Timesteps: {curriculum_results['total_timesteps']:,}"
        )
        self.logger.info(
            f"  Success: {'' if curriculum_results['curriculum_success'] else ''}"
        )

    def get_curriculum_progress(self) - Dict[str, Any]:

        return {
            "current_phase": self.current_phase.value,
            "phase_progress": {
                "current_timestep": self.total_timesteps,
                "phase_start_timestep": self.phase_start_timestep,
                "phase_progress_percent": (
                    (self.total_timesteps - self.phase_start_timestep)
                    / self._get_current_phase_steps()
                )
                 100,
            },
            "overall_progress": {
                "total_timesteps": self.total_timesteps,
                "total_target_steps": self.training_config.total_timesteps,
                "overall_progress_percent": (
                    self.total_timesteps / self.training_config.total_timesteps
                )
                 100,
            },
            "completed_phases": [phase.value for phase in self.phase_results.keys()],
            "phase_results_summary": {
                phase.value: {
                    "success_rate": result.get("final_success_rate", 0),
                    "energy": result.get("final_energy", 0),
                }
                for phase, result in self.phase_results.items()
            },
        }

    def _get_current_phase_steps(self) - int:

        if self.current_phase == CurriculumPhase.PHASE_1:
            return self.config.phase_1_steps
        elif self.current_phase == CurriculumPhase.PHASE_2:
            return self.config.phase_2_steps
        else:
            return self.config.phase_3_steps
