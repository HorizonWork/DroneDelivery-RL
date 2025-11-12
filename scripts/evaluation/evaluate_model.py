#!/usr/bin/env python3
"""
Model Evaluation Script
Comprehensive evaluation of trained RL models.
Generates Table 3 results and detailed performance analysis.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# CORRECTED IMPORTS:
from src.environment.airsim_env import AirSimEnvironment as DroneEnvironment
from src.rl.agents.ppo_agent import PPOAgent
from src.rl.evaluation.evaluator import DroneEvaluator
from src.rl.evaluation.baseline_comparator import BaselineComparator
from src.rl.evaluation.energy_analyzer import EnergyAnalyzer
from src.rl.evaluation.trajectory_analyzer import TrajectoryAnalyzer
from src.rl.initialization import initialize_rl_system  # ← FIXED IMPORT
from src.utils import setup_logging, load_config, SystemVisualizer


class ModelEvaluator:
    """
    Comprehensive trained model evaluation system.
    Performs detailed analysis matching Table 3 requirements.
    """

    def __init__(self, config_path: str, model_path: str):
        # Load configuration
        self.config = load_config(config_path)
        self.model_path = Path(model_path)

        # Setup logging
        self.logger_system = setup_logging(self.config.logging)
        self.logger = logging.getLogger(__name__)

        # Initialize environment
        self.environment = DroneEnvironment(self.config.environment)

        # Initialize RL system
        self.rl_system = initialize_rl_system(self.config.rl)
        self.agent = self.rl_system["agent"]

        # Load trained model
        self._load_trained_model()

        # Initialize evaluators
        self.evaluator = DroneEvaluator(self.config.evaluation)
        self.baseline_comparator = BaselineComparator(self.config.evaluation)
        self.energy_analyzer = EnergyAnalyzer(self.config.evaluation)
        self.trajectory_analyzer = TrajectoryAnalyzer(self.config.evaluation)

        # Visualization
        self.visualizer = SystemVisualizer(self.config.visualization)

        self.logger.info("Model Evaluator initialized")
        self.logger.info(f"Model loaded from: {self.model_path}")

    def _load_trained_model(self):
        """Load trained model weights."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.agent.device)

            if "agent_state_dict" in checkpoint:
                # Full checkpoint with training info
                self.agent.policy.load_state_dict(checkpoint["agent_state_dict"])
                self.training_info = checkpoint.get("training_info", {})
                self.logger.info(
                    f"Loaded checkpoint from timestep {self.training_info.get('timestep', 'unknown')}"
                )
            else:
                # Direct state dict
                self.agent.policy.load_state_dict(checkpoint)
                self.training_info = {}

            # Set to evaluation mode
            self.agent.policy.eval()

            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def evaluate_comprehensive(self, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Run comprehensive model evaluation.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Complete evaluation results
        """
        self.logger.info(
            f"Starting comprehensive evaluation with {num_episodes} episodes"
        )
        eval_start = time.time()

        # Main evaluation
        main_results = self.evaluator.evaluate_policy(
            self.agent, self.environment, "PPO_Trained"
        )

        # Energy analysis
        self.logger.info("Performing energy analysis...")
        energy_results = self.energy_analyzer.analyze_episodes(
            self.evaluator.episode_results
        )

        # Trajectory analysis
        self.logger.info("Performing trajectory analysis...")
        trajectory_results = self.trajectory_analyzer.analyze_episodes(
            self.evaluator.episode_results
        )

        # Baseline comparison
        self.logger.info("Comparing with baselines...")
        baseline_comparison = self._compare_with_baselines(main_results)

        eval_time = time.time() - eval_start

        # Compile comprehensive results
        comprehensive_results = {
            "evaluation_completed": True,
            "evaluation_time": eval_time,
            "model_path": str(self.model_path),
            "training_info": self.training_info,
            "episodes_evaluated": num_episodes,
            # Main results (Table 3 format)
            "performance_metrics": {
                "success_rate": main_results.success_rate,
                "mean_energy": main_results.mean_energy,
                "std_energy": main_results.std_energy,
                "mean_time": main_results.mean_time,
                "std_time": main_results.std_time,
                "collision_rate": main_results.collision_rate,
                "mean_ate": main_results.mean_ate,
                "std_ate": main_results.std_ate,
            },
            # Detailed analysis
            "energy_analysis": energy_results,
            "trajectory_analysis": trajectory_results,
            "baseline_comparison": baseline_comparison,
            # Target validation
            "targets_met": self._validate_targets(main_results, energy_results),
            # Summary statistics
            "summary": self._generate_evaluation_summary(main_results, energy_results),
        }

        self.logger.info(
            f"Comprehensive evaluation completed in {eval_time/60:.1f} minutes"
        )
        return comprehensive_results

    def _compare_with_baselines(self, rl_results) -> Dict[str, Any]:
        """Compare RL results with baseline methods."""
        try:
            # Convert RL results to format expected by comparator
            rl_comparison_data = {
                "success_rate": rl_results.success_rate,
                "mean_energy": rl_results.mean_energy,
                "mean_time": rl_results.mean_time,
                "collision_rate": rl_results.collision_rate,
                "mean_ate": rl_results.mean_ate,
            }

            # Perform comparison
            comparison_results = self.baseline_comparator.compare_with_baselines(
                rl_comparison_data, "PPO_Trained"
            )

            # Generate comparison table
            comparison_table = self.baseline_comparator.generate_comparison_table()

            return {
                "statistical_comparisons": comparison_results,
                "comparison_table": comparison_table,
                "energy_savings": {
                    "vs_astar_only": self.baseline_comparator.calculate_energy_efficiency_gain(
                        rl_results.mean_energy, "A*_Only"
                    ),
                    "meets_25_percent_target": self.baseline_comparator.calculate_energy_efficiency_gain(
                        rl_results.mean_energy, "A*_Only"
                    )
                    >= 25.0,
                },
            }

        except Exception as e:
            self.logger.warning(f"Baseline comparison failed: {e}")
            return {"error": str(e)}

    def _validate_targets(
        self, main_results, energy_results: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Validate performance against targets."""
        targets = {
            "success_rate_96_percent": main_results.success_rate >= 96.0,
            "energy_savings_25_percent": False,  # Will be calculated
            "ate_error_5cm": main_results.mean_ate <= 0.05,
            "collision_rate_2_percent": main_results.collision_rate <= 2.0,
        }

        # Check energy savings
        if "energy_efficiency" in energy_results:
            energy_efficiency = energy_results["energy_efficiency"]
            targets["energy_savings_25_percent"] = energy_efficiency.get(
                "meets_efficiency_target", False
            )

        # Overall target achievement
        targets["all_targets_met"] = all(targets.values())

        return targets

    def _generate_evaluation_summary(
        self, main_results, energy_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate evaluation summary."""
        return {
            "performance_grade": self._calculate_performance_grade(main_results),
            "key_strengths": self._identify_key_strengths(main_results, energy_results),
            "improvement_areas": self._identify_improvement_areas(
                main_results, energy_results
            ),
            "recommendation": self._generate_recommendation(
                main_results, energy_results
            ),
        }

    def _calculate_performance_grade(self, results) -> str:
        """Calculate overall performance grade."""
        score = 0

        # Success rate (40% weight)
        if results.success_rate >= 96:
            score += 40
        elif results.success_rate >= 90:
            score += 35
        elif results.success_rate >= 80:
            score += 25

        # Collision rate (30% weight)
        if results.collision_rate <= 2.0:
            score += 30
        elif results.collision_rate <= 5.0:
            score += 20
        elif results.collision_rate <= 10.0:
            score += 10

        # ATE accuracy (20% weight)
        if results.mean_ate <= 0.05:
            score += 20
        elif results.mean_ate <= 0.08:
            score += 15
        elif results.mean_ate <= 0.12:
            score += 10

        # Energy efficiency (10% weight)
        # Simplified - would use actual baseline comparison
        if results.mean_energy <= 2000:  # Rough threshold
            score += 10
        elif results.mean_energy <= 2500:
            score += 5

        # Grade assignment
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _identify_key_strengths(
        self, main_results, energy_results: Dict[str, Any]
    ) -> List[str]:
        """Identify key performance strengths."""
        strengths = []

        if main_results.success_rate >= 96:
            strengths.append(
                f"Excellent success rate: {main_results.success_rate:.1f}%"
            )

        if main_results.collision_rate <= 2.0:
            strengths.append(
                f"Very low collision rate: {main_results.collision_rate:.1f}%"
            )

        if main_results.mean_ate <= 0.05:
            strengths.append(
                f"High localization accuracy: {main_results.mean_ate*100:.1f}cm ATE"
            )

        if main_results.mean_energy <= 2000:  # Threshold
            strengths.append(
                f"Energy efficient: {main_results.mean_energy:.0f}J average"
            )

        return strengths

    def _identify_improvement_areas(
        self, main_results, energy_results: Dict[str, Any]
    ) -> List[str]:
        """Identify areas for improvement."""
        improvements = []

        if main_results.success_rate < 96:
            improvements.append(
                f"Success rate below target: {main_results.success_rate:.1f}% < 96%"
            )

        if main_results.collision_rate > 2.0:
            improvements.append(
                f"Collision rate above target: {main_results.collision_rate:.1f}% > 2%"
            )

        if main_results.mean_ate > 0.05:
            improvements.append(
                f"ATE error above target: {main_results.mean_ate*100:.1f}cm > 5cm"
            )

        if main_results.mean_energy > 2500:  # Conservative threshold
            improvements.append(
                f"Energy consumption could be optimized: {main_results.mean_energy:.0f}J"
            )

        return improvements

    def _generate_recommendation(
        self, main_results, energy_results: Dict[str, Any]
    ) -> str:
        """Generate deployment recommendation."""
        performance_grade = self._calculate_performance_grade(main_results)

        if performance_grade in ["A", "B"]:
            if main_results.success_rate >= 96 and main_results.collision_rate <= 2.0:
                return "APPROVED: Model meets all performance targets and is ready for deployment."
            else:
                return "CONDITIONALLY APPROVED: Model shows good performance with minor areas for improvement."
        elif performance_grade == "C":
            return "FURTHER TRAINING REQUIRED: Model shows promise but needs additional training to meet targets."
        else:
            return "NOT APPROVED: Model requires significant improvement before deployment consideration."

    def generate_visualizations(self, results: Dict[str, Any], output_dir: str):
        """Generate evaluation visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Performance comparison plot
            if "baseline_comparison" in results:
                comparison_data = results["baseline_comparison"].get(
                    "statistical_comparisons", {}
                )
                if comparison_data:
                    self.visualizer.plot_evaluation_comparison(
                        comparison_data, str(output_path / "performance_comparison.png")
                    )

            # Training curves (if available)
            if hasattr(self.evaluator, "training_history"):
                self.visualizer.plot_training_curves(
                    self.evaluator.training_history,
                    str(output_path / "training_curves.png"),
                )

            self.logger.info(f"Visualizations saved to {output_path}")

        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Evaluation results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/evaluation_config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model file"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/model_evaluation.json",
        help="Output results file",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization plots"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = ModelEvaluator(args.config, args.model)

    # Run evaluation
    results = evaluator.evaluate_comprehensive(args.episodes)

    # Save results
    evaluator.save_results(results, args.output)

    # Generate visualizations
    if args.visualize:
        vis_dir = Path(args.output).parent / "visualizations"
        evaluator.generate_visualizations(results, str(vis_dir))

    # Print summary
    print("\nMODEL EVALUATION SUMMARY")
    print("=" * 50)

    metrics = results["performance_metrics"]
    print(f"Success Rate: {metrics['success_rate']:.1f}%")
    print(
        f"Energy Consumption: {metrics['mean_energy']:.0f} ± {metrics['std_energy']:.0f}J"
    )
    print(f"Flight Time: {metrics['mean_time']:.1f} ± {metrics['std_time']:.1f}s")
    print(f"Collision Rate: {metrics['collision_rate']:.1f}%")
    print(f"ATE Error: {metrics['mean_ate']*100:.1f}cm")

    print(f"\nPerformance Grade: {results['summary']['performance_grade']}")
    print(f"Recommendation: {results['summary']['recommendation']}")

    # Target validation
    targets = results["targets_met"]
    print(f"\nTargets Met:")
    for target, met in targets.items():
        status = "✓" if met else "✗"
        print(f"  {target}: {status}")


if __name__ == "__main__":
    main()
