#!/usr/bin/env python3
"""
Energy Analysis Utility
Advanced energy consumption analysis for drone delivery system.
Analyzes energy patterns, efficiency, and optimization opportunities.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.utils.logger import setup_logging
from src.utils.visualization import SystemVisualizer


class EnergyAnalyzer:
    """
    Comprehensive energy consumption analyzer.
    Analyzes patterns, trends, and optimization opportunities.
    """

    def __init__(self, config_path: str = None):
        # Setup logging
        if config_path:
            from src.utils import load_config

            config = load_config(config_path)
            self.logger_system = setup_logging(config.logging)
        else:
            logging.basicConfig(level=logging.INFO)

        self.logger = logging.getLogger(__name__)

        # Energy analysis configuration
        self.analysis_config = {
            "energy_components": {
                "thrust": 0.70,  # 70% thrust/propulsion
                "avionics": 0.20,  # 20% computing/sensors
                "communication": 0.05,  # 5% radio/telemetry
                "other": 0.05,  # 5% other systems
            },
            "baseline_energy": {
                "A_star_only": 2800,  # From research paper
                "RRT_PID": 2400,
                "Random": 3500,
            },
            "efficiency_targets": {
                "excellent": 500,  # <500J excellent
                "good": 700,  # 500-700J good
                "acceptable": 1000,  # 700-1000J acceptable
                "poor": 2000,  # >2000J poor
            },
        }

        # Visualization setup
        self.visualizer = SystemVisualizer(
            {
                "save_plots": True,
                "figure_size": (12, 8),
                "output_dir": "results/energy_analysis",
            }
        )

        self.logger.info("Energy Analyzer initialized")

    def analyze_training_energy(self, training_results_file: str) -> Dict[str, Any]:
        """
        Analyze energy consumption during training.

        Args:
            training_results_file: Path to training results JSON

        Returns:
            Energy analysis results
        """
        self.logger.info("Analyzing training energy consumption")

        # Load training data
        with open(training_results_file, "r") as f:
            training_data = json.load(f)

        # Extract energy data
        energy_history = training_data.get("training_history", {}).get(
            "episode_energies", []
        )

        if not energy_history:
            self.logger.warning("No energy data found in training results")
            return {"error": "No energy data available"}

        # Phase-wise analysis
        phase_analysis = self._analyze_phase_energy_progression(training_data)

        # Efficiency trends
        efficiency_trends = self._calculate_efficiency_trends(energy_history)

        # Energy distribution analysis
        distribution_analysis = self._analyze_energy_distribution(energy_history)

        # Baseline comparison
        baseline_comparison = self._compare_with_baselines(energy_history)

        # Optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(
            energy_history
        )

        analysis_results = {
            "analysis_completed": True,
            "total_episodes_analyzed": len(energy_history),
            "phase_analysis": phase_analysis,
            "efficiency_trends": efficiency_trends,
            "distribution_analysis": distribution_analysis,
            "baseline_comparison": baseline_comparison,
            "optimization_opportunities": optimization_opportunities,
            "summary": self._generate_energy_summary(energy_history),
        }

        self.logger.info("Training energy analysis completed")
        return analysis_results

    def analyze_evaluation_energy(self, evaluation_results_file: str) -> Dict[str, Any]:
        """
        Analyze energy consumption from evaluation results.

        Args:
            evaluation_results_file: Path to evaluation results JSON

        Returns:
            Evaluation energy analysis
        """
        self.logger.info("Analyzing evaluation energy consumption")

        # Load evaluation data
        with open(evaluation_results_file, "r") as f:
            eval_data = json.load(f)

        # Extract performance metrics
        performance_metrics = eval_data.get("performance_metrics", {})
        mean_energy = performance_metrics.get("mean_energy", 0)
        std_energy = performance_metrics.get("std_energy", 0)

        # Calculate efficiency metrics
        efficiency_analysis = {
            "mean_energy_consumption": mean_energy,
            "energy_std_deviation": std_energy,
            "energy_efficiency_grade": self._grade_energy_efficiency(mean_energy),
            "energy_consistency": self._calculate_energy_consistency(
                std_energy, mean_energy
            ),
            "baseline_comparison": self._calculate_energy_savings(mean_energy),
            "component_breakdown": self._estimate_energy_components(mean_energy),
            "battery_life_impact": self._calculate_battery_life_impact(mean_energy),
        }

        # Statistical analysis
        statistical_analysis = self._perform_energy_statistical_analysis(eval_data)

        evaluation_energy_analysis = {
            "analysis_type": "evaluation_energy",
            "efficiency_analysis": efficiency_analysis,
            "statistical_analysis": statistical_analysis,
            "recommendations": self._generate_energy_recommendations(
                mean_energy, std_energy
            ),
        }

        self.logger.info(
            f"Evaluation energy analysis: {mean_energy:.0f}J ± {std_energy:.0f}J"
        )
        return evaluation_energy_analysis

    def _analyze_phase_energy_progression(
        self, training_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze energy progression across curriculum phases."""
        phase_performance = training_data.get("phase_performance_summary", {})
        energy_history = training_data.get("training_history", {}).get(
            "episode_energies", []
        )

        if not phase_performance or not energy_history:
            return {"error": "Insufficient phase data"}

        # Estimate phase boundaries (simplified)
        total_episodes = len(energy_history)
        phase_boundaries = [
            int(total_episodes * 0.2),  # Phase 1: First 20%
            int(total_episodes * 0.6),  # Phase 2: Next 40%
            total_episodes,  # Phase 3: Final 40%
        ]

        phases = ["Phase_1_SingleFloor", "Phase_2_TwoFloor", "Phase_3_FiveFloor"]
        phase_analysis = {}

        start_idx = 0
        for i, (phase_name, end_idx) in enumerate(zip(phases, phase_boundaries)):
            phase_energies = energy_history[start_idx:end_idx]

            if phase_energies:
                phase_analysis[phase_name] = {
                    "episodes": len(phase_energies),
                    "mean_energy": float(np.mean(phase_energies)),
                    "std_energy": float(np.std(phase_energies)),
                    "min_energy": float(np.min(phase_energies)),
                    "max_energy": float(np.max(phase_energies)),
                    "energy_trend": (
                        "decreasing"
                        if i > 0
                        and np.mean(phase_energies)
                        < np.mean(energy_history[max(0, start_idx - 100) : start_idx])
                        else "stable"
                    ),
                    "efficiency_improvement": self._calculate_phase_efficiency_improvement(
                        phase_energies, i
                    ),
                }

            start_idx = end_idx

        return {
            "phases_analyzed": len(phases),
            "phase_details": phase_analysis,
            "overall_progression": self._analyze_overall_energy_progression(
                phase_analysis
            ),
        }

    def _calculate_efficiency_trends(
        self, energy_history: List[float]
    ) -> Dict[str, Any]:
        """Calculate energy efficiency trends over training."""
        if len(energy_history) < 100:
            return {"error": "Insufficient data for trend analysis"}

        # Moving averages
        window_sizes = [50, 100, 200]
        moving_averages = {}

        for window in window_sizes:
            if len(energy_history) >= window:
                moving_avg = []
                for i in range(window - 1, len(energy_history)):
                    avg = np.mean(energy_history[i - window + 1 : i + 1])
                    moving_avg.append(avg)
                moving_averages[f"ma_{window}"] = moving_avg

        # Trend analysis
        recent_energy = np.mean(energy_history[-100:])
        early_energy = np.mean(energy_history[:100])

        energy_improvement = (early_energy - recent_energy) / early_energy * 100

        # Learning curve analysis
        learning_curve = self._analyze_energy_learning_curve(energy_history)

        return {
            "moving_averages": moving_averages,
            "energy_improvement_percent": energy_improvement,
            "recent_mean_energy": recent_energy,
            "early_mean_energy": early_energy,
            "learning_curve": learning_curve,
            "trend_direction": (
                "improving"
                if energy_improvement > 0
                else "stable" if abs(energy_improvement) < 5 else "degrading"
            ),
        }

    def _analyze_energy_distribution(
        self, energy_history: List[float]
    ) -> Dict[str, Any]:
        """Analyze energy consumption distribution."""
        if not energy_history:
            return {"error": "No energy data"}

        energy_array = np.array(energy_history)

        # Basic statistics
        distribution_stats = {
            "mean": float(np.mean(energy_array)),
            "median": float(np.median(energy_array)),
            "std": float(np.std(energy_array)),
            "min": float(np.min(energy_array)),
            "max": float(np.max(energy_array)),
            "range": float(np.max(energy_array) - np.min(energy_array)),
        }

        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        distribution_stats["percentiles"] = {
            f"p{p}": float(np.percentile(energy_array, p)) for p in percentiles
        }

        # Distribution shape analysis
        from scipy import stats

        skewness = stats.skew(energy_array)
        kurtosis = stats.kurtosis(energy_array)

        distribution_stats["shape"] = {
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "distribution_type": self._classify_distribution_shape(skewness, kurtosis),
        }

        # Efficiency categories
        efficiency_categories = self._categorize_energy_efficiency(energy_array)

        return {
            "statistics": distribution_stats,
            "efficiency_categories": efficiency_categories,
            "consistency_score": self._calculate_energy_consistency_score(energy_array),
        }

    def _compare_with_baselines(self, energy_history: List[float]) -> Dict[str, Any]:
        """Compare energy consumption with baseline methods."""
        if not energy_history:
            return {"error": "No energy data for comparison"}

        # Current performance
        recent_energy = (
            np.mean(energy_history[-100:])
            if len(energy_history) >= 100
            else np.mean(energy_history)
        )

        # Calculate savings vs each baseline
        savings_analysis = {}
        for baseline_name, baseline_energy in self.analysis_config[
            "baseline_energy"
        ].items():
            savings_percent = (baseline_energy - recent_energy) / baseline_energy * 100
            savings_analysis[baseline_name] = {
                "baseline_energy": baseline_energy,
                "current_energy": recent_energy,
                "energy_savings_percent": savings_percent,
                "energy_savings_joules": baseline_energy - recent_energy,
                "target_met": (
                    savings_percent >= 25.0 if baseline_name == "A_star_only" else True
                ),
            }

        # Overall efficiency rating
        max_savings = max(
            s["energy_savings_percent"] for s in savings_analysis.values()
        )
        efficiency_rating = (
            "excellent"
            if max_savings >= 50
            else (
                "good" if max_savings >= 25 else "fair" if max_savings >= 10 else "poor"
            )
        )

        return {
            "baseline_comparisons": savings_analysis,
            "maximum_savings_percent": max_savings,
            "efficiency_rating": efficiency_rating,
            "primary_target_met": savings_analysis.get("A_star_only", {}).get(
                "target_met", False
            ),
        }

    def _identify_optimization_opportunities(
        self, energy_history: List[float]
    ) -> List[Dict[str, Any]]:
        """Identify energy optimization opportunities."""
        opportunities = []

        if not energy_history:
            return opportunities

        recent_energy = (
            np.mean(energy_history[-100:])
            if len(energy_history) >= 100
            else np.mean(energy_history)
        )
        energy_std = (
            np.std(energy_history[-100:])
            if len(energy_history) >= 100
            else np.std(energy_history)
        )

        # High energy consumption
        if recent_energy > self.analysis_config["efficiency_targets"]["good"]:
            opportunities.append(
                {
                    "type": "high_consumption",
                    "description": f"Energy consumption ({recent_energy:.0f}J) above optimal range",
                    "recommendation": "Optimize flight paths and reduce unnecessary maneuvers",
                    "potential_savings": "10-20%",
                }
            )

        # High energy variance
        if energy_std > (recent_energy * 0.15):  # >15% coefficient of variation
            opportunities.append(
                {
                    "type": "high_variance",
                    "description": f"High energy variance ({energy_std:.0f}J std)",
                    "recommendation": "Improve flight consistency and reduce erratic behaviors",
                    "potential_savings": "5-15%",
                }
            )

        # Component-specific opportunities
        component_analysis = self._estimate_energy_components(recent_energy)

        # Thrust optimization
        thrust_energy = component_analysis["thrust_energy"]
        if thrust_energy > 450:  # High thrust consumption
            opportunities.append(
                {
                    "type": "thrust_optimization",
                    "description": f"High thrust energy ({thrust_energy:.0f}J)",
                    "recommendation": "Optimize acceleration profiles and reduce aggressive maneuvers",
                    "potential_savings": "15-25%",
                }
            )

        # Hovering time optimization
        if recent_energy > 800:  # May indicate excessive hovering
            opportunities.append(
                {
                    "type": "hovering_reduction",
                    "description": "Potentially excessive hovering time",
                    "recommendation": "Reduce decision-making time and improve navigation efficiency",
                    "potential_savings": "10-20%",
                }
            )

        return opportunities

    def _estimate_energy_components(self, total_energy: float) -> Dict[str, float]:
        """Estimate energy breakdown by components."""
        components = self.analysis_config["energy_components"]

        return {
            "thrust_energy": total_energy * components["thrust"],
            "avionics_energy": total_energy * components["avionics"],
            "communication_energy": total_energy * components["communication"],
            "other_energy": total_energy * components["other"],
            "total_energy": total_energy,
        }

    def _calculate_battery_life_impact(self, mean_energy: float) -> Dict[str, Any]:
        """Calculate battery life impact."""
        # Typical drone battery: 10Wh (36kJ) at 3.7V
        battery_capacity_joules = 36000

        # Missions per charge
        missions_per_charge = battery_capacity_joules / mean_energy

        # Battery degradation consideration (80% after 500 cycles)
        missions_with_degradation = missions_per_charge * 0.8

        # Daily operation estimate
        missions_per_day = (
            missions_per_charge / 3
        )  # Conservative estimate (charging time)

        return {
            "missions_per_charge": missions_per_charge,
            "missions_with_degradation": missions_with_degradation,
            "estimated_missions_per_day": missions_per_day,
            "battery_efficiency": (
                "excellent"
                if missions_per_charge >= 50
                else (
                    "good"
                    if missions_per_charge >= 30
                    else "fair" if missions_per_charge >= 20 else "poor"
                )
            ),
        }

    def _generate_energy_summary(self, energy_history: List[float]) -> Dict[str, Any]:
        """Generate comprehensive energy summary."""
        if not energy_history:
            return {"error": "No energy data"}

        recent_energy = (
            np.mean(energy_history[-100:])
            if len(energy_history) >= 100
            else np.mean(energy_history)
        )

        # Efficiency grade
        efficiency_grade = self._grade_energy_efficiency(recent_energy)

        # Target achievement
        target_achievement = self._assess_energy_targets(recent_energy)

        # Key metrics
        summary = {
            "final_energy_consumption": recent_energy,
            "efficiency_grade": efficiency_grade,
            "target_achievements": target_achievement,
            "energy_consistency": self._calculate_energy_consistency_score(
                np.array(energy_history)
            ),
            "overall_assessment": self._generate_overall_energy_assessment(
                recent_energy, efficiency_grade
            ),
        }

        return summary

    def _grade_energy_efficiency(self, energy: float) -> str:
        """Grade energy efficiency."""
        targets = self.analysis_config["efficiency_targets"]

        if energy <= targets["excellent"]:
            return "A"
        elif energy <= targets["good"]:
            return "B"
        elif energy <= targets["acceptable"]:
            return "C"
        elif energy <= targets["poor"]:
            return "D"
        else:
            return "F"

    def _assess_energy_targets(self, energy: float) -> Dict[str, bool]:
        """Assess energy target achievements."""
        baseline_astar = self.analysis_config["baseline_energy"]["A_star_only"]
        energy_savings = (baseline_astar - energy) / baseline_astar * 100

        return {
            "energy_savings_25_percent": energy_savings >= 25.0,
            "energy_below_700j": energy <= 700.0,
            "energy_below_1000j": energy <= 1000.0,
            "better_than_rrt_pid": energy
            < self.analysis_config["baseline_energy"]["RRT_PID"],
        }

    def _calculate_energy_consistency_score(self, energy_array: np.ndarray) -> float:
        """Calculate energy consistency score (0-100)."""
        if len(energy_array) == 0:
            return 0.0

        # Coefficient of variation (lower is more consistent)
        cv = (
            np.std(energy_array) / np.mean(energy_array)
            if np.mean(energy_array) > 0
            else 1.0
        )

        # Convert to consistency score (0-100)
        consistency_score = max(0, 100 - (cv * 200))  # CV of 0.5 = 0 points

        return float(consistency_score)

    def generate_energy_visualizations(
        self, analysis_results: Dict[str, Any], output_dir: str
    ):
        """Generate energy analysis visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Energy progression plot
            if "efficiency_trends" in analysis_results:
                self._plot_energy_trends(
                    analysis_results["efficiency_trends"], output_path
                )

            # Energy distribution plot
            if "distribution_analysis" in analysis_results:
                self._plot_energy_distribution(
                    analysis_results["distribution_analysis"], output_path
                )

            # Baseline comparison plot
            if "baseline_comparison" in analysis_results:
                self._plot_baseline_comparison(
                    analysis_results["baseline_comparison"], output_path
                )

            # Component breakdown plot
            if "summary" in analysis_results:
                self._plot_component_breakdown(analysis_results["summary"], output_path)

            self.logger.info(f"Energy visualizations saved to {output_path}")

        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze energy consumption patterns")
    parser.add_argument(
        "--training-results", type=str, help="Path to training results JSON file"
    )
    parser.add_argument(
        "--evaluation-results", type=str, help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--output", type=str, default="results/energy_analysis", help="Output directory"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization plots"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/main_config.yaml",
        help="Configuration file",
    )

    args = parser.parse_args()

    if not args.training_results and not args.evaluation_results:
        print("Error: Must provide either --training-results or --evaluation-results")
        sys.exit(1)

    # Create analyzer
    analyzer = EnergyAnalyzer(args.config)

    # Run analysis
    results = {}

    if args.training_results:
        training_analysis = analyzer.analyze_training_energy(args.training_results)
        results["training_analysis"] = training_analysis

    if args.evaluation_results:
        evaluation_analysis = analyzer.analyze_evaluation_energy(
            args.evaluation_results
        )
        results["evaluation_analysis"] = evaluation_analysis

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "energy_analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate visualizations
    if args.visualize:
        for analysis_type, analysis_data in results.items():
            analyzer.generate_energy_visualizations(
                analysis_data, str(output_dir / analysis_type)
            )

    # Print summary
    print("\n⚡ ENERGY ANALYSIS SUMMARY")
    print("=" * 50)

    if "evaluation_analysis" in results:
        eval_efficiency = results["evaluation_analysis"]["efficiency_analysis"]
        print(f"Mean Energy: {eval_efficiency['mean_energy_consumption']:.0f}J")
        print(f"Efficiency Grade: {eval_efficiency['efficiency_grade']}")
        print(
            f"Energy Savings vs A*: {eval_efficiency['baseline_comparison']['A_star_only']['energy_savings_percent']:.1f}%"
        )
        print(
            f"Target Met: {'✅ YES' if eval_efficiency['baseline_comparison']['A_star_only']['target_met'] else '❌ NO'}"
        )

    print(f"📄 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
