import os
import sys
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.utils import setup_logging

class PerformanceValidator:

    def __init__(self, config_path: str = None):
        if config_path:
            from src.utils import load_config

            config = load_config(config_path)
            self.logger_system = setup_logging(config.logging)

        self.logger = logging.getLogger(__name__)

        self.targets = {
            "success_rate": {
                "min_value": 96.0,
                "unit": "",
                "description": "Navigation success rate",
            },
            "energy_savings": {
                "min_value": 25.0,
                "unit": "",
                "description": "Energy efficiency improvement",
            },
            "ate_error": {
                "max_value": 0.05,
                "unit": "m",
                "description": "Absolute trajectory error",
            },
            "collision_rate": {
                "max_value": 2.0,
                "unit": "",
                "description": "Safety collision rate",
            },
            "flight_time": {
                "max_value": 120.0,
                "unit": "s",
                "description": "Mission completion time",
            },
        }

        self.statistical_requirements = {
            "min_sample_size": 100,
            "confidence_level": 0.95,
            "significance_level": 0.05,
        }

        self.validation_results = {}

        self.logger.info("Performance Validator initialized")
        self.logger.info(f"Validating against {len(self.targets)} performance targets")

    def validate_results(
        self, evaluation_file: str, baseline_file: str = None
    ) - Dict[str, Any]:

        self.logger.info("Starting performance validation")

        with open(evaluation_file, "r") as f:
            evaluation_data = json.load(f)

        baseline_data = None
        if baseline_file and Path(baseline_file).exists():
            with open(baseline_file, "r") as f:
                baseline_data = json.load(f)

        validation_results = {}

        target_validation = self._validate_targets(evaluation_data)
        validation_results["target_compliance"] = target_validation

        statistical_validation = self._validate_statistics(evaluation_data)
        validation_results["statistical_validation"] = statistical_validation

        if baseline_data:
            baseline_validation = self._validate_baseline_comparison(
                evaluation_data, baseline_data
            )
            validation_results["baseline_validation"] = baseline_validation

        robustness_validation = self._validate_robustness(evaluation_data)
        validation_results["robustness_validation"] = robustness_validation

        overall_assessment = self._assess_overall_compliance(validation_results)
        validation_results["overall_assessment"] = overall_assessment

        self.validation_results = validation_results

        self.logger.info("Performance validation completed")
        return validation_results

    def _validate_targets(self, evaluation_data: Dict[str, Any]) - Dict[str, Any]:

        metrics = evaluation_data.get("performance_metrics", {})
        target_results = {}

        success_rate = metrics.get("success_rate", 0)
        target_results["success_rate"] = {
            "value": success_rate,
            "target": self.targets["success_rate"]["min_value"],
            "met": success_rate = self.targets["success_rate"]["min_value"],
            "margin": success_rate - self.targets["success_rate"]["min_value"],
        }

        ate_error = metrics.get("mean_ate", float("inf"))
        target_results["ate_error"] = {
            "value": ate_error,
            "target": self.targets["ate_error"]["max_value"],
            "met": ate_error = self.targets["ate_error"]["max_value"],
            "margin": self.targets["ate_error"]["max_value"] - ate_error,
        }

        collision_rate = metrics.get("collision_rate", 100)
        target_results["collision_rate"] = {
            "value": collision_rate,
            "target": self.targets["collision_rate"]["max_value"],
            "met": collision_rate = self.targets["collision_rate"]["max_value"],
            "margin": self.targets["collision_rate"]["max_value"] - collision_rate,
        }

        flight_time = metrics.get("mean_time", float("inf"))
        target_results["flight_time"] = {
            "value": flight_time,
            "target": self.targets["flight_time"]["max_value"],
            "met": flight_time = self.targets["flight_time"]["max_value"],
            "margin": self.targets["flight_time"]["max_value"] - flight_time,
        }

        targets_met_count = sum(
            1 for result in target_results.values() if result["met"]
        )

        return {
            "targets_tested": len(target_results),
            "targets_met": targets_met_count,
            "compliance_rate": targets_met_count / len(target_results)  100,
            "individual_targets": target_results,
            "all_targets_met": targets_met_count == len(target_results),
        }

    def _validate_statistics(self, evaluation_data: Dict[str, Any]) - Dict[str, Any]:

        episodes_evaluated = evaluation_data.get("episodes_evaluated", 0)

        return {
            "sample_size_adequate": episodes_evaluated
            = self.statistical_requirements["min_sample_size"],
            "episodes_evaluated": episodes_evaluated,
            "min_required_episodes": self.statistical_requirements["min_sample_size"],
            "confidence_level": self.statistical_requirements["confidence_level"],
            "significance_level": self.statistical_requirements["significance_level"],
            "statistical_power": (
                "Adequate" if episodes_evaluated = 100 else "Insufficient"
            ),
        }

    def _validate_baseline_comparison(
        self, evaluation_data: Dict[str, Any], baseline_data: Dict[str, Any]
    ) - Dict[str, Any]:

        baseline_comparison = evaluation_data.get("baseline_comparison", {})

        energy_savings = baseline_comparison.get("energy_savings", {})
        astar_savings = energy_savings.get("vs_astar_only", 0)

        return {
            "energy_savings_vs_astar": astar_savings,
            "energy_target_met": astar_savings = 25.0,
            "baseline_methods_compared": len(baseline_data) if baseline_data else 0,
            "statistical_significance": baseline_comparison.get(
                "statistical_comparisons", {}
            )
            != {},
            "comparison_validity": (
                "Valid" if baseline_data and len(baseline_data) = 3 else "Limited"
            ),
        }

    def _validate_robustness(self, evaluation_data: Dict[str, Any]) - Dict[str, Any]:

        scenario_data = evaluation_data.get("scenario_results")

        if not scenario_data:
            return {
                "robustness_tested": False,
                "recommendation": "Run scenario testing for robustness validation",
            }

        scenario_success_rates = [
            results.get("success_rate", 0) for results in scenario_data.values()
        ]

        min_scenario_success = (
            min(scenario_success_rates) if scenario_success_rates else 0
        )
        mean_scenario_success = (
            np.mean(scenario_success_rates) if scenario_success_rates else 0
        )

        return {
            "robustness_tested": True,
            "scenarios_tested": len(scenario_data),
            "min_scenario_success_rate": min_scenario_success,
            "mean_scenario_success_rate": mean_scenario_success,
            "robustness_adequate": min_scenario_success = 70.0,
            "stress_test_passed": mean_scenario_success = 80.0,
        }

    def _assess_overall_compliance(
        self, validation_results: Dict[str, Any]
    ) - Dict[str, Any]:

        target_compliance = validation_results.get("target_compliance", {})
        all_targets_met = target_compliance.get("all_targets_met", False)

        statistical_validation = validation_results.get("statistical_validation", {})
        statistical_adequate = statistical_validation.get("sample_size_adequate", False)

        baseline_validation = validation_results.get("baseline_validation", {})
        baseline_adequate = baseline_validation.get("comparison_validity") == "Valid"

        compliance_score = 0

        if all_targets_met:
            compliance_score += 40
        elif target_compliance.get("compliance_rate", 0) = 75:
            compliance_score += 30

        if statistical_adequate:
            compliance_score += 25

        if baseline_adequate:
            compliance_score += 20

        if validation_results.get("robustness_validation", {}).get(
            "robustness_adequate", False
        ):
            compliance_score += 15

        if compliance_score = 90:
            compliance_grade = "A - Full Compliance"
        elif compliance_score = 80:
            compliance_grade = "B - High Compliance"
        elif compliance_score = 70:
            compliance_grade = "C - Acceptable Compliance"
        elif compliance_score = 60:
            compliance_grade = "D - Limited Compliance"
        else:
            compliance_grade = "F - Non-Compliant"

        return {
            "compliance_score": compliance_score,
            "compliance_grade": compliance_grade,
            "deployment_ready": compliance_score = 80,
            "research_requirements_met": all_targets_met and statistical_adequate,
            "recommendations": self._generate_compliance_recommendations(
                validation_results
            ),
        }

    def _generate_compliance_recommendations(
        self, validation_results: Dict[str, Any]
    ) - List[str]:

        recommendations = []

        target_compliance = validation_results.get("target_compliance", {})
        if not target_compliance.get("all_targets_met", False):
            unmet_targets = [
                name
                for name, result in target_compliance.get(
                    "individual_targets", {}
                ).items()
                if not result["met"]
            ]
            recommendations.append(f"Address unmet targets: {', '.join(unmet_targets)}")

        statistical_validation = validation_results.get("statistical_validation", {})
        if not statistical_validation.get("sample_size_adequate", False):
            recommendations.append(
                "Increase evaluation episodes for statistical validity"
            )

        return recommendations

def main():
    parser = argparse.ArgumentParser(
        description="Validate model performance against targets"
    )
    parser.add_argument(
        "--evaluation", type=str, required=True, help="Path to evaluation results JSON"
    )
    parser.add_argument("--baselines", type=str, help="Path to baseline results JSON")
    parser.add_argument(
        "--output",
        type=str,
        default="results/performance_validation.json",
        help="Output validation report",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/evaluation_config.yaml",
        help="Configuration file",
    )

    args = parser.parse_args()

    validator = PerformanceValidator(args.config)

    results = validator.validate_results(args.evaluation, args.baselines)

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\nPERFORMANCE VALIDATION SUMMARY")
    print("="  50)

    overall = results["overall_assessment"]
    print(f"Compliance Grade: {overall['compliance_grade']}")
    print(f"Deployment Ready: {' YES' if overall['deployment_ready'] else ' NO'}")
    print(
        f"Research Requirements: {' MET' if overall['research_requirements_met'] else ' NOT MET'}"
    )

    target_compliance = results["target_compliance"]
    print(
        f"\nTarget Compliance: {target_compliance['targets_met']}/{target_compliance['targets_tested']} "
        f"({target_compliance['compliance_rate']:.1f})"
    )

    for target_name, target_result in target_compliance["individual_targets"].items():
        status = "" if target_result["met"] else ""
        print(
            f"  {status} {target_name}: {target_result['value']:.3f} "
            f"(target: {target_result['target']:.3f})"
        )

if __name__ == "__main__":
    main()
