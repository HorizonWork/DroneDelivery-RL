#!/usr/bin/env python3
"""
Smoke test: reset AirSim environment multiple times and verify spawn/takeoff.
"""

import argparse
import logging
import sys
import time
from copy import deepcopy
from typing import Any, Dict

import numpy as np

ROOT = None
if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.utils import load_config
from src.environment.airsim_env import AirSimEnvironment as DroneEnvironment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spawn + takeoff smoke test for DroneDelivery-RL"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/main_config.yaml",
        help="Path to main configuration file",
    )
    parser.add_argument(
        "--resets",
        type=int,
        default=5,
        help="Number of reset cycles to execute",
    )
    parser.add_argument(
        "--stabilize",
        type=float,
        default=1.0,
        help="Seconds to wait after each reset/takeoff",
    )
    parser.add_argument(
        "--alt-tolerance",
        type=float,
        default=0.2,
        help="Tolerance (meters) for altitude check",
    )
    return parser.parse_args()


def make_env_config(config: Any) -> Dict[str, Any]:
    env_cfg = getattr(config, "environment", {})
    if isinstance(env_cfg, dict):
        return deepcopy(env_cfg)
    return dict(env_cfg)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = load_config(args.config)
    env_config = make_env_config(config)
    logging.info("Loaded config from %s", args.config)

    env: DroneEnvironment = DroneEnvironment(env_config)
    alt_target = getattr(env, "initial_takeoff_altitude", 3.0)

    try:
        for attempt in range(1, args.resets + 1):
            logging.info("=== Reset %d/%d ===", attempt, args.resets)
            try:
                observation, info = env.reset()
            except Exception as exc:  # Requires live AirSim
                logging.error("Reset failed: %s", exc)
                break

            spawn_cfg = info.get("spawn_location", env.airsim_bridge.spawn_location)
            orientation_cfg = info.get(
                "spawn_orientation", env.airsim_bridge.spawn_orientation
            )
            drone_state = env.airsim_bridge.get_drone_state()
            actual_position = drone_state.position
            actual_altitude = abs(actual_position[2])
            altitude_diff = abs(actual_altitude - alt_target)
            passed = altitude_diff <= args.alt_tolerance

            logging.info("Observation shape: %s", np.shape(observation))
            logging.info(
                "Commanded spawn=%s | orientation=%s",
                tuple(round(v, 3) for v in spawn_cfg),
                tuple(round(v, 3) for v in orientation_cfg),
            )
            logging.info(
                "AirSim pose=%s | altitude=%.2fm | target=%.2fm | diff=%.3fm (%s)",
                tuple(round(v, 3) for v in actual_position),
                actual_altitude,
                alt_target,
                altitude_diff,
                "PASS" if passed else "FAIL",
            )

            time.sleep(args.stabilize)

    except KeyboardInterrupt:
        logging.info("Test interrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
