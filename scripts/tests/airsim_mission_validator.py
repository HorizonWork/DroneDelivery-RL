#!/usr/bin/env python3
"""
AirSim Mission Validator
========================

Utility script to verify end-to-end connectivity with the AirSim simulator,
execute a multi-waypoint mission, and capture rich telemetry/energy statistics.

Example:
    python scripts/tests/airsim_mission_validator.py \
        --waypoints "0,0,-2;5,0,-2;5,5,-2" \
        --vehicle-name Drone1 \
        --cruise-speed 2.0 \
        --log-file logs/airsim_mission_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bridges.airsim_bridge import AirSimBridge


def parse_waypoints(spec: str) -> List[Tuple[float, float, float]]:
    """Parse a waypoint specification string."""
    waypoints: List[Tuple[float, float, float]] = []
    for chunk in spec.split(";"):
        values = [float(v.strip()) for v in chunk.split(",")]
        if len(values) != 3:
            raise argparse.ArgumentTypeError(
                f"Waypoint '{chunk}' must contain three comma-separated values."
            )
        waypoints.append(tuple(values))
    return waypoints


def default_waypoints() -> List[Tuple[float, float, float]]:
    """Return an elongated rectangle path to stress-test API throughput."""
    return [
        (0.0, 0.0, -2.0),
        (8.0, 0.0, -2.0),
        (8.0, 6.0, -2.0),
        (0.0, 6.0, -2.0),
    ]


def quaternion_to_tilt_deg(quat: Sequence[float]) -> float:
    """Return the maximum tilt (deg) from a quaternion (w, x, y, z)."""
    rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    roll, pitch, _ = rotation.as_euler("xyz", degrees=True)
    return max(abs(roll), abs(pitch))


class TelemetryRecorder:
    """Collect telemetry samples and estimate energy usage."""

    def __init__(
        self,
        poll_interval: float,
        horiz_energy_per_meter: float,
        vert_energy_per_meter: float,
    ) -> None:
        self.poll_interval = poll_interval
        self.horiz_coef = horiz_energy_per_meter
        self.vert_coef = vert_energy_per_meter
        self.records: List[dict] = []
        self.total_energy_j = 0.0
        self.max_tilt_deg = 0.0
        self._segment_index = -1
        self._target: Tuple[float, float, float] | None = None

    def start_segment(self, idx: int, target: Tuple[float, float, float]) -> None:
        self._segment_index = idx
        self._target = target

    def capture(self, state, battery_level: float) -> None:
        if state is None or self._target is None:
            return

        timestamp = time.time()
        position = np.array(state.position, dtype=np.float32)
        velocity = np.array(state.linear_velocity, dtype=np.float32)
        speed = float(np.linalg.norm(velocity))
        tilt = quaternion_to_tilt_deg(state.orientation)
        self.max_tilt_deg = max(self.max_tilt_deg, tilt)

        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        vertical_speed = abs(float(velocity[2]))
        energy_increment = (
            horizontal_speed * self.horiz_coef + vertical_speed * self.vert_coef
        ) * self.poll_interval
        self.total_energy_j += energy_increment

        self.records.append(
            {
                "timestamp": timestamp,
                "segment": self._segment_index,
                "target": list(self._target),
                "position": position.tolist(),
                "velocity": velocity.tolist(),
                "speed_mps": speed,
                "tilt_deg": tilt,
                "battery": battery_level,
                "energy_j": self.total_energy_j,
            }
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate AirSim connection and drone control via scripted mission."
    )
    parser.add_argument(
        "--vehicle-name",
        default="Drone1",
        help="AirSim vehicle name (matches settings.json).",
    )
    parser.add_argument(
        "--waypoints",
        type=parse_waypoints,
        default=default_waypoints(),
        help="Semicolon-separated list of waypoints 'x,y,z;...'. Defaults to a 5m square.",
    )
    parser.add_argument(
        "--takeoff-altitude",
        type=float,
        default=2.0,
        help="Takeoff altitude in meters (positive).",
    )
    parser.add_argument(
        "--cruise-speed",
        type=float,
        default=3.0,
        help="Cruise speed in m/s for velocity guidance.",
    )
    parser.add_argument(
        "--cruise-timeout",
        type=float,
        default=10.0,
        help="Maximum seconds to attempt reaching each waypoint.",
    )
    parser.add_argument(
        "--velocity-command-duration",
        type=float,
        default=0.5,
        help="Duration for each velocity command burst (seconds).",
    )
    parser.add_argument(
        "--respawn-on-collision",
        action="store_true",
        help="Reset and re-takeoff automatically when a collision is detected.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.2,
        help="Telemetry polling interval while flying (seconds).",
    )
    parser.add_argument(
        "--energy-per-meter",
        type=float,
        default=35.0,
        help="Approximate energy usage (J) per meter of horizontal travel.",
    )
    parser.add_argument(
        "--energy-per-meter-vertical",
        type=float,
        default=45.0,
        help="Approximate energy usage (J) per meter of vertical travel.",
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.3,
        help="Distance tolerance (m) to consider a waypoint reached.",
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=1.0,
        help="Seconds to hover at each waypoint before moving on.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/airsim_mission_report.json"),
        help="Destination JSON file for telemetry + summary.",
    )
    parser.add_argument(
        "--land",
        action="store_true",
        help="Land the drone at the end of the mission.",
    )
    return parser


def telemetry_worker(
    bridge: AirSimBridge,
    recorder: TelemetryRecorder,
    stop_event: threading.Event,
    poll_interval: float,
) -> None:
    """Background thread that samples telemetry until stop_event is set."""
    while not stop_event.is_set():
        state = bridge.get_drone_state()
        battery = bridge.get_battery_level()
        recorder.capture(state, battery)
        stop_event.wait(poll_interval)


def fly_segment(
    bridge: AirSimBridge,
    recorder: TelemetryRecorder,
    target: Tuple[float, float, float],
    segment_index: int,
    args: argparse.Namespace,
) -> dict:
    recorder.start_segment(segment_index, target)
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=telemetry_worker,
        args=(bridge, recorder, stop_event, args.poll_interval),
        daemon=True,
    )
    monitor.start()
    summary = {
        "segment": segment_index,
        "target": target,
        "collision": False,
        "respawned": False,
    }
    try:
        deadline = time.time() + args.cruise_timeout
        while time.time() < deadline:
            state = bridge.get_drone_state()
            if state:
                current_pos = np.array(state.position, dtype=np.float32)
                delta = np.array(target, dtype=np.float32) - current_pos
                distance = float(np.linalg.norm(delta))
                if distance <= args.position_tolerance:
                    break
                direction = delta / max(distance, 1e-6)
                vx, vy, vz = direction * args.cruise_speed
            else:
                vx = vy = vz = 0.0

            bridge.send_velocity_command(
                float(vx),
                float(vy),
                float(vz),
                0.0,
                duration=args.velocity_command_duration,
            )
            time.sleep(args.velocity_command_duration * 0.75)

        stop_event.wait(args.hold_time)
        summary["collision"] = bridge.check_collision()
        state = bridge.get_drone_state()
        summary["final_position"] = list(state.position) if state else None
        summary["final_tilt_deg"] = (
            quaternion_to_tilt_deg(state.orientation) if state else None
        )
        if summary["collision"] and args.respawn_on_collision:
            logging.warning(
                "Collision detected on segment %d, respawning...", segment_index + 1
            )
            try:
                bridge.reset_drone()
                bridge.takeoff(args.takeoff_altitude)
                summary["respawned"] = True
            except Exception as exc:
                logging.error("Failed to respawn after collision: %s", exc)
    finally:
        stop_event.set()
        monitor.join()
    return summary


def run_mission(args: argparse.Namespace) -> dict:
    logging.info("Connecting to AirSim...")
    bridge = AirSimBridge({"drone_name": args.vehicle_name})
    if not bridge.connect():
        raise RuntimeError("Failed to connect to AirSim. Is the simulator running?")

    logging.info("Arming and taking off...")
    if not bridge.takeoff(args.takeoff_altitude, timeout=args.takeoff_altitude * 4):
        raise RuntimeError("Takeoff failed.")

    recorder = TelemetryRecorder(
        poll_interval=args.poll_interval,
        horiz_energy_per_meter=args.energy_per_meter,
        vert_energy_per_meter=args.energy_per_meter_vertical,
    )

    segment_summaries = []
    for idx, waypoint in enumerate(args.waypoints):
        logging.info("Flying segment %d -> target %s", idx + 1, waypoint)
        summary = fly_segment(bridge, recorder, waypoint, idx, args)
        segment_summaries.append(summary)
        logging.info(
            "Segment %d complete (collision=%s)",
            idx + 1,
            summary["collision"],
        )

    if args.land:
        logging.info("Landing...")
        bridge.land()

    mission_summary = {
        "vehicle": args.vehicle_name,
        "waypoints": [list(wp) for wp in args.waypoints],
        "segments": segment_summaries,
        "total_energy_j": recorder.total_energy_j,
        "max_tilt_deg": recorder.max_tilt_deg,
        "log_samples": len(recorder.records),
    }

    log_path: Path = args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"summary": mission_summary, "telemetry": recorder.records},
            handle,
            indent=2,
        )
    logging.info("Telemetry written to %s", log_path)

    bridge.disconnect()
    return mission_summary


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    summary = run_mission(args)
    logging.info(
        "Mission complete: %d segments, %.1f J total energy, max tilt %.2f°",
        len(summary["segments"]),
        summary["total_energy_j"],
        summary["max_tilt_deg"],
    )


if __name__ == "__main__":
    main()
