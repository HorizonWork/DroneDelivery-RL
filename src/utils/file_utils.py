import os
import shutil
import logging
import tempfile
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import yaml
from datetime import datetime, timedelta

class FileManager:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.base_dir = Path(config.get("base_dir", "."))
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "outputs"
        self.temp_dir = self.base_dir / "temp"
        self.models_dir = self.base_dir / "models"

        self.subdirs = {
            "trajectories": self.data_dir / "trajectories",
            "checkpoints": self.models_dir / "checkpoints",
            "evaluations": self.output_dir / "evaluations",
            "logs": self.output_dir / "logs",
            "maps": self.data_dir / "maps",
            "configs": self.base_dir / "config",
        }

        self.cleanup_enabled = config.get("cleanup_enabled", True)
        self.temp_file_retention = timedelta(
            hours=config.get("temp_retention_hours", 24)
        )
        self.log_retention = timedelta(days=config.get("log_retention_days", 30))

        self._create_directories()

        if self.cleanup_enabled:
            self._schedule_cleanup()

        self.logger.info("File Manager initialized")
        self.logger.info(f"Base directory: {self.base_dir}")

    def _create_directories(self):

        directories = [
            self.data_dir,
            self.output_dir,
            self.temp_dir,
            self.models_dir,
        ] + list(self.subdirs.values())

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.info("Directory structure created")

    def get_path(self, path_type: str, filename: Optional[str] = None) - Path:

        if path_type not in self.subdirs:
            raise ValueError(f"Unknown path type: {path_type}")

        base_path = self.subdirs[path_type]

        if filename:
            return base_path / filename

        return base_path

    def save_json(self, data: Any, path_type: str, filename: str) - Path:

        filepath = self.get_path(path_type, filename)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.debug(f"JSON saved: {filepath}")
        return filepath

    def load_json(self, path_type: str, filename: str) - Any:

        filepath = self.get_path(path_type, filename)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        return data

    def save_yaml(self, data: Any, path_type: str, filename: str) - Path:

        filepath = self.get_path(path_type, filename)

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)

        self.logger.debug(f"YAML saved: {filepath}")
        return filepath

    def create_temp_file(self, suffix: str = "", prefix: str = "drone_") - Path:

        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix, prefix=prefix, dir=self.temp_dir, delete=False
        )
        temp_file.close()

        return Path(temp_file.name)

    def cleanup_temp_files(self):

        if not self.temp_dir.exists():
            return

        cutoff_time = datetime.now() - self.temp_file_retention
        removed_count = 0

        for temp_file in self.temp_dir.iterdir():
            if temp_file.is_file():
                file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                if file_time  cutoff_time:
                    try:
                        temp_file.unlink()
                        removed_count += 1
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to remove temp file {temp_file}: {e}"
                        )

        if removed_count  0:
            self.logger.info(f"Cleaned up {removed_count} temporary files")

    def _schedule_cleanup(self):

        self.cleanup_temp_files()

class PathManager:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def resolve_path(
        self, path: Union[str, Path], base_dir: Optional[Path] = None
    ) - Path:

        path_obj = Path(path)

        if path_obj.is_absolute():
            return path_obj

        if base_dir:
            return (base_dir / path_obj).resolve()

        return path_obj.resolve()

    def ensure_directory(self, directory: Union[str, Path]) - Path:

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def get_file_info(self, filepath: Union[str, Path]) - Dict[str, Any]:

        file_path = Path(filepath)

        if not file_path.exists():
            return {"exists": False}

        stat = file_path.stat()

        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024  1024),
            "modified_time": datetime.fromtimestamp(stat.st_mtime),
            "is_file": file_path.is_file(),
            "is_directory": file_path.is_dir(),
            "extension": file_path.suffix,
        }
