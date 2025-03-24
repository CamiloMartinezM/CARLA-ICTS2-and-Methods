"""Provides utility functions for loading and manipulating different sources."""

import json
import shutil
from pathlib import Path

import yaml

from carla_icts2.config import logger


def load_yaml(file_path: str = "config.yaml") -> dict:
    """Load a YAML file.

    Args:
        file_path (str): Path to the YAML file. Defaults to "config.yaml".

    Returns:
        dict: Loaded data from the YAML file
    """
    with Path(file_path).open("r") as f:
        return yaml.safe_load(f)


def load_json(
    file_path: Path | str,
    exclude_keys: set[str] | None = None,
    *,
    warn: bool = False,
    raise_error: bool = False,
) -> dict:
    """Load annotations from a JSON file.

    Args:
        file_path (str): Path to the JSON file
        exclude_keys (set[str] | None): Keys to exclude from the loaded data
        warn (bool): Whether to print a warning if the file is not found
        raise_error (bool): Whether to raise an error if the file is not found

    Returns:
        dict: Loaded data from the JSON file or an empty dictionary if not found

    Raises:
        FileNotFoundError: If the file is not found and raise_error is True
    """
    result = {}
    if Path(file_path).exists():
        with Path(file_path).open("r") as f:
            result = json.load(f)
    elif raise_error:
        raise FileNotFoundError(f"File not found: {file_path}")
    elif warn:
        logger.warning(f"File not found: {file_path}")

    if exclude_keys:
        for key in exclude_keys:
            if key in result:
                del result[key]

    return result


def save_json(data: dict, file_path: Path | str) -> None:
    """Save annotations to a JSON file.

    Args:
        data (dict): The data to save
        file_path (Path | str): Path to save the JSON file
    """
    # Create backup before saving
    create_backup(file_path)

    with Path(file_path).open("w") as f:
        # Pretty printing with consistent indentation
        json.dump(data, f, indent=4)


def create_backup(file_path: Path | str) -> None:
    """Create a backup of a file with .bak extension.

    Args:
        file_path (str): Path to the file to backup
    """
    if Path(file_path).exists():
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)


def restore_from_backup(file_path: Path | str) -> bool:
    """Restore a file from its backup (.bak) and make the original file the new backup.

    Args:
        file_path (Path | str): Path to the file to restore

    Returns:
        bool: True if restoration was successful, False otherwise
    """
    backup_path = f"{file_path}.bak"

    # Check if backup exists
    if not Path(backup_path).exists():
        logger.info(f"No backup file found at {backup_path}")
        return False

    # Create a new backup of the current file
    try:
        if Path(file_path).exists():
            temp_path = f"{file_path}.tmp"
            shutil.copy2(file_path, temp_path)

        # Restore the backup to the original filename
        shutil.copy2(backup_path, file_path)

        # Replace the old backup with the temp file
        if Path(temp_path).exists():
            shutil.move(temp_path, backup_path)
    except Exception as e:
        logger.error(f"Error restoring backup: {e}")
        return False
    else:
        return True
