"""Configuration loading utilities."""
from pathlib import Path
from typing import Any, Dict
import yaml
from loguru import logger

def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load a single YAML configuration file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"Config file not found: {path_obj}")
        return {}
    with open(path_obj, "r") as f:
        return yaml.safe_load(f) or {}

def load_all(config_dir: str | Path = "config") -> Dict[str, Any]:
    """
    Load all YAML configuration files in a directory and merge them into one dictionary.

    Args:
        config_dir (str | Path): Directory containing .yaml files.

    Returns:
        Dict[str, Any]: Merged configuration dictionary.
    """
    config_dir_obj = Path(config_dir)
    merged_config: Dict[str, Any] = {}
    if not config_dir_obj.exists():
        logger.warning(f"Config directory not found: {config_dir_obj}")
        return merged_config
    
    for yaml_file in config_dir_obj.glob("*.yaml"):
        file_config = load_config(yaml_file)
        if file_config and isinstance(file_config, dict):
            merged_config.update(file_config)
        
    return merged_config
