"""
Configuration loader for ASL Recognition project.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Loads and manages configuration settings."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'model.architecture')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigLoader(config_path='{self.config_path}')"


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Utility function to load configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.config
