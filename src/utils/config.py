"""
Configuration management for Smart City Traffic System.
Loads and provides access to configuration parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        if config_path is None:
            # Get project root directory
            self.project_root = Path(__file__).parent.parent.parent
            config_path = self.project_root / "config.yaml"
        else:
            config_path = Path(config_path)
            self.project_root = config_path.parent
        
        self.config_path = config_path
        self._config = self._load_config()
        self._ensure_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        paths = self._config.get('paths', {})
        for path_key, path_value in paths.items():
            full_path = self.project_root / path_value
            full_path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key in dot notation (e.g., 'models.xgboost.n_estimators')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_path(self, path_key: str) -> Path:
        """
        Get absolute path for a configured path.
        
        Args:
            path_key: Key in 'paths' section of config
            
        Returns:
            Absolute Path object
        """
        relative_path = self.get(f'paths.{path_key}')
        if relative_path is None:
            raise KeyError(f"Path '{path_key}' not found in configuration")
        
        return self.project_root / relative_path
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __repr__(self) -> str:
        return f"Config(path='{self.config_path}')"


# Global configuration instance
_global_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        Config instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = Config(config_path)
    
    return _global_config


def reload_config(config_path: str = None):
    """
    Reload configuration from file.
    
    Args:
        config_path: Path to config file
    """
    global _global_config
    _global_config = Config(config_path)


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    
    print("Configuration loaded successfully!")
    print(f"Project root: {config.project_root}")
    print(f"\nSample configurations:")
    print(f"  Number of road segments: {config.get('data_generation.num_road_segments')}")
    print(f"  XGBoost estimators: {config.get('models.xgboost.n_estimators')}")
    print(f"  Min green time: {config.get('optimization.min_green_time')}")
    print(f"\nData paths:")
    print(f"  Raw data: {config.get_path('data_raw')}")
    print(f"  Models: {config.get_path('models')}")
