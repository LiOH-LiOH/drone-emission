import os
import sys
from pathlib import Path
from typing import Any, Dict

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Default cache root (project-level ./cache)
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent)
_cache_root: str = os.path.join(_PROJECT_ROOT, "cache")


def get_cache_dir(*subdirs: str) -> str:
    """Return a cache subdirectory path, creating it if needed."""
    path = os.path.join(_cache_root, *subdirs)
    os.makedirs(path, exist_ok=True)
    return path


def configure_cache(exp_config: Dict[str, Any]) -> None:
    """Set the cache root from exp_config['cache_dir'], then update library caches."""
    global _cache_root
    _cache_root = exp_config.get("cache_dir", _cache_root)
    os.makedirs(_cache_root, exist_ok=True)

    # OSMnx cache (only if osmnx is already imported)
    try:
        import osmnx as ox
        ox.settings.cache_folder = get_cache_dir("osmnx")
    except ImportError:
        pass


def load_config(config_path: str, env: str = None) -> Dict[str, Any]:
    """
    Load configuration from a TOML file.

    Args:
        config_path: Path to the TOML configuration file.
        env: The environment to load (e.g., 'default', 'test').
             If None, it looks for 'active_env' in the config file,
             defaulting to 'default'.

    Returns:
        A dictionary containing the configuration.
    """
    with open(config_path, "rb") as f:
        toml_data = tomllib.load(f)

    if env is None:
        env = toml_data.get("active_env", "default")

    # Start with a copy of the whole data to keep top-level defaults
    config = toml_data.copy()

    # If a section for the environment exists, override defaults
    if env != "default" and env in toml_data:
        # Update with environment-specific values
        # Note: This will overwrite top-level keys with keys from the [env] section
        config.update(toml_data[env])

    from drone_emission.paths import apply_path_overrides
    return apply_path_overrides(config, env=env)
