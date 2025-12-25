import tomllib
from typing import Any, Dict


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

    return config
