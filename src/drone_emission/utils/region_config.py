"""
Region Configuration Management Module

Using Dataclass + JSON serialization, supports:
1. Static region configuration
2. Runtime statistics data persistence (stats)
3. Computed data caching for reuse (cache)
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RegionConfig:
    """Region configuration data class"""

    region_name_cn: str
    region_name_en: str
    region_level: str  # "city" or "district"
    parent_city: Optional[str] = None
    boundaries_dict: Optional[Dict[str, float]] = None
    stats: Dict[str, Any] = field(default_factory=dict)  # Statistics for output
    cache: Dict[str, Any] = field(default_factory=dict)  # Computed data for reuse

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return asdict(self)

    def set_stat(self, key: str, value: Any):
        """Set a statistics value"""
        # Convert numpy types to Python native types for JSON serialization
        if hasattr(value, 'item'):
            value = value.item()
        self.stats[key] = value

    def get_stat(self, key: str, default: Any = None) -> Any:
        """Get a statistics value"""
        return self.stats.get(key, default)

    def set_cache(self, key: str, value: Any):
        """Set a cached computed value"""
        self.cache[key] = value

    def get_cache(self, key: str, default: Any = None) -> Any:
        """Get a cached computed value"""
        return self.cache.get(key, default)


class RegionRegistry:
    """
    Region Configuration Registry

    Features:
    - Load configuration from JSON
    - Support runtime dynamic region registration
    - Persist to JSON
    """

    def __init__(self, config_path: Optional[str] = None):
        self._regions: Dict[str, RegionConfig] = {}
        self._config_path = config_path

        if config_path and Path(config_path).exists():
            self.load(config_path)

    def load(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for region_key, region_data in data.items():
            self._regions[region_key] = RegionConfig(
                region_name_cn=region_data["region_name_cn"],
                region_name_en=region_data["region_name_en"],
                region_level=region_data["region_level"],
                parent_city=region_data.get("parent_city"),
                boundaries_dict=region_data.get("boundaries_dict"),
                stats=region_data.get("stats", {}),
                cache=region_data.get("cache", {}),
            )
        self._config_path = config_path

    def save(self, config_path: Optional[str] = None):
        """Save configuration to JSON file"""
        path = config_path or self._config_path
        if not path:
            raise ValueError("No config path specified")

        data = {key: region.to_dict() for key, region in self._regions.items()}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def register(
        self,
        region_key: str,
        region_name_cn: str,
        region_name_en: str,
        region_level: str,
        parent_city: Optional[str] = None,
        boundaries_dict: Optional[Dict[str, float]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> RegionConfig:
        """Register a region configuration"""
        config = RegionConfig(
            region_name_cn=region_name_cn,
            region_name_en=region_name_en,
            region_level=region_level,
            parent_city=parent_city,
            boundaries_dict=boundaries_dict,
            stats=stats or {},
        )
        self._regions[region_key] = config
        return config

    def get(self, region_key: str) -> Optional[RegionConfig]:
        """Get region configuration"""
        return self._regions.get(region_key)

    def __getitem__(self, region_key: str) -> RegionConfig:
        """Support registry["nanshan"] syntax"""
        if region_key not in self._regions:
            raise KeyError(f"Region '{region_key}' not found")
        return self._regions[region_key]

    def __contains__(self, region_key: str) -> bool:
        """Support 'nanshan' in registry syntax"""
        return region_key in self._regions

    def keys(self):
        """Return all region keys"""
        return self._regions.keys()

    def items(self):
        """Return all region items"""
        return self._regions.items()

    def values(self):
        """Return all region configurations"""
        return self._regions.values()

    def list_by_level(self, level: str) -> Dict[str, RegionConfig]:
        """Filter regions by level"""
        return {k: v for k, v in self._regions.items() if v.region_level == level}

    def list_by_parent(self, parent_city: str) -> Dict[str, RegionConfig]:
        """Get all districts under a city"""
        return {k: v for k, v in self._regions.items() if v.parent_city == parent_city}


# ===== Singleton Pattern =====

_default_registry: Optional[RegionRegistry] = None


def get_registry() -> RegionRegistry:
    """Get the default region registry"""
    global _default_registry
    if _default_registry is None:
        _default_registry = RegionRegistry()
    return _default_registry


def init_registry(config_path: str) -> RegionRegistry:
    """Initialize the region registry"""
    global _default_registry
    _default_registry = RegionRegistry(config_path)
    return _default_registry
