"""
Region Configuration Management Module

Using Dataclass + JSON serialization, supports:
1. Static region configuration (read-only regions.json)
2. Runtime statistics data persistence (per-region stats files)
3. Computed data caching for reuse (cache)

Stats/cache are stored per-region in data/{level}/{city}/region_stats.json
to avoid write conflicts during parallel multiprocessing runs.
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

    def __post_init__(self):
        # Path to per-region stats file, set by RegionRegistry
        self._stats_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return asdict(self)

    def set_stat(self, key: str, value: Any):
        """Set a statistics value"""
        # Convert numpy types to Python native types for JSON serialization
        if hasattr(value, "item"):
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

    def save_stats(self):
        """Save stats and cache to per-region stats file."""
        if not self._stats_path:
            return
        path = Path(self._stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"stats": self.stats, "cache": self.cache}
        from drone_emission.utils.atomic import atomic_open

        with atomic_open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def load_stats(self):
        """Load stats and cache from per-region stats file if it exists."""
        if not self._stats_path:
            return
        path = Path(self._stats_path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.stats.update(data.get("stats", {}))
        self.cache.update(data.get("cache", {}))


class RegionRegistry:
    """
    Region Configuration Registry

    Features:
    - Load static configuration from regions.json (read-only)
    - Load/save runtime stats from per-region files in data/{level}/{city}/region_stats.json
    - Support runtime dynamic region registration
    """

    def __init__(self, config_path: Optional[str] = None, data_dir: Optional[str] = None):
        self._regions: Dict[str, RegionConfig] = {}
        self._config_path = config_path
        self._data_dir = data_dir

        if config_path and Path(config_path).exists():
            self.load(config_path)

    def _stats_path_for(self, region: RegionConfig) -> Optional[str]:
        """Return the per-region stats file path, or None if data_dir is not set."""
        if not self._data_dir:
            return None
        return str(
            Path(self._data_dir)
            / region.region_level
            / region.region_name_en.lower()
            / "region_stats.json"
        )

    def load(self, config_path: str):
        """Load static config from JSON, then load per-region stats from data dir."""
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for region_key, region_data in data.items():
            region = RegionConfig(
                region_name_cn=region_data["region_name_cn"],
                region_name_en=region_data["region_name_en"],
                region_level=region_data["region_level"],
                parent_city=region_data.get("parent_city"),
                boundaries_dict=region_data.get("boundaries_dict"),
                # Start with stats/cache from regions.json as fallback
                stats=region_data.get("stats", {}),
                cache=region_data.get("cache", {}),
            )
            region._stats_path = self._stats_path_for(region)
            # Override with per-region stats file if it exists
            region.load_stats()
            self._regions[region_key] = region
        self._config_path = config_path

    def save(self):
        """Save all regions' stats to their per-region files."""
        for region in self._regions.values():
            region.save_stats()

    def save_region(self, region_key: str):
        """Save only one region's stats (used in per-city processing)."""
        region = self._regions[region_key]
        region.save_stats()

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
        config._stats_path = self._stats_path_for(config)
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


def init_registry(config_path: str, data_dir: Optional[str] = None) -> RegionRegistry:
    """Initialize the region registry.

    Args:
        config_path: Path to regions.json (static config).
        data_dir: Base data directory. When set, per-region stats are loaded
            from and saved to data/{level}/{city}/region_stats.json instead of
            writing back to regions.json.
    """
    global _default_registry
    _default_registry = RegionRegistry(config_path, data_dir=data_dir)
    return _default_registry
