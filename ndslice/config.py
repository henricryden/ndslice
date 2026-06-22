"""Persistent user configuration for ndslice."""

from dataclasses import dataclass, replace
from pathlib import Path
from platformdirs import user_config_dir

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised only on Python < 3.11
    import tomli as tomllib


INITIAL_INDEX_FIRST = "first"
INITIAL_INDEX_CENTER = "center"
INITIAL_INDEX_LAST = "last"
VALID_INITIAL_INDEX = {INITIAL_INDEX_FIRST, INITIAL_INDEX_CENTER, INITIAL_INDEX_LAST}

DEFAULT_COLORMAP = "gray"
DEFAULT_DISPLAY_MODE = "square_pixels"
SUPPORTED_DISPLAY_MODES = (
    "square_pixels",
    "square_fov",
    "fit",
)


@dataclass(frozen=True) # cannot change values after creation
class ViewerConfig:
    initial_index: str = INITIAL_INDEX_FIRST
    default_colormap: str = DEFAULT_COLORMAP
    default_display_mode: str = DEFAULT_DISPLAY_MODE


def get_config_path():
    return Path(user_config_dir("ndslice", appauthor=False)) / "config.toml"


def load_config(path=None, colormap_names=None):
    """Load user config, falling back to defaults for any invalid value."""
    config_path = Path(path) if path is not None else get_config_path()
    valid_colormaps = tuple(colormap_names) if colormap_names is not None else None

    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return ViewerConfig()

    startup = raw.get("startup", {})
    if not isinstance(startup, dict):
        startup = {}

    display = raw.get("display", {})
    if not isinstance(display, dict):
        display = {}

    initial_index = startup.get("initial_index", INITIAL_INDEX_FIRST)
    if initial_index not in VALID_INITIAL_INDEX:
        initial_index = INITIAL_INDEX_FIRST

    default_colormap = display.get("default_colormap", DEFAULT_COLORMAP)
    if valid_colormaps is not None and default_colormap not in valid_colormaps:
        default_colormap = DEFAULT_COLORMAP

    default_display_mode = display.get("default_display_mode", DEFAULT_DISPLAY_MODE)
    if default_display_mode not in SUPPORTED_DISPLAY_MODES:
        default_display_mode = DEFAULT_DISPLAY_MODE

    return ViewerConfig(
        initial_index=initial_index,
        default_colormap=default_colormap,
        default_display_mode=default_display_mode,
    )


def save_config(config, path=None, colormap_names=None):
    """Write the complete viewer config to TOML."""
    # use the provided config path if one was given
    config_path = Path(path) if path is not None else Path(user_config_dir("ndslice", appauthor=False)) / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    valid_colormaps = tuple(colormap_names) if colormap_names is not None else None

    initial_index = config.initial_index
    if initial_index not in VALID_INITIAL_INDEX:
        initial_index = INITIAL_INDEX_FIRST

    default_colormap = config.default_colormap
    if valid_colormaps is not None and default_colormap not in valid_colormaps:
        default_colormap = DEFAULT_COLORMAP

    default_display_mode = config.default_display_mode
    if default_display_mode not in SUPPORTED_DISPLAY_MODES:
        default_display_mode = DEFAULT_DISPLAY_MODE

    text = (
        "[startup]\n"
        f'initial_index = "{initial_index}"\n'
        "\n"
        "[display]\n"
        f'default_colormap = "{default_colormap}"\n'
        f'default_display_mode = "{default_display_mode}"\n'
    )
    config_path.write_text(text, encoding="utf-8")


def update_config(path=None, colormap_names=None, **changes):
    config = replace(load_config(path, colormap_names=colormap_names), **changes)
    save_config(config, path, colormap_names=colormap_names)
    return config
