"""Persistent user configuration for ndslice."""

from dataclasses import dataclass, replace
from pathlib import Path
import tomllib

from platformdirs import user_config_dir


INITIAL_INDEX_FIRST = "first"
INITIAL_INDEX_CENTER = "center"
INITIAL_INDEX_LAST = "last"
VALID_INITIAL_INDEX = {INITIAL_INDEX_FIRST, INITIAL_INDEX_CENTER, INITIAL_INDEX_LAST}
DEFAULT_SLICE_FIRST = "first"
DEFAULT_SLICE_LAST = "last"
VALID_DEFAULT_SLICE = {DEFAULT_SLICE_FIRST, DEFAULT_SLICE_LAST}

DEFAULT_COLORMAP = "gray"
ANGLE_COLORMAP_SAME = "same"
DEFAULT_CHANNEL = "auto"
SUPPORTED_CHANNELS = (
    "auto",
    "real",
    "abs",
    "imag",
    "angle",
)
DEFAULT_ORIGIN = "lower_left"
SUPPORTED_ORIGINS = (
    "upper_left",
    "lower_left",
    "upper_right",
    "lower_right",
)
DEFAULT_DISPLAY_MODE = "square_pixels"
SUPPORTED_DISPLAY_MODES = (
    "square_pixels",
    "square_fov",
    "auto",
)
DISPLAY_MODE_ALIASES = {
    "fit": "auto",
}
DEFAULT_MASK_OPACITY = 0.8
DEFAULT_APPLY_SCALING = True
COLOR_SCHEME_SYSTEM = "system"
COLOR_SCHEME_LIGHT = "light"
COLOR_SCHEME_DARK = "dark"
SUPPORTED_COLOR_SCHEMES = (
    COLOR_SCHEME_SYSTEM,
    COLOR_SCHEME_LIGHT,
    COLOR_SCHEME_DARK,
)


@dataclass(frozen=True)
class ViewerConfig:
    initial_index: str = INITIAL_INDEX_FIRST
    default_slice: str = DEFAULT_SLICE_FIRST
    initial_origin: str = DEFAULT_ORIGIN
    default_channel: str = DEFAULT_CHANNEL
    default_colormap: str = DEFAULT_COLORMAP
    angle_colormap: str = ANGLE_COLORMAP_SAME
    default_display_mode: str = DEFAULT_DISPLAY_MODE
    default_mask_opacity: float = DEFAULT_MASK_OPACITY
    apply_scaling: bool = DEFAULT_APPLY_SCALING
    color_scheme: str = COLOR_SCHEME_SYSTEM


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

    loading = raw.get("loading", {})
    if not isinstance(loading, dict):
        loading = {}

    initial_index = startup.get("initial_index", INITIAL_INDEX_FIRST)
    if initial_index not in VALID_INITIAL_INDEX:
        initial_index = INITIAL_INDEX_FIRST

    default_slice = startup.get("default_slice", DEFAULT_SLICE_FIRST)
    if default_slice not in VALID_DEFAULT_SLICE:
        default_slice = DEFAULT_SLICE_FIRST

    initial_origin = startup.get("initial_origin", DEFAULT_ORIGIN)
    if initial_origin not in SUPPORTED_ORIGINS:
        initial_origin = DEFAULT_ORIGIN

    default_channel = display.get("default_channel", DEFAULT_CHANNEL)
    if default_channel not in SUPPORTED_CHANNELS:
        default_channel = DEFAULT_CHANNEL

    default_colormap = display.get("default_colormap", DEFAULT_COLORMAP)
    if valid_colormaps is not None and default_colormap not in valid_colormaps:
        default_colormap = DEFAULT_COLORMAP

    angle_colormap = display.get("angle_colormap", ANGLE_COLORMAP_SAME)
    valid_angle_colormaps = (
        (ANGLE_COLORMAP_SAME,) + valid_colormaps
        if valid_colormaps is not None
        else None
    )
    if valid_angle_colormaps is not None and angle_colormap not in valid_angle_colormaps:
        angle_colormap = ANGLE_COLORMAP_SAME

    default_display_mode = display.get("default_display_mode", DEFAULT_DISPLAY_MODE)
    default_display_mode = DISPLAY_MODE_ALIASES.get(default_display_mode, default_display_mode)
    if default_display_mode not in SUPPORTED_DISPLAY_MODES:
        default_display_mode = DEFAULT_DISPLAY_MODE

    default_mask_opacity = _clean_mask_opacity(
        display.get("default_mask_opacity", DEFAULT_MASK_OPACITY)
    )
    color_scheme = display.get("color_scheme", COLOR_SCHEME_SYSTEM)
    if color_scheme not in SUPPORTED_COLOR_SCHEMES:
        color_scheme = COLOR_SCHEME_SYSTEM
    apply_scaling = loading.get("apply_scaling", DEFAULT_APPLY_SCALING)
    if not isinstance(apply_scaling, bool):
        apply_scaling = DEFAULT_APPLY_SCALING

    return ViewerConfig(
        initial_index=initial_index,
        default_slice=default_slice,
        initial_origin=initial_origin,
        default_channel=default_channel,
        default_colormap=default_colormap,
        angle_colormap=angle_colormap,
        default_display_mode=default_display_mode,
        default_mask_opacity=default_mask_opacity,
        apply_scaling=apply_scaling,
        color_scheme=color_scheme,
    )


def save_config(config, path=None, colormap_names=None):
    """Write the complete viewer config to TOML."""
    config_path = Path(path) if path is not None else get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    valid_colormaps = tuple(colormap_names) if colormap_names is not None else None

    initial_index = config.initial_index
    if initial_index not in VALID_INITIAL_INDEX:
        initial_index = INITIAL_INDEX_FIRST

    default_slice = config.default_slice
    if default_slice not in VALID_DEFAULT_SLICE:
        default_slice = DEFAULT_SLICE_FIRST

    initial_origin = config.initial_origin
    if initial_origin not in SUPPORTED_ORIGINS:
        initial_origin = DEFAULT_ORIGIN

    default_channel = config.default_channel
    if default_channel not in SUPPORTED_CHANNELS:
        default_channel = DEFAULT_CHANNEL

    default_colormap = config.default_colormap
    if valid_colormaps is not None and default_colormap not in valid_colormaps:
        default_colormap = DEFAULT_COLORMAP

    angle_colormap = config.angle_colormap
    valid_angle_colormaps = (
        (ANGLE_COLORMAP_SAME,) + valid_colormaps
        if valid_colormaps is not None
        else None
    )
    if valid_angle_colormaps is not None and angle_colormap not in valid_angle_colormaps:
        angle_colormap = ANGLE_COLORMAP_SAME

    default_display_mode = config.default_display_mode
    default_display_mode = DISPLAY_MODE_ALIASES.get(default_display_mode, default_display_mode)
    if default_display_mode not in SUPPORTED_DISPLAY_MODES:
        default_display_mode = DEFAULT_DISPLAY_MODE

    default_mask_opacity = _clean_mask_opacity(config.default_mask_opacity)
    color_scheme = config.color_scheme
    if color_scheme not in SUPPORTED_COLOR_SCHEMES:
        color_scheme = COLOR_SCHEME_SYSTEM
    apply_scaling = (
        config.apply_scaling
        if isinstance(config.apply_scaling, bool)
        else DEFAULT_APPLY_SCALING
    )

    text = (
        "[startup]\n"
        f'initial_index = "{initial_index}"\n'
        f'default_slice = "{default_slice}"\n'
        f'initial_origin = "{initial_origin}"\n'
        "\n"
        "[display]\n"
        f'default_channel = "{default_channel}"\n'
        f'default_colormap = "{default_colormap}"\n'
        f'angle_colormap = "{angle_colormap}"\n'
        f'default_display_mode = "{default_display_mode}"\n'
        f"default_mask_opacity = {default_mask_opacity:.6g}\n"
        f'color_scheme = "{color_scheme}"\n'
        "\n"
        "[loading]\n"
        f"apply_scaling = {str(apply_scaling).lower()}\n"
    )
    config_path.write_text(text, encoding="utf-8")


def update_config(path=None, colormap_names=None, **changes):
    config = replace(load_config(path, colormap_names=colormap_names), **changes)
    save_config(config, path, colormap_names=colormap_names)
    return config


def _clean_mask_opacity(value):
    try:
        opacity = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MASK_OPACITY
    if not 0.0 <= opacity <= 1.0:
        return DEFAULT_MASK_OPACITY
    return opacity
