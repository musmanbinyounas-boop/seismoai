"""SeismoAI I/O module — load and prepare SGY seismic files."""

from .io_core import load_sgy, load_folder, normalize_traces

__all__ = ["load_sgy", "load_folder", "normalize_traces"]
__version__ = "0.1.0"
