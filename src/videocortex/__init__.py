"""videocortex — drop a clip, see what it would do to an average brain.

A thin, opinionated pipeline around Meta's TRIBE v2 brain encoder
(https://github.com/facebookresearch/tribev2). Upstream owns the model;
this package owns everything around it: device selection that knows Apple
Silicon exists, a preflight check that fails loudly instead of three
gigabytes into a download, and a renderer that turns the raw
(n_timesteps x n_vertices) matrix into brain-map stills you can actually
put in a slide.
"""

__version__ = "0.1.0"

from videocortex.config import OverlayConfig, RenderConfig, RunConfig, VIEW_PRESETS
from videocortex.device import describe_device, resolve_device, select_device

__all__ = [
    "__version__",
    "OverlayConfig",
    "RenderConfig",
    "RunConfig",
    "VIEW_PRESETS",
    "describe_device",
    "resolve_device",
    "select_device",
]
