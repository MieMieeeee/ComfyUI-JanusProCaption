from .janus_pro_caption import JanusProModelLoader, JanusProDescribeImage
from .utils import add_suffix, add_emoji

NODE_CLASS_MAPPINGS = {
    add_suffix("JanusProModelLoader"): JanusProModelLoader,
    add_suffix("JanusProDescribeImage"): JanusProDescribeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    add_suffix("JanusProModelLoader"): add_emoji("Janus Pro Model Loader"),
    add_suffix("JanusProDescribeImage"): add_emoji("Janus Pro Describe Image"),
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
