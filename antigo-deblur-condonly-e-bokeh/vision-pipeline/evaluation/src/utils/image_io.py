import io

from PIL import Image


def get_pil_image(data) -> Image.Image:
    """Return a PIL image from a HuggingFace image cell.

    Accepts an already-decoded PIL image, raw PNG/JPEG bytes, or the
    ``{"bytes": ..., "path": ...}`` dict produced by the ``datasets`` Image
    feature.
    """
    if hasattr(data, "convert"):
        return data
    elif isinstance(data, bytes):
        return Image.open(io.BytesIO(data))
    elif isinstance(data, dict):
        if "bytes" in data and data["bytes"] is not None:
            return Image.open(io.BytesIO(data["bytes"]))
        elif "path" in data and data["path"] is not None:
            return Image.open(data["path"])
    raise ValueError(f"Unrecognized image data format: {type(data)}")
