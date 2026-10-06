import io

from PIL import Image

PIXEL_MULTIPLE = 16


def load_image_from_row(data) -> Image.Image:
    """Return a PIL image from a HuggingFace parquet image cell.

    Accepts raw PNG/JPEG bytes, the ``{"bytes": ...}`` dict produced by the
    ``datasets`` Image feature, or an already-decoded PIL image.
    """
    if isinstance(data, dict) and "bytes" in data:
        return Image.open(io.BytesIO(data["bytes"]))
    elif isinstance(data, bytes):
        return Image.open(io.BytesIO(data))
    return data


def resize_and_pad_image(img: Image.Image, target_long_side: int) -> Image.Image:
    """
    Resize an image and align width and height to multiples of 16.
    """
    width, height = img.size
    if target_long_side and target_long_side > 0:
        target_max = int(target_long_side)
        if width >= height:
            resized_width = target_max
            scale = target_max / width
            resized_height = int(height * scale)
        else:
            resized_height = target_max
            scale = target_max / height
            resized_width = int(width * scale)

        img = img.resize((resized_width, resized_height), Image.LANCZOS)
        final_width = max(
            (resized_width // PIXEL_MULTIPLE) * PIXEL_MULTIPLE, PIXEL_MULTIPLE
        )
        final_height = max(
            (resized_height // PIXEL_MULTIPLE) * PIXEL_MULTIPLE, PIXEL_MULTIPLE
        )
        left = (resized_width - final_width) // 2
        top = (resized_height - final_height) // 2
        return img.crop((left, top, left + final_width, top + final_height))

    final_width = ((width + PIXEL_MULTIPLE - 1) // PIXEL_MULTIPLE) * PIXEL_MULTIPLE
    final_height = ((height + PIXEL_MULTIPLE - 1) // PIXEL_MULTIPLE) * PIXEL_MULTIPLE
    if final_width == width and final_height == height:
        return img
    return img.resize((final_width, final_height), Image.LANCZOS)
