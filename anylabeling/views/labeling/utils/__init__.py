# flake8: noqa

from ._io import lblsave
from .image import (
    apply_exif_orientation,
    img_arr_to_b64,
    img_b64_to_arr,
    img_data_to_arr,
    img_data_to_pil,
    img_data_to_png_data,
    img_pil_to_data,
)
from .qt import (
    Struct,
    add_actions,
    distance,
    distance_to_line,
    squared_distance_to_line,
    fmt_shortcut,
    label_validator,
    new_action,
    new_button,
    new_icon,
)


def encode_rle(data):
    """Encode data using Run-Length Encoding"""
    if len(data) == 0:
        return []
    res = []
    current_val = data[0]
    current_count = 0
    for val in data:
        if val == current_val:
            current_count += 1
        else:
            res.extend([int(current_val), int(current_count)])
            current_val = val
            current_count = 1
    res.extend([int(current_val), int(current_count)])
    return res


def decode_rle(rle):
    """Decode data using Run-Length Encoding"""
    res = []
    for i in range(0, len(rle), 2):
        val = rle[i]
        count = rle[i + 1]
        res.extend([val] * count)
    return res


from .shape import (
    masks_to_bboxes,
    polygons_to_mask,
    shape_to_mask,
    shapes_to_label,
)

# Export utilities
from .export_formats import FormatExporter
from .export_worker import ExportSignals, ExportWorker
