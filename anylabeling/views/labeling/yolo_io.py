"""Utilities for reading and writing YOLO-format annotations."""

import os
import os.path as osp


def find_dataset_yaml(dirpath):
    """Walk up from dirpath looking for a data.yaml file.

    Searches up to 5 parent directories. Returns the full path to
    data.yaml if found, None otherwise.
    """
    current = osp.abspath(dirpath)
    for _ in range(5):
        candidate = osp.join(current, "data.yaml")
        if osp.exists(candidate):
            return candidate
        parent = osp.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def read_dataset_yaml(yaml_path):
    """Parse a YOLO data.yaml file and return class name information.

    Returns (nc, names_list, id_to_label_dict).  Any of the three
    return values may be empty / 0 when the key is absent from the
    YAML file.
    """
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    nc = data.get("nc", 0)
    raw_names = data.get("names", [])

    if isinstance(raw_names, list):
        id_to_label = {i: name for i, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        id_to_label = {int(k): v for k, v in raw_names.items()}
    else:
        id_to_label = {}
    return nc, raw_names, id_to_label


def read_yolo_label(txt_path, img_w, img_h, id_to_label, label_to_id=None):
    """Parse a YOLO-format .txt label file into AnyLabeling shape dicts.

    Each line in the file is expected to follow the YOLO detection
    format::

        <class_id> <x_center> <y_center> <width> <height>

    Coordinates are normalised (0-1).  Returns a list of dicts ready
    to be passed to ``LabelingWidget.load_labels()``.

    When *label_to_id* is provided the dict will be updated in-place
    so that every label string that appears in the file maps back to
    its original class ID, ensuring round-trip fidelity.
    """
    shapes = []
    if not osp.exists(txt_path):
        return shapes

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Normalised → absolute pixel coords
            w_abs = width * img_w
            h_abs = height * img_h
            cx_abs = x_center * img_w
            cy_abs = y_center * img_h

            x1 = cx_abs - w_abs / 2.0
            y1 = cy_abs - h_abs / 2.0
            x2 = cx_abs + w_abs / 2.0
            y2 = cy_abs + h_abs / 2.0

            label = id_to_label.get(class_id)
            if label is None:
                label = f"class_{class_id}"

            if label_to_id is not None and label not in label_to_id:
                label_to_id[label] = class_id

            shape = {
                "label": label,
                "text": "",
                "points": [[x1, y1], [x2, y2]],
                "shape_type": "rectangle",
                "group_id": None,
                "flags": {},
                "other_data": {},
            }
            shapes.append(shape)

    return shapes


def write_yolo_label(txt_path, shapes, img_w, img_h, label_to_id):
    """Write AnyLabeling shape dicts back to a YOLO-format .txt file.

    *label_to_id* is mutated in-place: labels that are not already
    present receive the next unused integer class ID so that saving
    never silently drops annotations.
    """
    used_ids = set(label_to_id.values())
    next_id = 0
    if used_ids:
        next_id = max(used_ids) + 1

    lines = []
    for shape in shapes:
        if shape["shape_type"] not in ("rectangle",):
            continue
        points = shape["points"]
        if len(points) < 2:
            continue

        label = shape["label"]

        if label not in label_to_id:
            while next_id in used_ids:
                next_id += 1
            label_to_id[label] = next_id
            used_ids.add(next_id)
            next_id += 1

        class_id = label_to_id[label]

        # Absolute pixel coords → normalised YOLO format
        x1, y1 = points[0]
        x2, y2 = points[1]

        x_center = ((x1 + x2) / 2.0) / img_w
        y_center = ((y1 + y2) / 2.0) / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def resolve_yolo_label_path(image_path):
    """Locate the YOLO .txt label file that corresponds to *image_path*.

    Checks, in order:
    1.  Same directory as the image (``<name>.txt``)
    2.  A sibling ``labels/`` directory (``../labels/<name>.txt``)

    Returns the absolute path to the .txt file or ``None``.
    """
    img_dir = osp.dirname(image_path)
    base_stem = osp.splitext(osp.basename(image_path))[0]

    # 1. Side-by-side: same directory
    candidate = osp.join(img_dir, base_stem + ".txt")
    if osp.exists(candidate):
        return candidate

    # 2. Standard YOLO layout: ../labels/<name>.txt
    parent = osp.dirname(img_dir)
    candidate = osp.join(parent, "labels", base_stem + ".txt")
    if osp.exists(candidate):
        return candidate

    return None
