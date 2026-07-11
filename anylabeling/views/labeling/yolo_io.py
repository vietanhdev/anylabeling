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

    Supports two YOLO formats — detection (bounding box) and segmentation
    (polygon) — detected by the number of values on each line:

    * **Detection** (5 values)::

        <class_id> <x_center> <y_center> <width> <height>

    * **Segmentation** (7+ values, even coordinate count)::

        <class_id> <x1> <y1> <x2> <y2> … <xn> <yn>

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

            label = id_to_label.get(class_id)
            if label is None:
                label = f"class_{class_id}"

            if label_to_id is not None and label not in label_to_id:
                label_to_id[label] = class_id

            # Detection format (exactly 5 values) vs segmentation (7+ values)
            if len(parts) == 5:
                # YOLO detection: class_id x_center y_center width height
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

                shape = {
                    "label": label,
                    "text": "",
                    "points": [[x1, y1], [x2, y2]],
                    "shape_type": "rectangle",
                    "group_id": None,
                    "flags": {},
                    "other_data": {},
                }
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                # YOLO segmentation: class_id x1 y1 x2 y2 … xn yn
                coords = [float(v) for v in parts[1:]]
                points = [
                    [coords[i] * img_w, coords[i + 1] * img_h]
                    for i in range(0, len(coords), 2)
                ]
                shape = {
                    "label": label,
                    "text": "",
                    "points": points,
                    "shape_type": "polygon",
                    "group_id": None,
                    "flags": {},
                    "other_data": {},
                }
            else:
                continue

            shapes.append(shape)

    return shapes


def write_yolo_label(txt_path, shapes, img_w, img_h, label_to_id):
    """Write AnyLabeling shape dicts back to a YOLO-format .txt file.

    *label_to_id* is mutated in-place: labels that are not already
    present receive the next unused integer class ID so that saving
    never silently drops annotations.

    Output format is *mixed-mode*:
    * Rectangles → YOLO detection format (5 values)
    * Polygons   → YOLO segmentation format (variable values)
    * Other shape types are skipped (not representable in YOLO format)
    """
    used_ids = set(label_to_id.values())
    next_id = 0
    if used_ids:
        next_id = max(used_ids) + 1

    lines = []
    for shape in shapes:
        shape_type = shape["shape_type"]
        points = shape["points"]

        label = shape["label"]

        if label not in label_to_id:
            while next_id in used_ids:
                next_id += 1
            label_to_id[label] = next_id
            used_ids.add(next_id)
            next_id += 1

        class_id = label_to_id[label]

        if shape_type == "rectangle" and len(points) >= 2:
            # YOLO detection: class_id x_center y_center width height
            x1, y1 = points[0]
            x2, y2 = points[1]

            x_center = ((x1 + x2) / 2.0) / img_w
            y_center = ((y1 + y2) / 2.0) / img_h
            width = abs(x2 - x1) / img_w
            height = abs(y2 - y1) / img_h

            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        elif shape_type == "polygon" and len(points) >= 3:
            # YOLO segmentation: class_id x1 y1 x2 y2 … xn yn
            normalized = []
            for x, y in points:
                normalized.append(f"{x / img_w:.6f}")
                normalized.append(f"{y / img_h:.6f}")
            lines.append(f"{class_id} {' '.join(normalized)}")

        # Circles, lines, linestrips, points — not representable in YOLO

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def find_dataset_config(dirpath):
    """Walk up from dirpath looking for a YOLO dataset config file.

    Checks, in order:
    1.  ``data.yaml``  (YOLO dataset descriptor, YAML format)
    2.  ``classes.txt`` (one class name per line, plain text)

    Searches up to 5 parent directories.  Returns ``(path, type)``
    where *type* is ``"yaml"`` or ``"txt"``, or ``(None, None)``.
    """
    current = osp.abspath(dirpath)
    for _ in range(5):
        for filename, cfg_type in (("data.yaml", "yaml"), ("classes.txt", "txt")):
            candidate = osp.join(current, filename)
            if osp.exists(candidate):
                return candidate, cfg_type
        parent = osp.dirname(current)
        if parent == current:
            break
        current = parent
    return None, None


def read_classes_txt(txt_path):
    """Parse a classes.txt file (one class name per line).

    Returns ``(nc, names_list, id_to_label_dict)``.
    """
    names = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                names.append(name)
    nc = len(names)
    id_to_label = {i: name for i, name in enumerate(names)}
    return nc, names, id_to_label


def update_dataset_config(config_path, config_type, label_to_id):
    """Write updated class names back to a YOLO dataset config file.

    * For ``"yaml"``: only the ``nc:`` and ``names:`` lines are
      replaced in-place via regex — every other line (comments,
      train/val paths, roboflow metadata, blank lines) is preserved
      character-for-character.

    * For ``"txt"``: the file is rewritten with one class name per
      line in class-ID order.

    *label_to_id* is a ``{name: class_id}`` mapping.
    """
    if not label_to_id:
        return
    max_id = max(label_to_id.values())
    id_to_label = {}
    for name, cid in label_to_id.items():
        id_to_label[cid] = name
    ordered_names = [id_to_label.get(i, "") for i in range(max_id + 1)]
    for i, name in enumerate(ordered_names):
        if not name:
            ordered_names[i] = f"class_{i}"

    if config_type == "yaml":
        import re

        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = re.sub(
            r"^nc:\s*\d+",
            f"nc: {len(ordered_names)}",
            content,
            flags=re.MULTILINE,
        )
        names_str = repr(ordered_names)
        content = re.sub(
            r"^names:\s*\[.*?\]|^names:\s*\n(\s+- .*\n?)*",
            f"names: {names_str}",
            content,
            flags=re.MULTILINE | re.DOTALL,
        )
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            for name in ordered_names:
                f.write(name + "\n")


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
