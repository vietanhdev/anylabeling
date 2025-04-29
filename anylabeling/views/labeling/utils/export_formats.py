"""Utilities for exporting annotations to different formats."""

import json
import os.path as osp
import xml.etree.ElementTree as ET
from xml.dom import minidom


from anylabeling.app_info import __version__


class FormatExporter:
    """A utility class for exporting annotations to different formats."""

    @staticmethod
    def export_to_yolo(
        shapes, image_height, image_width, label_map=None, output_path=None
    ):
        """Export annotations to YOLO format.

        Args:
            shapes: List of annotation shapes
            image_height: Height of the image
            image_width: Width of the image
            label_map: Dictionary mapping labels to class indices
            output_path: Path to save the YOLO annotations

        Returns:
            Exported YOLO annotations as string
        """
        if label_map is None:
            # Create label map from shapes
            labels = sorted(list(set(shape["label"] for shape in shapes)))
            label_map = {label: i for i, label in enumerate(labels)}

        results = []
        for shape in shapes:
            if shape["shape_type"] != "rectangle" and shape["shape_type"] != "polygon":
                continue

            label = shape["label"]
            if label not in label_map:
                continue

            class_idx = label_map[label]
            points = shape["points"]

            if shape["shape_type"] == "rectangle":
                # Convert rectangle to YOLO format [x_center, y_center, width, height]
                x1, y1 = points[0]
                x2, y2 = points[1]
                x_center = (x1 + x2) / (2 * image_width)
                y_center = (y1 + y2) / (2 * image_height)
                width = abs(x2 - x1) / image_width
                height = abs(y2 - y1) / image_height
                results.append(f"{class_idx} {x_center} {y_center} {width} {height}")
            elif shape["shape_type"] == "polygon":
                # For polygons, convert to bbox firs
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                x_center = (x_min + x_max) / (2 * image_width)
                y_center = (y_min + y_max) / (2 * image_height)
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height
                results.append(f"{class_idx} {x_center} {y_center} {width} {height}")

        result_text = "\n".join(results)
        if output_path:
            with open(output_path, "w") as f:
                f.write(result_text)

        return result_text, label_map

    @staticmethod
    def export_to_pascal_voc(
        shapes, image_path, image_height, image_width, output_path=None
    ):
        """Export annotations to Pascal VOC format.

        Args:
            shapes: List of annotation shapes
            image_path: Path to the image
            image_height: Height of the image
            image_width: Width of the image
            output_path: Path to save the Pascal VOC annotations

        Returns:
            Exported Pascal VOC annotations as string
        """
        image_name = osp.basename(image_path)

        # Create XML structure
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = osp.dirname(image_path)
        ET.SubElement(annotation, "filename").text = image_name
        ET.SubElement(annotation, "path").text = image_path

        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Unknown"

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(image_width)
        ET.SubElement(size, "height").text = str(image_height)
        ET.SubElement(size, "depth").text = "3"

        ET.SubElement(annotation, "segmented").text = "0"

        # Add objects
        for shape in shapes:
            if shape["shape_type"] != "rectangle" and shape["shape_type"] != "polygon":
                continue

            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = shape["label"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            points = shape["points"]
            if shape["shape_type"] == "rectangle":
                x1, y1 = points[0]
                x2, y2 = points[1]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
            else:  # polygon
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(x_min))
            ET.SubElement(bndbox, "ymin").text = str(int(y_min))
            ET.SubElement(bndbox, "xmax").text = str(int(x_max))
            ET.SubElement(bndbox, "ymax").text = str(int(y_max))

        # Convert to string with pretty formatting
        xml_str = ET.tostring(annotation, encoding="utf-8")
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")

        if output_path:
            with open(output_path, "w") as f:
                f.write(pretty_xml)

        return pretty_xml

    @staticmethod
    def export_to_coco(
        shapes, image_paths, image_heights, image_widths, output_path=None
    ):
        """Export annotations to COCO format.

        Args:
            shapes: List of annotation shapes per image
            image_paths: List of image paths
            image_heights: List of image heights
            image_widths: List of image widths
            output_path: Path to save the COCO annotations

        Returns:
            Exported COCO annotations as dictionary
        """
        # Initialize COCO structure
        coco_dict = {
            "info": {
                "description": "Dataset exported from AnyLabeling",
                "url": "",
                "version": __version__,
                "year": 2023,
                "contributor": "AnyLabeling",
                "date_created": "",
            },
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        # Create categories from unique labels across all shapes
        all_labels = set()
        for image_shapes in shapes:
            for shape in image_shapes:
                all_labels.add(shape["label"])

        categories = sorted(list(all_labels))
        for i, category in enumerate(categories):
            coco_dict["categories"].append(
                {
                    "id": i + 1,
                    "name": category,
                    "supercategory": "none",
                }
            )

        # Map category names to ids
        category_map = {cat["name"]: cat["id"] for cat in coco_dict["categories"]}

        annotation_id = 1

        # Process each image
        for image_idx, (
            image_path,
            image_height,
            image_width,
            image_shapes,
        ) in enumerate(zip(image_paths, image_heights, image_widths, shapes)):
            image_id = image_idx + 1
            image_name = osp.basename(image_path)

            # Add image info
            coco_dict["images"].append(
                {
                    "id": image_id,
                    "file_name": image_name,
                    "width": image_width,
                    "height": image_height,
                    "license": 1,
                    "date_captured": "",
                }
            )

            # Add annotations
            for shape in image_shapes:
                if shape["shape_type"] not in ["rectangle", "polygon"]:
                    continue

                category_id = category_map[shape["label"]]
                points = shape["points"]

                if shape["shape_type"] == "rectangle":
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)

                    width = x_max - x_min
                    height = y_max - y_min
                    bbox = [x_min, y_min, width, height]

                    # Convert rectangle to segmentation forma
                    segmentation = [
                        [
                            x_min,
                            y_min,
                            x_max,
                            y_min,
                            x_max,
                            y_max,
                            x_min,
                            y_max,
                        ]
                    ]

                    area = width * height

                elif shape["shape_type"] == "polygon":
                    # Flatten points for segmentation
                    segmentation = [[coord for point in points for coord in point]]

                    # Calculate bbox
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                    # Calculate polygon area
                    area = 0
                    for i in range(len(points)):
                        x1, y1 = points[i]
                        x2, y2 = points[(i + 1) % len(points)]
                        area += 0.5 * abs(x1 * y2 - x2 * y1)

                coco_dict["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": segmentation,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0,
                    }
                )

                annotation_id += 1

        if output_path:
            with open(output_path, "w") as f:
                json.dump(coco_dict, f, indent=2)

        return coco_dict

    @staticmethod
    def export_to_createml(
        shapes, image_paths, image_heights, image_widths, output_path=None
    ):
        """Export annotations to CreateML format.

        Args:
            shapes: List of annotation shapes per image
            image_paths: List of image paths
            image_heights: List of image heights
            image_widths: List of image widths
            output_path: Path to save the CreateML annotations

        Returns:
            Exported CreateML annotations as list of dictionaries
        """
        createml_list = []

        # Process each image
        for image_path, image_height, image_width, image_shapes in zip(
            image_paths, image_heights, image_widths, shapes
        ):
            image_name = osp.basename(image_path)

            image_data = {"image": image_name, "annotations": []}

            for shape in image_shapes:
                if shape["shape_type"] not in ["rectangle", "polygon"]:
                    continue

                label = shape["label"]
                points = shape["points"]

                if shape["shape_type"] == "rectangle":
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)

                    width = x_max - x_min
                    height = y_max - y_min

                    annotation = {
                        "label": label,
                        "coordinates": {
                            "x": x_min,
                            "y": y_min,
                            "width": width,
                            "height": height,
                        },
                    }

                elif shape["shape_type"] == "polygon":
                    # Calculate bbox
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    width = x_max - x_min
                    height = y_max - y_min

                    annotation = {
                        "label": label,
                        "coordinates": {
                            "x": x_min,
                            "y": y_min,
                            "width": width,
                            "height": height,
                        },
                    }

                image_data["annotations"].append(annotation)

            createml_list.append(image_data)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(createml_list, f, indent=2)

        return createml_list
