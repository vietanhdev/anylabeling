"""Worker class for exporting annotations in the background."""

import os
import os.path as osp
import json
import random
import uuid
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable
from .export_formats import FormatExporter


class ExportSignals(QObject):
    """Signals for the export worker."""

    started = pyqtSignal()
    finished = pyqtSignal()
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)


class ExportWorker(QRunnable):
    """Worker for exporting annotations in the background."""

    def __init__(
        self,
        export_format,
        input_dir,
        output_dir,
        split_data=False,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        recursive=False,
        use_random_names=False,
    ):
        """Initialize the export worker.

        Args:
            export_format: Format to export to (yolo, pascal_voc, coco, createml)
            input_dir: Input directory containing JSON annotations
            output_dir: Output directory for exported annotations
            split_data: Whether to split data into train/val/test sets
            train_ratio: Ratio of data for training set
            val_ratio: Ratio of data for validation set
            test_ratio: Ratio of data for test set
            recursive: Whether to scan input directory recursively
            use_random_names: Whether to use random UUID4 names for exported items
        """
        super().__init__()
        self.signals = ExportSignals()
        self.export_format = export_format
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.split_data = split_data
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.recursive = recursive
        self.use_random_names = use_random_names
        self.running = False

    def _create_split_dirs(self):
        """Create directories for data splits."""
        if self.split_data:
            os.makedirs(osp.join(self.output_dir, "train"), exist_ok=True)
            os.makedirs(osp.join(self.output_dir, "val"), exist_ok=True)
            os.makedirs(osp.join(self.output_dir, "test"), exist_ok=True)

            if self.export_format == "yolo":
                # Create label directories for YOLO forma
                os.makedirs(osp.join(self.output_dir, "train", "labels"), exist_ok=True)
                os.makedirs(osp.join(self.output_dir, "val", "labels"), exist_ok=True)
                os.makedirs(osp.join(self.output_dir, "test", "labels"), exist_ok=True)

                # Create image directories for YOLO forma
                os.makedirs(osp.join(self.output_dir, "train", "images"), exist_ok=True)
                os.makedirs(osp.join(self.output_dir, "val", "images"), exist_ok=True)
                os.makedirs(osp.join(self.output_dir, "test", "images"), exist_ok=True)
        else:
            if self.export_format == "yolo":
                os.makedirs(osp.join(self.output_dir, "labels"), exist_ok=True)
                os.makedirs(osp.join(self.output_dir, "images"), exist_ok=True)

    def _get_json_files(self):
        """Get all JSON files in the input directory."""
        if not self.recursive:
            # Non-recursive mode: only get files from the top-level directory
            return [
                f
                for f in os.listdir(self.input_dir)
                if f.lower().endswith(".json")
                and osp.isfile(osp.join(self.input_dir, f))
            ]

        # Recursive mode: get files from all subdirectories
        json_files = []
        for root, _, files in os.walk(self.input_dir):
            for f in files:
                if f.lower().endswith(".json"):
                    # Store the relative path from input_dir
                    rel_path = osp.relpath(osp.join(root, f), self.input_dir)
                    json_files.append(rel_path)
        return json_files

    def _load_json_file(self, json_file):
        """Load a JSON annotation file."""
        try:
            with open(osp.join(self.input_dir, json_file), "r") as f:
                return json.load(f)
        except Exception as e:
            self.signals.error.emit(f"Error loading {json_file}: {str(e)}")
            return None

    def _get_corresponding_image_file(self, json_data, json_file):
        """Get the corresponding image file for a JSON annotation file."""
        # First try to use the imagePath from the JSON
        if "imagePath" in json_data:
            image_path = json_data["imagePath"]
            # Check if it's a relative path
            if not osp.isabs(image_path):
                # Create full path relative to the location of the json file
                json_full_path = osp.join(self.input_dir, json_file)
                image_path = osp.join(
                    osp.dirname(json_full_path),
                    image_path,
                )
            if osp.exists(image_path):
                return image_path

        # Otherwise, try to find an image with the same base name
        json_full_path = osp.join(self.input_dir, json_file)
        json_dir = osp.dirname(json_full_path)
        base_name = osp.splitext(osp.basename(json_file))[0]

        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_path = osp.join(json_dir, base_name + ext)
            if osp.exists(image_path):
                return image_path

        return None

    def _split_data(self, json_files):
        """Split the data into train/val/test sets."""
        if not self.split_data:
            return {"all": json_files}

        # Shuffle the files for random spli
        random.shuffle(json_files)
        n_files = len(json_files)

        n_train = int(n_files * self.train_ratio)
        n_val = int(n_files * self.val_ratio)

        train_files = json_files[:n_train]
        val_files = json_files[n_train : n_train + n_val]
        test_files = json_files[n_train + n_val :]

        return {"train": train_files, "val": val_files, "test": test_files}

    def _get_output_path(self, json_file, split):
        """Get the output path for an exported annotation file."""
        # Get the base name (without extension)
        base_name = osp.splitext(osp.basename(json_file))[0]

        # Generate a random name if requested
        if self.use_random_names:
            base_name = str(uuid.uuid4())

        # If recursive mode and json_file has directories, preserve the structure
        subdir = ""
        if self.recursive and osp.dirname(json_file):
            subdir = osp.dirname(json_file)

        if self.split_data:
            if self.export_format == "yolo":
                # For YOLO, we don't preserve subdirectories since it uses a flat structure
                return osp.join(self.output_dir, split, "labels", base_name + ".txt")
            elif self.export_format == "pascal_voc":
                # For Pascal VOC, preserve subdirectories
                if subdir:
                    os.makedirs(osp.join(self.output_dir, split, subdir), exist_ok=True)
                    return osp.join(self.output_dir, split, subdir, base_name + ".xml")
                else:
                    return osp.join(self.output_dir, split, base_name + ".xml")
            else:
                return None  # COCO and CreateML are dataset-wide formats
        else:
            if self.export_format == "yolo":
                # For YOLO, we don't preserve subdirectories since it uses a flat structure
                return osp.join(self.output_dir, "labels", base_name + ".txt")
            elif self.export_format == "pascal_voc":
                # For Pascal VOC, preserve subdirectories
                if subdir:
                    os.makedirs(osp.join(self.output_dir, subdir), exist_ok=True)
                    return osp.join(self.output_dir, subdir, base_name + ".xml")
                else:
                    return osp.join(self.output_dir, base_name + ".xml")
            else:
                return None

    def _export_yolo(self, json_files_by_split):
        """Export annotations to YOLO format."""
        # Create a global label map
        all_labels = set()
        for split, json_files in json_files_by_split.items():
            for json_file in json_files:
                json_data = self._load_json_file(json_file)
                if json_data and "shapes" in json_data:
                    for shape in json_data["shapes"]:
                        all_labels.add(shape["label"])

        labels = sorted(list(all_labels))
        label_map = {label: i for i, label in enumerate(labels)}

        # Save label map
        with open(osp.join(self.output_dir, "classes.txt"), "w") as f:
            f.write("\n".join(labels))

        # Export annotations
        for split, json_files in json_files_by_split.items():
            for i, json_file in enumerate(json_files):
                if not self.running:
                    return

                self.signals.progress.emit(
                    int((i / len(json_files)) * 100),
                    f"Exporting YOLO annotations ({split}): {i + 1}/{len(json_files)}",
                )

                json_data = self._load_json_file(json_file)
                if not json_data or "shapes" not in json_data:
                    continue

                # Get image dimensions
                image_height = json_data.get("imageHeight", 0)
                image_width = json_data.get("imageWidth", 0)

                # Get output path
                output_path = self._get_output_path(json_file, split)
                if not output_path:
                    continue

                # Create directory if it doesn't exis
                os.makedirs(osp.dirname(output_path), exist_ok=True)

                # Export to YOLO forma
                FormatExporter.export_to_yolo(
                    json_data["shapes"],
                    image_height,
                    image_width,
                    label_map,
                    output_path,
                )

                # Copy image file
                image_path = self._get_corresponding_image_file(json_data, json_file)
                if image_path:
                    import shutil

                    # Get file name and extension for the image
                    img_name, img_ext = osp.splitext(osp.basename(image_path))

                    # Generate a random name if requested
                    output_img_name = osp.basename(image_path)
                    if self.use_random_names:
                        # Extract base_name from the output_path
                        random_base = osp.splitext(osp.basename(output_path))[0]
                        output_img_name = random_base + img_ext

                    if self.split_data:
                        image_output_path = osp.join(
                            self.output_dir,
                            split,
                            "images",
                            output_img_name,
                        )
                    else:
                        image_output_path = osp.join(
                            self.output_dir,
                            "images",
                            output_img_name,
                        )
                    shutil.copy2(image_path, image_output_path)

    def _export_pascal_voc(self, json_files_by_split):
        """Export annotations to Pascal VOC format."""
        for split, json_files in json_files_by_split.items():
            split_dir = (
                osp.join(self.output_dir, split) if self.split_data else self.output_dir
            )
            os.makedirs(split_dir, exist_ok=True)

            for i, json_file in enumerate(json_files):
                if not self.running:
                    return

                self.signals.progress.emit(
                    int((i / len(json_files)) * 100),
                    f"Exporting Pascal VOC annotations ({split}): {i + 1}/{len(json_files)}",
                )

                json_data = self._load_json_file(json_file)
                if not json_data or "shapes" not in json_data:
                    continue

                # Get image path and dimensions
                image_path = self._get_corresponding_image_file(json_data, json_file)
                if not image_path:
                    continue

                image_height = json_data.get("imageHeight", 0)
                image_width = json_data.get("imageWidth", 0)

                # Get output path
                base_name = osp.splitext(json_file)[0]
                # Apply random name if requested
                if self.use_random_names:
                    base_name = str(uuid.uuid4())
                output_path = osp.join(split_dir, base_name + ".xml")

                # Export to Pascal VOC forma
                FormatExporter.export_to_pascal_voc(
                    json_data["shapes"],
                    image_path,
                    image_height,
                    image_width,
                    output_path,
                )

                # Copy image file
                if image_path:
                    import shutil

                    # Get file name and extension for the image
                    img_name, img_ext = osp.splitext(osp.basename(image_path))

                    # Generate a random name if requested
                    output_img_name = osp.basename(image_path)
                    if self.use_random_names:
                        # Extract base_name from the output_path
                        random_base = osp.splitext(osp.basename(output_path))[0]
                        output_img_name = random_base + img_ext

                    image_output_path = osp.join(split_dir, output_img_name)
                    shutil.copy2(image_path, image_output_path)

    def _export_coco(self, json_files_by_split):
        """Export annotations to COCO format."""
        for split, json_files in json_files_by_split.items():
            if not json_files:
                continue

            split_dir = (
                osp.join(self.output_dir, split) if self.split_data else self.output_dir
            )
            os.makedirs(split_dir, exist_ok=True)

            # Create lists to collect all image data
            all_shapes = []
            all_image_paths = []
            all_image_heights = []
            all_image_widths = []

            self.signals.progress.emit(
                0, f"Loading annotations for COCO export ({split})"
            )

            # Load all JSON files for this spli
            for i, json_file in enumerate(json_files):
                if not self.running:
                    return

                self.signals.progress.emit(
                    int(
                        (i / len(json_files)) * 50
                    ),  # Use first half of progress for loading
                    f"Loading annotations for COCO export ({split}): {i + 1}/{len(json_files)}",
                )

                json_data = self._load_json_file(json_file)
                if not json_data or "shapes" not in json_data:
                    continue

                # Get image path and dimensions
                image_path = self._get_corresponding_image_file(json_data, json_file)
                if not image_path:
                    continue

                image_height = json_data.get("imageHeight", 0)
                image_width = json_data.get("imageWidth", 0)

                all_shapes.append(json_data["shapes"])
                all_image_paths.append(image_path)
                all_image_heights.append(image_height)
                all_image_widths.append(image_width)

                # Copy image file
                import shutil

                # Get file name and extension for the image
                img_name, img_ext = osp.splitext(osp.basename(image_path))

                # Generate a random name if requested
                output_img_name = osp.basename(image_path)
                if self.use_random_names:
                    random_base = str(uuid.uuid4())
                    output_img_name = random_base + img_ext
                    # Update the image path in all_image_paths for the export formats
                    all_image_paths[-1] = output_img_name

                image_output_path = osp.join(split_dir, output_img_name)
                shutil.copy2(image_path, image_output_path)

            self.signals.progress.emit(
                50,  # Start second half of progress
                f"Exporting COCO annotations ({split})",
            )

            # Export to COCO forma
            output_path = osp.join(split_dir, "annotations.json")
            FormatExporter.export_to_coco(
                all_shapes,
                all_image_paths,
                all_image_heights,
                all_image_widths,
                output_path,
            )

            self.signals.progress.emit(100, f"Completed COCO export ({split})")

    def _export_createml(self, json_files_by_split):
        """Export annotations to CreateML format."""
        for split, json_files in json_files_by_split.items():
            if not json_files:
                continue

            split_dir = (
                osp.join(self.output_dir, split) if self.split_data else self.output_dir
            )
            os.makedirs(split_dir, exist_ok=True)

            # Create lists to collect all image data
            all_shapes = []
            all_image_paths = []
            all_image_heights = []
            all_image_widths = []

            self.signals.progress.emit(
                0, f"Loading annotations for CreateML export ({split})"
            )

            # Load all JSON files for this spli
            for i, json_file in enumerate(json_files):
                if not self.running:
                    return

                self.signals.progress.emit(
                    int(
                        (i / len(json_files)) * 50
                    ),  # Use first half of progress for loading
                    f"Loading annotations for CreateML export ({split}): {i + 1}/{len(json_files)}",
                )

                json_data = self._load_json_file(json_file)
                if not json_data or "shapes" not in json_data:
                    continue

                # Get image path and dimensions
                image_path = self._get_corresponding_image_file(json_data, json_file)
                if not image_path:
                    continue

                image_height = json_data.get("imageHeight", 0)
                image_width = json_data.get("imageWidth", 0)

                all_shapes.append(json_data["shapes"])
                all_image_paths.append(image_path)
                all_image_heights.append(image_height)
                all_image_widths.append(image_width)

                # Copy image file
                import shutil

                # Get file name and extension for the image
                img_name, img_ext = osp.splitext(osp.basename(image_path))

                # Generate a random name if requested
                output_img_name = osp.basename(image_path)
                if self.use_random_names:
                    random_base = str(uuid.uuid4())
                    output_img_name = random_base + img_ext
                    # Update the image path in all_image_paths for the export formats
                    all_image_paths[-1] = output_img_name

                image_output_path = osp.join(split_dir, output_img_name)
                shutil.copy2(image_path, image_output_path)

            self.signals.progress.emit(
                50,  # Start second half of progress
                f"Exporting CreateML annotations ({split})",
            )

            # Export to CreateML forma
            output_path = osp.join(split_dir, "annotations.json")
            FormatExporter.export_to_createml(
                all_shapes,
                all_image_paths,
                all_image_heights,
                all_image_widths,
                output_path,
            )

            self.signals.progress.emit(100, f"Completed CreateML export ({split})")

    @pyqtSlot()
    def run(self):
        """Run the export process."""
        self.running = True
        self.signals.started.emit()

        try:
            # Create output directory if it doesn't exis
            os.makedirs(self.output_dir, exist_ok=True)

            # Create split directories if needed
            self._create_split_dirs()

            # Get all JSON files
            json_files = self._get_json_files()
            if not json_files:
                self.signals.error.emit(
                    "No JSON annotation files found in the input directory"
                )
                self.signals.finished.emit()
                self.running = False
                return

            # Split the data if needed
            json_files_by_split = self._split_data(json_files)

            # Export annotations based on forma
            if self.export_format == "yolo":
                self._export_yolo(json_files_by_split)
            elif self.export_format == "pascal_voc":
                self._export_pascal_voc(json_files_by_split)
            elif self.export_format == "coco":
                self._export_coco(json_files_by_split)
            elif self.export_format == "createml":
                self._export_createml(json_files_by_split)
            else:
                self.signals.error.emit(
                    f"Unsupported export format: {self.export_format}"
                )

        except Exception as e:
            self.signals.error.emit(f"Export error: {str(e)}")
        finally:
            self.signals.finished.emit()
            self.running = False

    def stop(self):
        """Stop the export process."""
        self.running = False
