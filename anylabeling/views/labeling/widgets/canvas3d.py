import numpy as np
import pyvista as pv
import vtk
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from .. import utils
from ..shape import Shape


class ViewControls3D(QWidget):
    """Collapsible controls panel for 3D mesh viewing options"""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- Mode toggle group ---
        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(2)

        self.mode_btn_group = QButtonGroup(self)
        self.mode_btn_group.setExclusive(True)

        self.view_btn = QPushButton("View")
        self.view_btn.setCheckable(True)
        self.view_btn.setChecked(True)
        self.view_btn.setFixedHeight(28)

        self.brush_btn = QPushButton("Brush")
        self.brush_btn.setCheckable(True)
        self.brush_btn.setFixedHeight(28)

        self.keypoint_btn = QPushButton("Keypoint")
        self.keypoint_btn.setCheckable(True)
        self.keypoint_btn.setFixedHeight(28)

        self.mode_btn_group.addButton(self.view_btn, 0)
        self.mode_btn_group.addButton(self.brush_btn, 1)
        self.mode_btn_group.addButton(self.keypoint_btn, 2)

        mode_layout.addWidget(self.view_btn)
        mode_layout.addWidget(self.brush_btn)
        mode_layout.addWidget(self.keypoint_btn)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        self.mode_btn_group.idClicked.connect(self._on_mode_clicked)

        # --- Brush settings group ---
        brush_group = QGroupBox("Brush")
        brush_layout = QVBoxLayout()
        brush_layout.setSpacing(4)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size:"))
        self.brush_size_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(1, 100)
        self.brush_size_slider.setValue(20)
        self.brush_size_slider.valueChanged.connect(self._on_brush_size)
        size_row.addWidget(self.brush_size_slider)
        self.brush_size_label = QLabel("2.0%")
        self.brush_size_label.setFixedWidth(42)
        size_row.addWidget(self.brush_size_label)
        brush_layout.addLayout(size_row)

        brush_group.setLayout(brush_layout)
        layout.addWidget(brush_group)

        # --- Display group ---
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout()
        display_layout.setSpacing(4)

        # Representation mode
        rep_row = QHBoxLayout()
        rep_row.addWidget(QLabel("Style:"))
        self.rep_combo = QComboBox()
        self.rep_combo.addItems(["Surface", "Wireframe", "Points", "Surface + Edges"])
        self.rep_combo.currentTextChanged.connect(self._on_representation)
        rep_row.addWidget(self.rep_combo)
        display_layout.addLayout(rep_row)

        # Show edges
        self.edges_cb = QCheckBox("Show edges")
        self.edges_cb.toggled.connect(self._on_edges_toggled)
        display_layout.addWidget(self.edges_cb)

        # Double-sided rendering
        self.doublesided_cb = QCheckBox("Double-sided")
        self.doublesided_cb.setChecked(True)
        self.doublesided_cb.toggled.connect(self._on_doublesided)
        display_layout.addWidget(self.doublesided_cb)

        # Smooth shading
        self.smooth_cb = QCheckBox("Smooth shading")
        self.smooth_cb.setChecked(True)
        self.smooth_cb.toggled.connect(self._on_smooth_shading)
        display_layout.addWidget(self.smooth_cb)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # --- Material group ---
        material_group = QGroupBox("Material")
        material_layout = QVBoxLayout()
        material_layout.setSpacing(4)

        # Opacity slider
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        opacity_row.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("100%")
        self.opacity_label.setFixedWidth(36)
        opacity_row.addWidget(self.opacity_label)
        material_layout.addLayout(opacity_row)

        # Metallic slider
        metallic_row = QHBoxLayout()
        metallic_row.addWidget(QLabel("Metallic:"))
        self.metallic_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.metallic_slider.setRange(0, 100)
        self.metallic_slider.setValue(10)
        self.metallic_slider.valueChanged.connect(self._on_metallic)
        metallic_row.addWidget(self.metallic_slider)
        self.metallic_label = QLabel("0.10")
        self.metallic_label.setFixedWidth(36)
        metallic_row.addWidget(self.metallic_label)
        material_layout.addLayout(metallic_row)

        # Roughness slider
        roughness_row = QHBoxLayout()
        roughness_row.addWidget(QLabel("Roughness:"))
        self.roughness_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.roughness_slider.setRange(0, 100)
        self.roughness_slider.setValue(50)
        self.roughness_slider.valueChanged.connect(self._on_roughness)
        roughness_row.addWidget(self.roughness_slider)
        self.roughness_label = QLabel("0.50")
        self.roughness_label.setFixedWidth(36)
        roughness_row.addWidget(self.roughness_label)
        material_layout.addLayout(roughness_row)

        # Color preset
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(
            ["White", "Light Gray", "Beige", "Light Blue", "Cyan", "Gold"]
        )
        self.color_combo.currentTextChanged.connect(self._on_color)
        color_row.addWidget(self.color_combo)
        material_layout.addLayout(color_row)

        material_group.setLayout(material_layout)
        layout.addWidget(material_group)

        # --- Camera group ---
        camera_group = QGroupBox("Camera")
        camera_layout = QVBoxLayout()
        camera_layout.setSpacing(4)

        # View preset buttons
        view_row1 = QHBoxLayout()
        for label, view in [
            ("+X", (1, 0, 0)),
            ("-X", (-1, 0, 0)),
            ("+Y", (0, 1, 0)),
            ("-Y", (0, -1, 0)),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.clicked.connect(
                lambda _, v=view: self.canvas.set_viewup_and_position(v)
            )
            view_row1.addWidget(btn)
        camera_layout.addLayout(view_row1)

        view_row2 = QHBoxLayout()
        for label, view in [
            ("+Z", (0, 0, 1)),
            ("-Z", (0, 0, -1)),
            ("Iso", "iso"),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.clicked.connect(
                lambda _, v=view: self.canvas.set_viewup_and_position(v)
            )
            view_row2.addWidget(btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setFixedHeight(24)
        reset_btn.clicked.connect(self.canvas.reset_camera)
        view_row2.addWidget(reset_btn)
        camera_layout.addLayout(view_row2)

        # Projection
        self.parallel_cb = QCheckBox("Parallel projection")
        self.parallel_cb.toggled.connect(self._on_parallel_projection)
        camera_layout.addWidget(self.parallel_cb)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        layout.addStretch()
        self.setLayout(layout)

    # --- Callbacks ---

    def _on_representation(self, text):
        self.canvas.set_representation(text)

    def _on_edges_toggled(self, checked):
        # Sync combo when toggling edges independently
        if checked and self.rep_combo.currentText() == "Surface":
            self.rep_combo.blockSignals(True)
            self.rep_combo.setCurrentText("Surface + Edges")
            self.rep_combo.blockSignals(False)
        elif not checked and self.rep_combo.currentText() == "Surface + Edges":
            self.rep_combo.blockSignals(True)
            self.rep_combo.setCurrentText("Surface")
            self.rep_combo.blockSignals(False)
        self.canvas.set_show_edges(checked)

    def _on_doublesided(self, checked):
        self.canvas.set_doublesided(checked)

    def _on_smooth_shading(self, checked):
        self.canvas.set_smooth_shading(checked)

    def _on_opacity(self, value):
        self.opacity_label.setText(f"{value}%")
        self.canvas.set_opacity(value / 100.0)

    def _on_metallic(self, value):
        v = value / 100.0
        self.metallic_label.setText(f"{v:.2f}")
        self.canvas.set_metallic(v)

    def _on_roughness(self, value):
        v = value / 100.0
        self.roughness_label.setText(f"{v:.2f}")
        self.canvas.set_roughness(v)

    def _on_color(self, text):
        color_map = {
            "White": "white",
            "Light Gray": "lightgray",
            "Beige": "wheat",
            "Light Blue": "lightblue",
            "Cyan": "lightcyan",
            "Gold": "gold",
        }
        self.canvas.set_mesh_color(color_map.get(text, "white"))

    def _on_parallel_projection(self, checked):
        self.canvas.set_parallel_projection(checked)

    def _on_brush_size(self, value):
        pct = value / 10.0
        self.brush_size_label.setText(f"{pct:.1f}%")
        # Compute actual radius from mesh bounding box diagonal
        if self.canvas._main_mesh is not None:
            bounds = self.canvas._main_mesh.bounds
            diag = np.linalg.norm(
                np.array([bounds[1], bounds[3], bounds[5]])
                - np.array([bounds[0], bounds[2], bounds[4]])
            )
            self.canvas.set_brush_radius(diag * pct / 100.0)

    def _on_mode_clicked(self, btn_id):
        mode_map = {0: Canvas3D.VIEW, 1: Canvas3D.BRUSH, 2: Canvas3D.KEYPOINT}
        mode = mode_map.get(btn_id, Canvas3D.VIEW)
        self.canvas.set_mode(mode)

    def set_mode(self, mode):
        """Sync toggle buttons to reflect externally set mode"""
        mode_to_id = {Canvas3D.VIEW: 0, Canvas3D.BRUSH: 1, Canvas3D.KEYPOINT: 2}
        btn_id = mode_to_id.get(mode, 0)
        self.mode_btn_group.button(btn_id).setChecked(True)


class Canvas3D(QtInteractor):
    """3D Canvas for mesh visualization and annotation"""

    # Emitted once when a new label is first painted (not per stroke)
    new_shape = QtCore.pyqtSignal()
    # Emitted when existing shape data changes (for dirty tracking)
    shapes_updated = QtCore.pyqtSignal()
    selection_changed = QtCore.pyqtSignal(list)
    zoom_request = QtCore.pyqtSignal(int, QtCore.QPoint)
    scroll_request = QtCore.pyqtSignal(int, QtCore.Qt.Orientation)

    VIEW = "view"
    BRUSH = "brush"
    KEYPOINT = "keypoint"

    # Default rendering properties
    _DEFAULT_COLOR = "white"
    _DEFAULT_METALLIC = 0.1
    _DEFAULT_ROUGHNESS = 0.5
    _DEFAULT_OPACITY = 1.0

    # Unlabeled sentinel
    _NO_LABEL = -1

    def __init__(self, parent=None):
        # Suppress VTK warnings
        vtk.vtkObject.GlobalWarningDisplayOff()

        super().__init__(parent)
        self.parent = parent
        self._main_mesh = None
        self.mode = self.VIEW
        self.brush_radius = 0.05
        self.active_label = ""
        self.selected_shapes = []

        # --- Efficient data structures ---
        # Per-vertex label id: numpy int array, _NO_LABEL = unlabeled
        self._vertex_label_ids = np.array([], dtype=np.int32)
        # Bidirectional label <-> id mapping
        self._label_to_id = {}  # label_str -> int id
        self._id_to_label = []  # id -> label_str
        # Label colors: id -> (r, g, b) uint8
        self._label_colors = {}  # label_str -> (r, g, b)
        # One consolidated Shape per label for saving
        self._shapes_by_label = {}  # label_str -> Shape

        # Interaction state
        self._painting = False
        self._stroke_dirty = False  # vertices changed during current drag
        self._last_paint_pos = (
            None  # position of the last paint event for interpolation
        )
        self._last_screen_pos = (
            None  # position of the last screen paint event for interpolation
        )
        self._observer_tags = []
        self._cursor_actor_name = "__brush_cursor__"

        # Per-vertex RGB colors on the main mesh
        self._vertex_colors = None  # np.uint8 (n, 3) — current vertex colors
        self._base_color_rgb = np.array([255, 255, 255], dtype=np.uint8)

        # Reusable picker (avoid re-creating per event)
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.005)

        # VTK point locator for fast radius queries
        self._point_locator = None
        self._locator_dataset = None  # prevent GC
        self._mesh_points = None  # stable copy of mesh points

        # Current rendering state
        self._show_edges = False
        self._smooth_shading = True
        self._doublesided = True
        self._color = self._DEFAULT_COLOR
        self._metallic = self._DEFAULT_METALLIC
        self._roughness = self._DEFAULT_ROUGHNESS
        self._opacity = self._DEFAULT_OPACITY

        # Setup plotter
        self.enable_trackball_style()
        self.add_axes()
        self.set_background("slategray", top="white")
        self._setup_lighting()

    # --- Lighting ---

    def _setup_lighting(self):
        """Configure three-point lighting for better surface perception"""
        self.remove_all_lights()
        key_light = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), intensity=0.8)
        key_light.positional = False
        fill_light = pv.Light(
            position=(-1, 0.5, 0.5), focal_point=(0, 0, 0), intensity=0.4
        )
        fill_light.positional = False
        rim_light = pv.Light(
            position=(0, -1, 0.5), focal_point=(0, 0, 0), intensity=0.3
        )
        rim_light.positional = False
        self.add_light(key_light)
        self.add_light(fill_light)
        self.add_light(rim_light)

    # --- Mesh rendering ---

    def _get_main_actor(self):
        if "main_mesh" in self.renderer.actors:
            return self.renderer.actors["main_mesh"]
        return None

    def _redraw_mesh(self):
        if self._main_mesh is None:
            return

        # Ensure _vertex_colors size matches mesh size to avoid ValueError
        if (
            self._vertex_colors is not None
            and len(self._vertex_colors) != self._main_mesh.n_points
        ):
            return

        has_paint = self._vertex_colors is not None and np.any(
            self._vertex_label_ids != self._NO_LABEL
        )

        if has_paint:
            # Explicitly set scalars on the mesh object and ensure they are active
            self._main_mesh.point_data["label_colors"] = self._vertex_colors
            self._main_mesh.set_active_scalars("label_colors")
            self.add_mesh(
                self._main_mesh,
                name="main_mesh",
                pickable=True,
                scalars="label_colors",
                rgb=True,
                opacity=self._opacity,
                smooth_shading=self._smooth_shading,
                split_sharp_edges=self._smooth_shading,
                show_edges=self._show_edges,
                reset_camera=False,
            )
        else:
            # Clean mesh, no paint — use PBR material
            if "label_colors" in self._main_mesh.point_data:
                del self._main_mesh.point_data["label_colors"]
            self.add_mesh(
                self._main_mesh,
                name="main_mesh",
                pickable=True,
                pbr=True,
                metallic=self._metallic,
                roughness=self._roughness,
                color=self._color,
                opacity=self._opacity,
                smooth_shading=self._smooth_shading,
                split_sharp_edges=self._smooth_shading,
                show_edges=self._show_edges,
                reset_camera=False,
            )

        actor = self._get_main_actor()
        if actor:
            actor.GetProperty().SetBackfaceCulling(not self._doublesided)
        self.render()

    def load_mesh(self, filename):
        """Load and display a mesh file"""
        self.clear()
        self._shapes_by_label.clear()
        self._label_to_id.clear()
        self._id_to_label.clear()
        self._setup_lighting()
        try:
            self._main_mesh = pv.read(filename)
            if not self._main_mesh.n_points:
                raise ValueError("Mesh has no points")

            self._main_mesh.compute_normals(inplace=True)
            n = self._main_mesh.n_points

            # Stable copy of vertex positions (survives add_mesh transforms)
            self._mesh_points = np.array(self._main_mesh.points, copy=True)

            # Init vertex labels array
            self._vertex_label_ids = np.full(n, self._NO_LABEL, dtype=np.int32)

            # Init per-vertex colors (all base color)
            self._vertex_colors = np.tile(self._base_color_rgb, (n, 1)).astype(np.uint8)

            # Build VTK point locator for fast radius queries
            self._build_point_locator()

            self._redraw_mesh()
            self.reset_camera()

            # Brush radius = 2% of bounding box diagonal
            bounds = self._main_mesh.bounds
            diag = np.linalg.norm(
                np.array([bounds[1], bounds[3], bounds[5]])
                - np.array([bounds[0], bounds[2], bounds[4]])
            )
            self.brush_radius = diag * 0.02

        except Exception as e:
            print(f"Error loading mesh: {e}")

    def _build_point_locator(self):
        """Build a VTK static point locator for O(log n) radius queries"""
        # Keep reference to dataset to prevent garbage collection
        self._locator_dataset = pv.PolyData(self._mesh_points)
        self._point_locator = vtk.vtkStaticPointLocator()
        self._point_locator.SetDataSet(self._locator_dataset)
        self._point_locator.BuildLocator()

    # --- Label id management ---

    @property
    def vertex_label_ids(self):
        """Return array of vertex label ids"""
        return self._vertex_label_ids

    @vertex_label_ids.setter
    def vertex_label_ids(self, value):
        """Set vertex label ids"""
        n_verts = len(self._vertex_label_ids)
        ids = np.array(value, dtype=np.int32)
        if len(ids) == n_verts:
            self._vertex_label_ids = ids
            self._refresh_vertex_colors()

    def load_vertex_label_ids(self, ids):
        """Load vertex label ids from array of class ids (int) or RLE"""
        if self._main_mesh is None:
            return
        n_verts = len(self._vertex_label_ids)

        # Check if it's RLE-encoded (heuristically or by checking decoded length)
        if len(ids) != n_verts:
            try:
                decoded_ids = utils.decode_rle(ids)
                if len(decoded_ids) == n_verts:
                    ids = decoded_ids
            except Exception:
                pass

        ids_arr = np.array(ids, dtype=np.int32)
        if len(ids_arr) != n_verts:
            return

        self._vertex_label_ids = ids_arr

        # Build shapes from label ids
        self._shapes_by_label.clear()
        unique_lids = np.unique(self._vertex_label_ids)
        new_labels = []
        for lid in unique_lids:
            if lid == self._NO_LABEL:
                continue

            label = self._get_lid_to_label(lid)
            if not label:
                continue

            indices = np.where(self._vertex_label_ids == lid)[0]
            if len(indices) > 0:
                self._shapes_by_label[label] = Shape(
                    shape_type="brush_3d",
                    vertex_indices=sorted(indices.tolist()),
                    label=label,
                )
                new_labels.append(label)

        self._refresh_vertex_colors()
        for _ in new_labels:
            self.new_shape.emit()

    def _get_or_create_label_id(self, label):
        """Get numeric id for a label string, creating if needed"""
        if label not in self._label_to_id:
            lid = len(self._id_to_label)
            self._label_to_id[label] = lid
            self._id_to_label.append(label)
        return self._label_to_id[label]

    def _get_lid_to_label(self, lid):
        """Get label string for a numeric id"""
        if 0 <= lid < len(self._id_to_label):
            return self._id_to_label[lid]
        return None

    # --- View control methods ---

    def set_representation(self, mode):
        actor = self._get_main_actor()
        if not actor:
            return
        prop = actor.GetProperty()
        if mode == "Wireframe":
            prop.SetRepresentationToWireframe()
            self._show_edges = False
        elif mode == "Points":
            prop.SetRepresentationToPoints()
            prop.SetPointSize(3)
            self._show_edges = False
        elif mode == "Surface + Edges":
            prop.SetRepresentationToSurface()
            prop.SetEdgeVisibility(True)
            self._show_edges = True
        else:
            prop.SetRepresentationToSurface()
            prop.SetEdgeVisibility(False)
            self._show_edges = False
        self.render()

    def set_show_edges(self, show):
        self._show_edges = show
        actor = self._get_main_actor()
        if actor:
            actor.GetProperty().SetEdgeVisibility(show)
            self.render()

    def set_doublesided(self, enabled):
        self._doublesided = enabled
        actor = self._get_main_actor()
        if actor:
            actor.GetProperty().SetBackfaceCulling(not enabled)
            self.render()

    def set_smooth_shading(self, enabled):
        self._smooth_shading = enabled
        self._redraw_mesh()

    def set_opacity(self, opacity):
        self._opacity = opacity
        actor = self._get_main_actor()
        if actor:
            actor.GetProperty().SetOpacity(opacity)
            self.render()

    def set_metallic(self, value):
        self._metallic = value
        self._redraw_mesh()

    def set_roughness(self, value):
        self._roughness = value
        self._redraw_mesh()

    def set_mesh_color(self, color):
        self._color = color
        self._redraw_mesh()

    def set_parallel_projection(self, enabled):
        if enabled:
            self.enable_parallel_projection()
        else:
            self.disable_parallel_projection()

    def set_viewup_and_position(self, direction):
        if self._main_mesh is None:
            return
        if direction == "iso":
            self.view_isometric()
        else:
            dx, dy, dz = direction
            view_up = (0, 1, 0) if abs(dz) == 1 else (0, 0, 1)
            self.view_vector((-dx, -dy, -dz), viewup=view_up)
        self.reset_camera()

    def set_brush_radius(self, radius):
        self.brush_radius = radius

    # --- Mode and interaction ---

    def set_mode(self, mode):
        self.mode = mode
        self.disable_picking()
        self._remove_paint_observers()
        self._hide_cursor()
        self._painting = False
        if mode == self.VIEW:
            self.enable_trackball_style()
        else:
            style = vtk.vtkInteractorStyleUser()
            self.iren.interactor.SetInteractorStyle(style)
            self._install_paint_observers()

    def _install_paint_observers(self):
        self._remove_paint_observers()
        iren = self.iren.interactor
        self._observer_tags = [
            iren.AddObserver("MouseMoveEvent", self._on_mouse_move),
            iren.AddObserver("LeftButtonPressEvent", self._on_left_press),
            iren.AddObserver("LeftButtonReleaseEvent", self._on_left_release),
        ]

    def _remove_paint_observers(self):
        iren = self.iren.interactor
        for tag in self._observer_tags:
            iren.RemoveObserver(tag)
        self._observer_tags = []

    def _on_mouse_move(self, obj, event):
        if self._main_mesh is None:
            return
        pos = self.iren.interactor.GetEventPosition()
        hit = self._raycast(pos)
        if hit is not None:
            self._show_cursor(hit)
            if self._painting and self.mode == self.BRUSH:
                self._paint_at(hit, pos)
        else:
            self._hide_cursor()

    def _on_left_press(self, obj, event):
        if self._main_mesh is None:
            return
        self._painting = True
        self._stroke_dirty = False
        pos = self.iren.interactor.GetEventPosition()
        hit = self._raycast(pos)
        if hit is not None:
            self._paint_at(hit, pos)

    def _on_left_release(self, obj, event):
        if self._painting and self._stroke_dirty:
            self.shapes_updated.emit()
        self._painting = False
        self._stroke_dirty = False
        self._last_paint_pos = None
        self._last_screen_pos = None

    # --- Raycasting (reuses picker) ---

    def _raycast(self, screen_pos):
        hit = self._picker.Pick(screen_pos[0], screen_pos[1], 0, self.renderer)
        if hit:
            return np.array(self._picker.GetPickPosition())
        return None

    # --- Fast radius query via VTK locator ---

    def _find_vertices_in_radius(self, center):
        """O(log n) radius query using VTK static point locator"""
        if self._point_locator is None:
            return np.array([], dtype=np.int64)
        result = vtk.vtkIdList()
        self._point_locator.FindPointsWithinRadius(self.brush_radius, center, result)
        n = result.GetNumberOfIds()
        if n == 0:
            return np.array([], dtype=np.int64)

        # Optimized conversion from vtkIdList to numpy array
        indices = np.empty(n, dtype=np.int64)
        for i in range(n):
            indices[i] = result.GetId(i)
        return indices

    def _find_vertices_in_screen_radius(self, hit_point, screen_pos):
        """Find vertices whose projection is within the brush radius on screen."""
        if self._point_locator is None or self._main_mesh is None:
            return np.array([], dtype=np.int64)

        # 1. Estimate a loose 3D search radius to get candidates.
        # We use a large enough radius to cover the projected area.
        # This radius is approximated based on the brush radius and camera distance.
        # Brush radius is a percentage of diagonal, let's use it directly for 3D search first
        # to narrow down candidates.
        search_radius = self.brush_radius * 2.0

        candidates_ids = vtk.vtkIdList()
        self._point_locator.FindPointsWithinRadius(
            search_radius, hit_point, candidates_ids
        )
        n = candidates_ids.GetNumberOfIds()
        if n == 0:
            return np.array([], dtype=np.int64)

        # 2. Project candidate vertices to screen space and check distance to mouse.
        # Using vtkCoordinate for projection
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()

        # Determine the 2D brush radius in pixels
        # Currently brush_radius is in world units.
        # We need to find the screen-space radius.
        # Let's project two points separated by brush_radius in world space.
        p1_world = hit_point
        p2_world = hit_point + np.array([self.brush_radius, 0, 0])

        def world_to_screen(world_pt):
            coord.SetValue(*world_pt)
            return np.array(coord.GetComputedDoubleDisplayValue(self.renderer))

        p1_screen = world_to_screen(p1_world)
        p2_screen = world_to_screen(p2_world)
        pixel_radius = np.linalg.norm(p1_screen - p2_screen)

        target_screen = np.array(screen_pos)

        indices = []
        for i in range(n):
            idx = candidates_ids.GetId(i)
            pt_world = self._mesh_points[idx]
            pt_screen = world_to_screen(pt_world)
            if np.linalg.norm(pt_screen - target_screen) <= pixel_radius:
                indices.append(idx)

        return np.array(indices, dtype=np.int64)

    # --- Mesh overlay for painting & cursor ---

    def _apply_colors_and_render(self):
        """Apply vertex colors to the mesh and re-render"""
        self._redraw_mesh()

    def _show_cursor(self, center):
        """Show a transparent sphere at the brush position"""
        sphere = pv.Sphere(
            radius=self.brush_radius,
            center=center,
            theta_resolution=16,
            phi_resolution=16,
        )
        self.add_mesh(
            sphere,
            color="cyan",
            opacity=0.2,
            name=self._cursor_actor_name,
            pickable=False,
            reset_camera=False,
        )

    def _hide_cursor(self):
        """Remove the brush cursor sphere"""
        if self._cursor_actor_name in self.renderer.actors:
            self.remove_actor(self._cursor_actor_name)

    # --- Vertex painting ---

    def set_label_color(self, label, rgb):
        self._label_colors[label] = tuple(rgb)
        # Update existing painted vertices with new color
        if label in self._label_to_id and self._vertex_colors is not None:
            lid = self._label_to_id[label]
            mask = self._vertex_label_ids == lid
            if np.any(mask):
                self._vertex_colors[mask] = [rgb[0], rgb[1], rgb[2]]

    def _paint_at(self, point, screen_pos=None):
        """Paint vertices at a world-space point with immediate visual feedback."""
        label = self.active_label or ("point" if self.mode == self.KEYPOINT else "mask")
        lid = self._get_or_create_label_id(label)
        rgb = self._label_colors.get(label, (0, 255, 0))

        if self.mode == self.KEYPOINT:
            idx = self._locator_dataset.find_closest_point(point)
            self._vertex_label_ids[idx] = lid
            self._vertex_colors[idx] = [rgb[0], rgb[1], rgb[2]]
            self._merge_into_shape(label, [int(idx)], "keypoint_3d")
        elif self.mode == self.BRUSH:
            # Interpolate between last position and current position for smooth strokes
            points_to_paint = [point]
            screens_to_paint = [screen_pos] if screen_pos is not None else [None]

            if self._last_paint_pos is not None:
                diff = point - self._last_paint_pos
                dist = np.linalg.norm(diff)
                # If distance is more than half a radius, interpolate
                if dist > self.brush_radius * 0.5:
                    num_steps = int(dist / (self.brush_radius * 0.5))
                    for i in range(1, num_steps):
                        interp_point = self._last_paint_pos + diff * (i / num_steps)
                        points_to_paint.append(interp_point)
                        # Also interpolate screen position if available
                        if (
                            screen_pos is not None
                            and getattr(self, "_last_screen_pos", None) is not None
                        ):
                            s_diff = np.array(screen_pos) - np.array(
                                self._last_screen_pos
                            )
                            interp_screen = np.array(self._last_screen_pos) + s_diff * (
                                i / num_steps
                            )
                            screens_to_paint.append(interp_screen)
                        else:
                            screens_to_paint.append(None)

            new_indices_set = set()
            for p, s in zip(points_to_paint, screens_to_paint):
                if s is not None:
                    indices = self._find_vertices_in_screen_radius(p, s)
                else:
                    indices = self._find_vertices_in_radius(p)

                if len(indices) > 0:
                    self._vertex_label_ids[indices] = lid
                    self._vertex_colors[indices] = [rgb[0], rgb[1], rgb[2]]
                    new_indices_set.update(indices.tolist())

            if new_indices_set:
                self._merge_into_shape(label, new_indices_set, "brush_3d")

            self._last_paint_pos = point
            if screen_pos is not None:
                self._last_screen_pos = screen_pos

        self._stroke_dirty = True
        self._apply_colors_and_render()

    def _merge_into_shape(self, label, new_indices, shape_type):
        """Merge vertices into the consolidated shape for this label"""
        if label in self._shapes_by_label:
            shape = self._shapes_by_label[label]
            # Use a set for much faster merging, then convert back to sorted list
            existing = set(shape.vertex_indices)
            existing.update(new_indices)
            shape.vertex_indices = sorted(list(existing))
        else:
            shape = Shape(
                shape_type=shape_type,
                vertex_indices=sorted(list(set(new_indices))),
                label=label,
            )
            self._shapes_by_label[label] = shape
            self.new_shape.emit()

    def _refresh_vertex_colors(self):
        """Rebuild vertex colors from vertex label ids"""
        if self._vertex_colors is None:
            return

        # Reset all to base color
        self._vertex_colors[:] = self._base_color_rgb

        # Apply labels colors
        n_labels = len(self._id_to_label)
        for lid in range(n_labels):
            mask = self._vertex_label_ids == lid
            if not np.any(mask):
                continue
            label_str = self._id_to_label[lid]
            rgb = self._label_colors.get(label_str, (0, 255, 0))
            self._vertex_colors[mask] = [rgb[0], rgb[1], rgb[2]]

        self._apply_colors_and_render()

    # --- Shape management ---

    @property
    def shapes(self):
        """Return consolidated list of shapes (one per label)"""
        return list(self._shapes_by_label.values())

    @shapes.setter
    def shapes(self, value):
        """Allow assignment for compatibility (e.g. shapes = [])"""
        self._shapes_by_label.clear()
        for s in value:
            if s.label:
                self._shapes_by_label[s.label] = s

    def add_shape(self, shape):
        """Add a shape externally"""
        lid = self._get_or_create_label_id(shape.label)
        n_verts = len(self._vertex_label_ids)
        indices = np.array(shape.vertex_indices, dtype=np.int64)
        indices = indices[(indices >= 0) & (indices < n_verts)]
        if len(indices) > 0:
            self._vertex_label_ids[indices] = lid
        self._shapes_by_label[shape.label] = shape
        self._refresh_vertex_colors()
        self.new_shape.emit()

    def load_shapes(self, shapes, replace=True):
        if replace:
            # If we already have vertex labels, we don't want to clear them here
            # as they were likely loaded via load_vertex_label_ids already.
            # We only clear if no vertex labels are present.
            if np.all(self._vertex_label_ids == self._NO_LABEL):
                self.clear_shapes()
            else:
                # Only clear non-brush shapes from our tracking
                to_remove = [
                    l
                    for l, s in self._shapes_by_label.items()
                    if s.shape_type not in ("brush_3d", "keypoint_3d")
                ]
                for l in to_remove:
                    del self._shapes_by_label[l]

        n_verts = len(self._vertex_label_ids)
        for shape in shapes:
            if shape.shape_type not in ("brush_3d", "keypoint_3d"):
                # Handle other potential shape types if necessary
                continue

            # If it's a brush shape but we already have labels from vertex_label_ids,
            # we skip it to avoid redundancy/overwriting with potentially older data.
            if shape.shape_type == "brush_3d" and not np.all(
                self._vertex_label_ids == self._NO_LABEL
            ):
                continue

            if not shape.vertex_indices:
                continue
            label = shape.label
            lid = self._get_or_create_label_id(label)
            indices = np.array(shape.vertex_indices, dtype=np.int64)
            # Clamp to valid vertex range
            indices = indices[(indices >= 0) & (indices < n_verts)]
            if len(indices) == 0:
                continue
            self._vertex_label_ids[indices] = lid
            # Merge into consolidated shape (no signal during bulk load)
            if label in self._shapes_by_label:
                existing = set(self._shapes_by_label[label].vertex_indices)
                existing.update(indices.tolist())
                self._shapes_by_label[label].vertex_indices = sorted(existing)
            else:
                self._shapes_by_label[label] = Shape(
                    shape_type=shape.shape_type,
                    vertex_indices=sorted(indices.tolist()),
                    label=label,
                )
        self._refresh_vertex_colors()
        # Emit new_shape once per label for the label list
        for label in self._shapes_by_label:
            self.new_shape.emit()

    def clear_shapes(self):
        if len(self._vertex_label_ids) > 0:
            self._vertex_label_ids[:] = self._NO_LABEL
        if self._vertex_colors is not None:
            self._vertex_colors[:] = self._base_color_rgb
            self._apply_colors_and_render()
        self._shapes_by_label.clear()
        self._label_to_id.clear()
        self._id_to_label.clear()
