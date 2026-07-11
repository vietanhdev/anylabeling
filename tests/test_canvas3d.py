"""Headless tests for the Canvas3D data model.

The full mesh-labeling UX needs a real, working VTK render window. The
tests below exercise the data-model layer (mesh loading, vertex_label_ids
array, the label<->id mapping, save/load round-tripping, mode switching)
and skip cleanly when a *safe* backend isn't available.

"Safe" specifically means vtk-osmesa (a pure-software VTK build, no
display required):

    pip install --index-url https://wheels.vtk.org vtk-osmesa

A real DISPLAY is *not* a reliable substitute for this. Constructing a
live VTK render window with the regular `vtk` wheel can hard-crash the
whole process (SIGSEGV / an uncatchable native abort — not a Python
exception) even when a real X11/XWayland display is present; this was
confirmed empirically, not assumed. Only run this file with vtk-osmesa
installed, or expect the process to die without a traceback.

vtk-osmesa wheels are currently published for Python 3.11 and 3.12, not
3.13 — see .github/workflows/tests.yml for how CI enables this.

Run:
    QT_QPA_PLATFORM=offscreen python -m unittest tests.test_canvas3d -v
"""
import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import numpy as np
    import pyvista as pv
    import vtk
    from PyQt6.QtWidgets import QApplication

    pv.OFF_SCREEN = True
    _APP = QApplication.instance() or QApplication(sys.argv)

    from anylabeling.views.labeling import utils as labeling_utils
    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets.canvas3d import Canvas3D
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    Canvas3D = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def _has_safe_render_backend():
    """True only if VTK is actually running on vtkOSOpenGLRenderWindow
    (the vtk-osmesa software backend). This is a runtime capability check,
    not an environment guess (CI env var / platform name / DISPLAY
    presence) — those were tried and found unreliable."""
    if Canvas3D is None:
        return False
    try:
        return vtk.vtkRenderWindow().GetClassName() == "vtkOSOpenGLRenderWindow"
    except Exception:
        return False


_SAFE_BACKEND = _has_safe_render_backend()
_SKIP_REASON = (
    "Canvas3D needs vtk-osmesa for a safe headless run (a real DISPLAY is "
    "not a reliable substitute for this specific dependency — it can still "
    "hard-crash the process). Install with: "
    "pip install --index-url https://wheels.vtk.org vtk-osmesa"
)


@unittest.skipIf(Canvas3D is None, f"mesh deps unavailable: {_IMPORT_ERROR}")
@unittest.skipUnless(_SAFE_BACKEND, _SKIP_REASON)
class TestCanvas3DDataModel(unittest.TestCase):
    """Verify Canvas3D's pure-data operations work without an X display."""

    @classmethod
    def setUpClass(cls):
        # Use the bundled sphere sample — exercises a non-trivial vertex count
        # (530 verts) without an off-disk fixture. Falls back to a synthetic
        # sphere if the sample file moves.
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        sample = os.path.join(repo_root, "sample_meshes", "sphere.obj")
        if os.path.exists(sample):
            cls.mesh_path = sample
            cls.expected_n_points = pv.read(sample).n_points
        else:
            cls.mesh_path = "/tmp/anylabeling_test_mesh.ply"
            sphere = pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8)
            sphere.save(cls.mesh_path)
            cls.expected_n_points = sphere.n_points

        try:
            cls.canvas = Canvas3D()
            cls.canvas.load_mesh(cls.mesh_path)
        except Exception as exc:
            # CI runners without a real (or virtual) display can't bring up
            # the VTK render window. Skip cleanly rather than fail the cell.
            raise unittest.SkipTest(
                f"Canvas3D could not initialise (no working VTK backend): "
                f"{type(exc).__name__}: {exc}"
            )

    def test_load_mesh_creates_actor_and_locator(self):
        self.assertIsNotNone(self.canvas._get_main_actor())
        self.assertIsNotNone(self.canvas._point_locator)

    def test_vertex_label_ids_match_mesh_size(self):
        ids = self.canvas.vertex_label_ids
        self.assertIsNotNone(ids)
        self.assertEqual(len(ids), self.expected_n_points)
        self.assertEqual(ids.dtype.kind, "i")  # integer dtype

    def test_label_to_id_is_stable(self):
        a = self.canvas._get_or_create_label_id("alpha")
        b = self.canvas._get_or_create_label_id("beta")
        a_again = self.canvas._get_or_create_label_id("alpha")
        self.assertEqual(a, a_again, "same label string must yield same id")
        self.assertNotEqual(a, b, "different labels must yield different ids")

    def test_lid_reverse_lookup(self):
        a = self.canvas._get_or_create_label_id("gamma")
        self.assertEqual(self.canvas._get_lid_to_label(a), "gamma")

    def test_vertex_label_ids_round_trip(self):
        a = self.canvas._get_or_create_label_id("x")
        b = self.canvas._get_or_create_label_id("y")
        ids = self.canvas.vertex_label_ids
        ids[:3] = a
        ids[3:6] = b
        self.canvas.vertex_label_ids = ids
        snapshot = self.canvas.vertex_label_ids.copy()

        self.canvas.load_vertex_label_ids(snapshot.tolist())
        self.assertTrue(np.array_equal(self.canvas.vertex_label_ids, snapshot))

    def test_mode_switching_does_not_crash_in_headless(self):
        # The fix in canvas3d.py is exactly that these calls used to AttributeError
        # on `self.iren.interactor` when no interactive window existed.
        self.canvas.set_mode("brush")
        self.canvas.set_mode("view")
        self.canvas.set_mode("brush")

    def test_set_brush_radius_stores_value(self):
        self.canvas.set_brush_radius(0.42)
        self.assertEqual(self.canvas.brush_radius, 0.42)

    def test_in_place_paint_avoids_actor_rebuild(self):
        """After the first paint flips _scalar_mode_active to True, every
        subsequent _apply_colors_and_render must skip _redraw_mesh and just
        push the colour array to VTK in place. Regression test for the
        per-stroke add_mesh slowdown on dense meshes."""
        # First paint kicks off the switch from PBR to scalar shading. A
        # real paint sets both _vertex_label_ids and _vertex_colors
        # together (see _paint_at) — _scalar_mode_active is now derived
        # from has_paint (_vertex_label_ids), so the test must set both to
        # realistically simulate "something got painted", not just the
        # color array.
        self.canvas._scalar_mode_active = False
        lid = self.canvas._get_or_create_label_id("in_place_paint_regress")
        self.canvas._vertex_label_ids[0] = lid
        self.canvas._vertex_colors[0] = [10, 20, 30]
        self.canvas._apply_colors_and_render()
        self.assertTrue(self.canvas._scalar_mode_active)

        # Subsequent paints must NOT rebuild — verify by spying on _redraw_mesh.
        calls = {"n": 0}
        original = self.canvas._redraw_mesh
        self.canvas._redraw_mesh = lambda: calls.__setitem__("n", calls["n"] + 1)
        try:
            for _ in range(5):
                self.canvas._vertex_colors[0] = [40, 50, 60]
                self.canvas._apply_colors_and_render()
            self.assertEqual(
                calls["n"], 0,
                "_apply_colors_and_render should not rebuild the actor "
                "while scalar mode is already active",
            )
        finally:
            self.canvas._redraw_mesh = original

    def test_scalar_coloring_survives_clear_then_load_shapes(self):
        """Regression, found via manual video-recording verification, not
        code reading: clear_shapes() (called at the top of load_shapes()
        when reopening a freshly-loaded, still-unpainted mesh) invokes
        _apply_colors_and_render() while has_paint is False, which builds
        a PBR (non-scalar-colored) actor. _apply_colors_and_render() used
        to then unconditionally set _scalar_mode_active = True regardless
        of which branch _redraw_mesh() actually took. The very next call
        in the same load_shapes() — the one that has real label data —
        would then see _scalar_mode_active already True and take the fast
        in-place point_data mutation path against an actor whose mapper
        was never configured for scalar coloring, so the paint colors
        were written to the mesh's data but never actually rendered."""
        self.canvas.load_mesh(self.mesh_path)  # fresh: definitely unpainted
        shapes = [
            Shape(shape_type="brush_3d", vertex_indices=[0, 1, 2], label="scalar_regress")
        ]
        self.canvas.load_shapes(shapes)  # clear_shapes() runs first internally
        actor = self.canvas._get_main_actor()
        self.assertEqual(
            actor.GetMapper().GetScalarVisibility(),
            1,
            "actor's mapper must be in scalar-coloring mode after loading "
            "real vertex labels, not left in PBR mode from clear_shapes()",
        )

    def test_cursor_actor_is_reused(self):
        """Brush cursor sphere must be one persistent actor that we just
        reposition + scale, not a new actor per mouse-move."""
        self.canvas._cursor_actor = None  # force first creation
        self.canvas._show_cursor((0.0, 0.0, 0.0))
        first = id(self.canvas._cursor_actor)
        self.assertIsNotNone(self.canvas._cursor_actor)
        for i in range(20):
            self.canvas._show_cursor((float(i) * 0.05, 0.0, 0.0))
        self.assertEqual(id(self.canvas._cursor_actor), first)

    def test_mode_constants_reduced_to_view_and_brush(self):
        """Keypoint mode was removed for simplicity; verify."""
        self.assertEqual(Canvas3D.VIEW, "view")
        self.assertEqual(Canvas3D.BRUSH, "brush")
        self.assertFalse(hasattr(Canvas3D, "KEYPOINT"))

    def test_multi_label_save_load_round_trip(self):
        """Regression test for a bundle of three interlocking bugs found
        while fixing this PR:

        1. label_widget.py's format_shape() never persisted vertex_indices,
           so the "shapes" JSON list was geometry-empty for mesh files.
        2. load_vertex_label_ids() ran before load_labels()/load_shapes(),
           so canvas_3d's label<->id mapping didn't exist yet and every
           reconstructed label silently dropped.
        3. load_shapes()'s "already have vertex labels, skip" guard was
           checked per-shape *inside* the loop that itself mutates
           _vertex_label_ids — so as soon as the first shape's vertices
           were written, every subsequent shape in the same call was
           silently skipped. Only the first painted label ever survived
           a save+reload.

        This simulates the full save -> reopen cycle (format_shape's
        output + the RLE vertex_label_ids write, then the now-correct
        load order) for two distinct labels and asserts both survive.
        """
        self.canvas.clear_shapes()
        lid_a = self.canvas._get_or_create_label_id("round_trip_label_a")
        lid_b = self.canvas._get_or_create_label_id("round_trip_label_b")
        self.canvas._vertex_label_ids[0:5] = lid_a
        self.canvas._vertex_label_ids[5:10] = lid_b
        self.canvas._shapes_by_label = {
            "round_trip_label_a": Shape(
                shape_type="brush_3d",
                vertex_indices=list(range(0, 5)),
                label="round_trip_label_a",
            ),
            "round_trip_label_b": Shape(
                shape_type="brush_3d",
                vertex_indices=list(range(5, 10)),
                label="round_trip_label_b",
            ),
        }

        # --- Simulate save: mirrors label_widget.py's format_shape() +
        # the other_data["vertex_label_ids"] RLE write in save_labels().
        saved_shapes = [
            {
                "label": s.label,
                "shape_type": s.shape_type,
                "vertex_indices": s.vertex_indices,
            }
            for s in self.canvas.shapes
        ]
        saved_vertex_label_ids = labeling_utils.encode_rle(
            self.canvas.vertex_label_ids.tolist()
        )

        # --- Simulate reopening the same mesh: load_mesh() resets all
        # label state, exactly like opening the file fresh.
        self.canvas.load_mesh(self.mesh_path)
        self.assertEqual(len(self.canvas.shapes), 0)

        # --- Simulate the (now fixed) load order in label_widget.py:
        # shapes/labels first, THEN the RLE vertex_label_ids array.
        loaded_shapes = [
            Shape(
                shape_type=d["shape_type"],
                vertex_indices=d["vertex_indices"],
                label=d["label"],
            )
            for d in saved_shapes
        ]
        self.canvas.load_shapes(loaded_shapes)
        self.canvas.load_vertex_label_ids(saved_vertex_label_ids)

        self.assertEqual(
            sorted(self.canvas._shapes_by_label.keys()),
            ["round_trip_label_a", "round_trip_label_b"],
            "both labels must survive a save+reload round trip, not just the first",
        )
        self.assertEqual(
            sorted(self.canvas._shapes_by_label["round_trip_label_a"].vertex_indices),
            list(range(0, 5)),
        )
        self.assertEqual(
            sorted(self.canvas._shapes_by_label["round_trip_label_b"].vertex_indices),
            list(range(5, 10)),
        )
        new_lid_a = self.canvas._label_to_id["round_trip_label_a"]
        new_lid_b = self.canvas._label_to_id["round_trip_label_b"]
        self.assertTrue(
            np.array_equal(self.canvas.vertex_label_ids[0:5], np.full(5, new_lid_a))
        )
        self.assertTrue(
            np.array_equal(self.canvas.vertex_label_ids[5:10], np.full(5, new_lid_b))
        )

    def test_load_shapes_does_not_emit_new_shape(self):
        """LabelingWidget.load_shapes() already adds every shape to the UI
        label list before calling canvas_3d.load_shapes() — this method
        must not also emit new_shape, or the label list gets duplicate
        entries and the just-opened file gets marked dirty."""
        self.canvas.clear_shapes()
        emitted = []
        self.canvas.new_shape.connect(lambda: emitted.append(1))
        try:
            shapes = [
                Shape(shape_type="brush_3d", vertex_indices=[0, 1], label="no_emit_a"),
                Shape(shape_type="brush_3d", vertex_indices=[2, 3], label="no_emit_b"),
            ]
            self.canvas.load_shapes(shapes)
            self.assertEqual(emitted, [])
        finally:
            self.canvas.new_shape.disconnect()

    def test_load_vertex_label_ids_does_not_emit_new_shape(self):
        self.canvas.clear_shapes()
        lid = self.canvas._get_or_create_label_id("no_emit_c")
        ids = self.canvas.vertex_label_ids.copy()
        ids[:] = self.canvas._NO_LABEL
        ids[0:3] = lid
        emitted = []
        self.canvas.new_shape.connect(lambda: emitted.append(1))
        try:
            self.canvas.load_vertex_label_ids(ids.tolist())
            self.assertEqual(emitted, [])
        finally:
            self.canvas.new_shape.disconnect()

    def test_load_mesh_returns_false_and_logs_on_bad_file(self):
        """load_mesh() must report failure instead of silently printing and
        leaving the caller with no signal (previously: bare `print()`,
        no return value)."""
        bad_path = os.path.join(
            os.path.dirname(self.mesh_path), "not_a_real_mesh_file.obj"
        )
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("this is not valid mesh data\n")
        try:
            ok = self.canvas.load_mesh(bad_path)
            self.assertFalse(ok)
            self.assertIsNone(self.canvas._main_mesh)
        finally:
            os.remove(bad_path)

    def test_load_mesh_returns_true_on_success(self):
        ok = self.canvas.load_mesh(self.mesh_path)
        self.assertTrue(ok)
        self.assertIsNotNone(self.canvas._main_mesh)


if __name__ == "__main__":
    unittest.main(verbosity=2)
