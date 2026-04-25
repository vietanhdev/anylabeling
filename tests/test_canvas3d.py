"""Headless tests for the Canvas3D data model.

The full mesh-labeling UX needs an interactive VTK render window — a real
display, or `vtk-osmesa` plus the offscreen Qt platform. The tests below
exercise only the data-model layer (mesh loading, vertex_label_ids array,
the label↔lid mapping, mode switching) and skip cleanly when the heavy
deps or a working VTK backend are not available.

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
    from PyQt6.QtWidgets import QApplication

    pv.OFF_SCREEN = True
    _APP = QApplication.instance() or QApplication(sys.argv)

    from anylabeling.views.labeling.widgets.canvas3d import Canvas3D
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    Canvas3D = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


# Skip in any CI environment. VTK/pyvista wheels on macOS GitHub runners
# segfault when Canvas3D() initialises without a real display — a hard SIGSEGV
# kills the process before any Python-level skipTest can fire. Linux's Mesa
# software fallback works fine, but rather than gate per-platform we just
# disable for all CI and run this test locally where a real GL context exists.
_IN_CI = os.environ.get("CI", "").lower() in ("1", "true", "yes")


@unittest.skipIf(Canvas3D is None, f"mesh deps unavailable: {_IMPORT_ERROR}")
@unittest.skipIf(_IN_CI, "Canvas3D test requires a real display; CI runners segfault on init")
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
        # First paint kicks off the switch from PBR to scalar shading.
        self.canvas._scalar_mode_active = False
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
