"""Tests for ModelRegistry."""
import sys
import os
import unittest
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.services.auto_labeling.registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):

    def setUp(self):
        # Save and clear the registry state before each test
        self._original_registry = dict(ModelRegistry._registry)
        # Remove any test keys that might be left from a previous run
        for key in list(ModelRegistry._registry.keys()):
            if key.startswith("_test_"):
                del ModelRegistry._registry[key]

    def tearDown(self):
        # Remove only the test keys we added
        for key in list(ModelRegistry._registry.keys()):
            if key.startswith("_test_"):
                del ModelRegistry._registry[key]

    def test_register_and_get(self):
        @ModelRegistry.register("_test_model_a")
        class FakeModelA:
            pass

        self.assertIs(ModelRegistry.get("_test_model_a"), FakeModelA)

    def test_list_models_contains_registered(self):
        @ModelRegistry.register("_test_model_b")
        class FakeModelB:
            pass

        self.assertIn("_test_model_b", ModelRegistry.list_models())

    def test_get_missing_returns_none(self):
        self.assertIsNone(ModelRegistry.get("_test_nonexistent_xyz"))

    def test_register_returns_class_unchanged(self):
        @ModelRegistry.register("_test_model_c")
        class FakeModelC:
            pass

        # The decorator must return the original class (so it can still be used)
        self.assertEqual(FakeModelC.__name__, "FakeModelC")

    def test_overwrite_logs_warning(self):
        @ModelRegistry.register("_test_model_d")
        class FakeModelD1:
            pass

        with self.assertLogs("root", level="WARNING"):
            @ModelRegistry.register("_test_model_d")
            class FakeModelD2:
                pass

        # After overwrite, the new class should be stored
        self.assertIs(ModelRegistry.get("_test_model_d"), FakeModelD2)

    def test_list_models_is_list(self):
        result = ModelRegistry.list_models()
        self.assertIsInstance(result, list)

    def test_segment_anything_registered(self):
        # Importing segment_anything registers the class via the decorator
        import anylabeling.services.auto_labeling.segment_anything  # noqa: F401
        self.assertIn("segment_anything", ModelRegistry.list_models())


if __name__ == "__main__":
    unittest.main()
