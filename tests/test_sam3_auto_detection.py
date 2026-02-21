import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.services.auto_labeling.segment_anything import SegmentAnything

class TestSAM3AutoDetection(unittest.TestCase):
    
    @patch('anylabeling.services.auto_labeling.segment_anything.onnx.load')
    @patch('anylabeling.services.auto_labeling.segment_anything.SegmentAnything3ONNX')
    @patch('anylabeling.services.auto_labeling.segment_anything.SegmentAnything.get_model_abs_path')
    @patch('os.path.isfile', return_value=True)
    def test_sam3_detection_by_config(self, mock_isfile, mock_get_path, mock_sam3_class, mock_onnx_load):
        # Configuration containing language_encoder_path
        config = {
            "name": "sam3_test",
            "type": "segment_anything",
            "display_name": "SAM3 Test",
            "encoder_model_path": "enc.onnx",
            "decoder_model_path": "dec.onnx",
            "language_encoder_path": "lang.onnx",
            "input_size": 1008,
            "max_width": 1008,
            "max_height": 1008
        }
        
        mock_get_path.side_effect = lambda cfg, field: cfg.get(field)
        
        # Instantiate SegmentAnything with the SAM3 config
        # It should detect it's SAM3 because 'language_encoder_path' is in config
        model = SegmentAnything(config, on_message=print)
        
        # Verify SegmentAnything3ONNX was instantiated
        mock_sam3_class.assert_called_once()
        print("SUCCESS: SAM3 correctly detected by config field.")

    @patch('anylabeling.services.auto_labeling.segment_anything.SegmentAnything2ONNX')
    @patch('anylabeling.services.auto_labeling.segment_anything.SegmentAnythingONNX')
    @patch('anylabeling.services.auto_labeling.segment_anything.onnx.load')
    @patch('anylabeling.services.auto_labeling.segment_anything.SegmentAnything3ONNX')
    @patch('anylabeling.services.auto_labeling.segment_anything.SegmentAnything.get_model_abs_path')
    @patch('os.path.isfile', return_value=True)
    def test_sam3_detection_fallback_to_onnx(self, mock_isfile, mock_get_path, mock_sam3_class, mock_onnx_load, mock_sam1_class, mock_sam2_class):
        # Configuration WITHOUT language_encoder_path
        config = {
            "name": "sam3_test_custom",
            "type": "segment_anything",
            "display_name": "SAM3 Test Custom",
            "encoder_model_path": "enc.onnx",
            "decoder_model_path": "dec.onnx",
            "input_size": 1008,
            "max_width": 1008,
            "max_height": 1008
        }
        
        mock_get_path.side_effect = lambda cfg, field: cfg.get(field)
        
        # Mock ONNX model to return SAM3 specific input names.
        # Detection uses backbone_fpn_0 or language_mask (NOT vision_pos_enc_0,
        # which onnxsim removes during simplification).
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "backbone_fpn_0"
        mock_model.graph.input = [mock_input]
        mock_onnx_load.return_value = mock_model

        # Instantiate
        model = SegmentAnything(config, on_message=print)

        # Verify fallback detection worked
        mock_sam3_class.assert_called_once()
        print("SUCCESS: SAM3 correctly detected by ONNX input fallback.")

    @patch('anylabeling.services.auto_labeling.segment_anything.SegmentAnything2ONNX')
    @patch('anylabeling.services.auto_labeling.segment_anything.SegmentAnythingONNX')
    @patch('anylabeling.services.auto_labeling.segment_anything.onnx.load')
    @patch('anylabeling.services.auto_labeling.segment_anything.SegmentAnything3ONNX')
    @patch('anylabeling.services.auto_labeling.segment_anything.SegmentAnything.get_model_abs_path')
    @patch('os.path.isfile', return_value=True)
    def test_sam2_detection(self, mock_isfile, mock_get_path, mock_sam3_class, mock_onnx_load, mock_sam1_class, mock_sam2_class):
        config = {
            "name": "sam2_test",
            "type": "segment_anything",
            "display_name": "SAM2 Test",
            "encoder_model_path": "enc.onnx",
            "decoder_model_path": "dec.onnx",
            "input_size": 1024,
            "max_width": 1024,
            "max_height": 1024
        }
        mock_get_path.side_effect = lambda cfg, field: cfg.get(field)
        
        # Mock ONNX model for SAM2
        mock_model = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "high_res_feats_0"
        mock_model.graph.input = [mock_input]
        mock_onnx_load.return_value = mock_model
        
        model = SegmentAnything(config, on_message=print)
        mock_sam2_class.assert_called_once()
        print("SUCCESS: SAM2 correctly detected.")

if __name__ == '__main__':
    unittest.main()
