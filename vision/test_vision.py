"""Focused tests for real detector/OCR wiring; no network or model download."""

import asyncio
import base64
from contextlib import nullcontext
import io
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw
from fastapi import HTTPException

import main
from benchmark_ocr import edit_distance


class VisionTests(unittest.TestCase):
    def image_base64(self):
        image = Image.new("RGB", (100, 80), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def test_roboflow_workflow_sends_configured_key_and_normalizes_box(self):
        workflow_response = {
            "outputs": [{
                "predictions": {
                    "image": {"width": 100, "height": 80},
                    "predictions": [{
                        "x": 20, "y": 20, "width": 30, "height": 20,
                        "class": "handwriting", "confidence": 0.91,
                    }],
                },
            }],
        }
        response = Mock()
        response.json.return_value = workflow_response
        with (
            patch.object(main, "ROBOFLOW_WORKFLOW_URL", "https://example.invalid/workflow"),
            patch("requests.post", return_value=response) as post,
        ):
            regions = main._roboflow_detect(self.image_base64(), "test-workflow", "test-secret")

        response.raise_for_status.assert_called_once()
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-secret")
        self.assertEqual(post.call_args.args[0], "https://example.invalid/workflow")
        self.assertEqual(regions[0].type, "text")
        self.assertEqual(regions[0].box, [5.0, 12.5, 30.0, 25.0])

    def test_local_grounding_dino_normalizes_and_deduplicates_regions(self):
        class FakeInputs(dict):
            def to(self, device):
                return self

            @property
            def input_ids(self):
                return self["input_ids_value"]

        class FakeProcessor:
            def __call__(self, **kwargs):
                return FakeInputs(pixel_values="pixels", input_ids_value="tokens")

            def post_process_grounded_object_detection(
                self,
                outputs,
                input_ids,
                threshold=0.25,
                text_threshold=0.25,
                target_sizes=None,
            ):
                return [{
                    "boxes": [
                        [0, 0, 100, 80],
                        [10, 8, 40, 32],
                        [11, 9, 39, 31],
                        [10, 8, 40, 32],
                        [50, 40, 100, 80],
                    ],
                    "scores": [0.98, 0.9, 0.8, 0.85, 0.7],
                    "text_labels": [
                        "flowchart box",
                        "flowchart box",
                        "flowchart box",
                        "flowchart arrow",
                        "handwritten text",
                    ],
                }]

        model = Mock(return_value=Mock())
        with (
            patch.dict("sys.modules", {"torch": Mock(inference_mode=nullcontext)}),
            patch.object(
                main,
                "_get_grounding_dino",
                return_value=(FakeProcessor(), model, "cpu"),
            ),
        ):
            regions = main._grounding_dino_detect(self.image_base64())

        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].type, "box")
        self.assertEqual(regions[0].box, [10.0, 10.0, 30.0, 30.0])
        self.assertEqual(regions[1].type, "text")
        self.assertEqual(regions[1].box, [50.0, 50.0, 50.0, 50.0])

    def test_grounding_label_uses_first_matching_class(self):
        self.assertEqual(
            main._grounding_region_type("flowchart boxchart arrowchart diamond"),
            "box",
        )

    def test_opencv_detect_finds_flowchart_shapes_connector_and_text(self):
        image = np.full((360, 640, 3), 255, dtype=np.uint8)
        cv2 = main.cv2
        cv2.rectangle(image, (40, 120), (190, 220), (0, 0, 0), 3)
        cv2.rectangle(image, (430, 120), (580, 220), (0, 0, 0), 3)
        cv2.putText(image, "Architecture", (245, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "START", (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, "FINISH", (450, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.line(image, (193, 170), (420, 170), (0, 0, 0), 2)
        cv2.line(image, (420, 170), (408, 160), (0, 0, 0), 2)
        cv2.line(image, (420, 170), (408, 180), (0, 0, 0), 2)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

        regions = main._opencv_detect(image_b64)
        region_types = [region.type for region in regions]

        self.assertGreaterEqual(region_types.count("box"), 2)
        self.assertIn("arrow", region_types)
        self.assertIn("text", region_types)
        self.assertTrue(
            all(region.box[2] < 80 for region in regions if region.type == "text")
        )
        graph_regions = [
            region.model_dump()
            for region in regions
            if region.type in ("box", "arrow")
        ]
        nodes, edges, mermaid = main._build_graph(graph_regions)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(edges), 1)
        self.assertIn("graph LR", mermaid)

    def test_detect_endpoint_uses_opencv_default_provider(self):
        with (
            patch.object(main, "DETECTOR_PROVIDER", "opencv"),
            patch.object(main, "_opencv_detect", return_value=[]),
        ):
            result = asyncio.run(
                main.detect_regions(file=None, clean_image_b64=self.image_base64())
            )

        self.assertEqual(result, {"regions": [], "source": "opencv"})

    def test_detect_endpoint_uses_local_provider_without_roboflow_config(self):
        with (
            patch.object(main, "DETECTOR_PROVIDER", "grounding-dino"),
            patch.object(main, "_grounding_dino_detect", return_value=[]),
        ):
            result = asyncio.run(
                main.detect_regions(file=None, clean_image_b64=self.image_base64())
            )

        self.assertEqual(result, {"regions": [], "source": "grounding-dino"})

    def test_detector_requires_configuration_instead_of_fake_regions(self):
        with (
            patch.object(main, "DETECTOR_PROVIDER", "roboflow"),
            patch.object(main, "ROBOFLOW_API_KEY", ""),
            patch.object(main, "ROBOFLOW_WORKFLOW_URL", ""),
            patch.object(main, "ALLOW_DEMO_DETECTOR_FALLBACK", False),
        ):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(main.detect_regions(file=None, clean_image_b64=self.image_base64()))
        self.assertEqual(caught.exception.status_code, 503)
        self.assertIn("ROBOFLOW_API_KEY", caught.exception.detail)

    def test_detector_request_failure_is_reported_as_gateway_error(self):
        with (
            patch.object(main, "DETECTOR_PROVIDER", "roboflow"),
            patch.object(main, "ROBOFLOW_API_KEY", "test-secret"),
            patch.object(main, "ROBOFLOW_WORKFLOW_URL", "https://example.invalid/workflow"),
            patch.object(main, "_roboflow_detect", side_effect=RuntimeError("upstream unavailable")),
        ):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(main.detect_regions(file=None, clean_image_b64=self.image_base64()))
        self.assertEqual(caught.exception.status_code, 502)

    def test_ocr_applies_model_output_to_text_regions(self):
        regions = [{
            "id": "r1", "type": "text", "box": [10, 10, 30, 20],
            "label": "handwriting", "confidence": 0.9,
        }]
        with patch.object(main, "_trocr_read", return_value="Written words") as read:
            result = main._ocr_regions(regions, self.image_base64())
        self.assertEqual(result[0]["text"], "Written words")
        self.assertEqual(result[0]["label"], "Written words")
        read.assert_called_once()

    def test_ocr_reads_text_inside_diagram_shapes(self):
        regions = [{
            "id": "r1", "type": "diamond", "box": [10, 10, 30, 20],
            "label": "flowchart diamond", "confidence": 0.9,
        }]
        with patch.object(main, "_trocr_read", return_value="Decision") as read:
            result = main._ocr_regions(regions, self.image_base64())
        self.assertEqual(result[0]["text"], "Decision")
        read.assert_called_once()

    def test_ocr_recognizes_multiline_region_as_separate_lines(self):
        image = Image.new("RGB", (500, 180), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 25), "First handwritten line", fill="black")
        draw.text((20, 115), "Second handwritten line", fill="black")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        regions = [{
            "id": "r1", "type": "text", "box": [0, 0, 100, 100],
            "label": "handwriting", "confidence": 0.9,
        }]

        with patch.object(main, "_trocr_read", side_effect=["First line", "Second line"]) as read:
            result = main._ocr_regions(regions, image_b64)

        self.assertEqual(result[0]["text"], "First line\nSecond line")
        self.assertEqual(read.call_count, 2)

    def test_multiline_text_crop_is_split_for_line_recognition(self):
        image = Image.new("RGB", (500, 180), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 25), "First handwritten line", fill="black")
        draw.text((20, 115), "Second handwritten line", fill="black")

        lines = main._split_text_lines(image)

        self.assertEqual(len(lines), 2)
        self.assertLess(lines[0].height, image.height)
        self.assertLess(lines[1].height, image.height)

    def test_ocr_does_not_hide_model_load_error(self):
        with patch.object(main, "_get_trocr", side_effect=ModuleNotFoundError("transformers")):
            with self.assertRaises(ModuleNotFoundError):
                main._trocr_read(Image.new("RGB", (30, 20), "white"))

    def test_ocr_endpoint_reports_model_failure(self):
        region = {"id": "r1", "type": "text", "box": [0, 0, 50, 50]}
        with (
            patch.object(main.importlib.util, "find_spec", return_value=object()),
            patch.object(main, "_ocr_regions", side_effect=RuntimeError("model failed")),
        ):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(main.read_text_regions(
                    regions_json=json.dumps([region]),
                    clean_image_b64=self.image_base64(),
                ))
        self.assertEqual(caught.exception.status_code, 503)

    def test_edit_distance_counts_word_and_character_edits(self):
        self.assertEqual(edit_distance(["one", "two"], ["one", "three"]), 1)
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_clean_pipeline_decodes_image_and_returns_quality(self):
        image = np.full((240, 320, 3), 245, dtype=np.uint8)
        image[40:100, 40:200] = 15
        success, encoded = main.cv2.imencode(".png", image)
        self.assertTrue(success)
        cleaned, metrics, quality = main.clean_whiteboard(encoded.tobytes())
        self.assertGreater(cleaned.size, 0)
        self.assertIn("contrastBoost", metrics)
        self.assertGreaterEqual(quality, 0)
        self.assertLessEqual(quality, 1)

    def test_graph_builder_produces_nodes_edge_and_mermaid(self):
        regions = [
            {"id": "box1", "type": "box", "box": [5, 20, 20, 20], "label": "Start"},
            {"id": "arrow1", "type": "arrow", "box": [35, 27, 10, 5], "label": "next"},
            {"id": "box2", "type": "box", "box": [55, 20, 20, 20], "label": "Finish"},
        ]
        nodes, edges, mermaid = main._build_graph(regions)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(edges), 1)
        self.assertIn("graph LR", mermaid)

    def test_graph_builder_converts_ocr_line_breaks_for_mermaid(self):
        regions = [{
            "id": "box1", "type": "box", "box": [5, 20, 20, 20],
            "label": "Start\nProcess",
        }]
        nodes, _, mermaid = main._build_graph(regions)
        self.assertEqual(nodes[0]["label"], "Start\nProcess")
        self.assertIn("Start<br/>Process", mermaid)

    def test_graph_builder_keeps_circle_and_table_nodes(self):
        regions = [
            {"id": "circle1", "type": "circle", "box": [5, 20, 20, 20], "label": "Start"},
            {"id": "table1", "type": "table", "box": [55, 20, 20, 20], "label": "Results"},
            {"id": "arrow1", "type": "arrow", "box": [35, 27, 10, 5], "label": "next"},
        ]
        nodes, edges, mermaid = main._build_graph(regions)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(edges), 1)
        self.assertIn('N0(("Start"))', mermaid)


if __name__ == "__main__":
    unittest.main()
