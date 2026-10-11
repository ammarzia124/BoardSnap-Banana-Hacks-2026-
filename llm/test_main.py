import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

import main


class GeminiClientTests(unittest.TestCase):
    def test_missing_key_returns_service_unavailable(self):
        with patch.object(main, "gemini_client", None):
            with self.assertRaises(HTTPException) as caught:
                main.generate_json("prompt")

        self.assertEqual(caught.exception.status_code, 503)

    def test_json_generation_uses_configured_model_and_json_mode(self):
        client = Mock()
        client.models.generate_content.return_value.text = '{"answer": "ok"}'

        with (
            patch.object(main, "gemini_client", client),
            patch.object(main, "MODEL_NAME", "test-model"),
        ):
            result = main.generate_json("board prompt")

        self.assertEqual(result, {"answer": "ok"})
        client.models.generate_content.assert_called_once_with(
            model="test-model",
            contents="board prompt",
            config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
            },
        )

    def test_invalid_model_response_is_reported_as_gateway_error(self):
        client = Mock()
        client.models.generate_content.return_value.text = "not JSON"

        with patch.object(main, "gemini_client", client):
            with self.assertRaises(HTTPException) as caught:
                main.generate_json("board prompt")

        self.assertEqual(caught.exception.status_code, 502)

    def test_health_reports_model_without_exposing_the_api_key(self):
        with (
            patch.object(main, "gemini_client", object()),
            patch.object(main, "MODEL_NAME", "test-model"),
        ):
            result = main.health()

        self.assertTrue(result["gemini_configured"])
        self.assertEqual(result["gemini_model"], "test-model")
        self.assertNotIn("api_key", result)


if __name__ == "__main__":
    unittest.main()
