"""Tests for the required local audio-folder batch command."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run


class FakeAgent:
    def respond(self, session_id: str, message: str) -> dict:
        self.session_id = session_id
        self.message = message
        return {"answer": "Please check the relevant policy. [RET-2]", "decision": "answer",
                "sources": [{"id": "RET-2", "title": "Returns"}], "tool_calls": [],
                "engine": "openai_rag"}


class BatchRunnerTests(unittest.TestCase):
    def test_batch_saves_transcript_reply_audio_and_source_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir, output_dir = root / "in", root / "out"
            input_dir.mkdir()
            (input_dir / "01 question.mp3").write_bytes(b"synthetic-audio")
            fake_agent = FakeAgent()
            with patch.dict("os.environ", {"GNANI_API_KEY": "test-key"}, clear=True), \
                    patch.object(run.voice_adapter, "transcribe_file", return_value="Can I return an item?"), \
                    patch.object(run.voice_adapter, "synthesize_stream", return_value=b"reply-wav"), \
                    patch.object(run, "AGENT", fake_agent):
                succeeded, failed = run.process_folder(input_dir, output_dir)
            self.assertEqual((succeeded, failed), (1, 0))
            self.assertEqual((output_dir / "01_question.txt").read_text(encoding="utf-8").count("Can I return"), 1)
            self.assertEqual((output_dir / "01_question.wav").read_bytes(), b"reply-wav")
            result = json.loads((output_dir / "01_question.json").read_text(encoding="utf-8"))
            self.assertEqual(result["sources"][0]["id"], "RET-2")
            self.assertEqual(result["engine"], "openai_rag")
            self.assertEqual(fake_agent.message, "Can I return an item?")

    def test_batch_requires_gnani_key_before_processing(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GNANI_API_KEY"):
                run.process_folder(Path(temp), Path(temp) / "out")


if __name__ == "__main__":
    unittest.main()
