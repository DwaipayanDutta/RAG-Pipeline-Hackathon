#!/usr/bin/env python3
"""
Tests for the Jev-augmented RAG pipeline.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

# Import the pipeline under test
import rag_pipeline_jev as pipeline


class TestJevConfiguration(unittest.TestCase):
    """Test that Jev configuration is correctly parsed."""

    def test_jev_defaults(self):
        config = pipeline.load_config()
        self.assertFalse(config["use_jev"])
        self.assertEqual(config["jev_threshold"], 0.2)
        self.assertEqual(config["jev_api_key_env"], "TYPESAFE_API_KEY")

    def test_jev_cli_overrides(self):
        # Simulate argparse overrides
        config = pipeline.DEFAULT_CONFIG.copy()
        config["use_jev"] = True
        config["jev_threshold"] = 0.5
        self.assertTrue(config["use_jev"])
        self.assertEqual(config["jev_threshold"], 0.5)


class TestJevGating(unittest.TestCase):
    """Test the Jev gating method in Retriever."""

    def setUp(self):
        self.config = pipeline.DEFAULT_CONFIG.copy()
        self.config["use_jev"] = True
        self.config["jev_threshold"] = 0.2

        # Create dummy chunks
        self.chunks = [
            pipeline.Chunk(chunk_id=0, page=1, section=None, source_type="text", text="Revenue was high."),
            pipeline.Chunk(chunk_id=1, page=2, section=None, source_type="text", text="Profit was low."),
            pipeline.Chunk(chunk_id=2, page=3, section=None, source_type="text", text="Irrelevant passage."),
        ]

        # Mock the JevReranker
        self.mock_jev = MagicMock()
        self.mock_jev.relevance_rerank.return_value = {
            "results": [
                {"text": "Revenue was high.", "score": 0.9, "document_index": 0},
                {"text": "Profit was low.", "score": 0.8, "document_index": 1},
                # "Irrelevant passage." is below threshold and omitted
            ]
        }

    @patch("rag_pipeline_jev.JEV_AVAILABLE", True)
    @patch("rag_pipeline_jev.JevReranker")
    def test_jev_gate_filters_chunks(self, mock_reranker_cls):
        mock_reranker_cls.return_value = self.mock_jev
        retriever = pipeline.Retriever(self.config)
        # Inject mock directly
        retriever.jev_reranker = self.mock_jev

        kept_chunks, scores = retriever.jev_gate("What was revenue?", self.chunks)

        self.assertEqual(len(kept_chunks), 2)
        self.assertIn(kept_chunks[0].text, ["Revenue was high.", "Profit was low."])
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0]["score"], 0.9)
        self.assertEqual(scores[1]["score"], 0.8)

    @patch("rag_pipeline_jev.JEV_AVAILABLE", True)
    @patch("rag_pipeline_jev.JevReranker")
    def test_jev_gate_all_filtered(self, mock_reranker_cls):
        self.mock_jev.relevance_rerank.return_value = {"results": []}
        mock_reranker_cls.return_value = self.mock_jev
        retriever = pipeline.Retriever(self.config)
        retriever.jev_reranker = self.mock_jev

        kept_chunks, scores = retriever.jev_gate("Unknown question", self.chunks)

        self.assertEqual(len(kept_chunks), 0)
        self.assertEqual(len(scores), 0)

    @patch("rag_pipeline_jev.JEV_AVAILABLE", False)
    def test_jev_not_available(self):
        retriever = pipeline.Retriever(self.config)
        self.assertIsNone(retriever.jev_reranker)

    def test_jev_disabled_by_default(self):
        config = pipeline.DEFAULT_CONFIG.copy()
        retriever = pipeline.Retriever(config)
        self.assertIsNone(retriever.jev_reranker)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the full pipeline (mocked LLM and Jev)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pdf_path = os.path.join(self.temp_dir, "dummy.pdf")
        # Create a minimal dummy PDF placeholder (actual extraction is mocked)
        with open(self.pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n%dummy")

        self.config = pipeline.DEFAULT_CONFIG.copy()
        self.config["use_jev"] = True
        self.config["jev_threshold"] = 0.2
        self.config["index_dir"] = os.path.join(self.temp_dir, "index")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("rag_pipeline_jev.extract_pdf_content")
    @patch("rag_pipeline_jev.Retriever.jev_gate")
    @patch("rag_pipeline_jev.generate_answer")
    def test_pipeline_jev_gating_path(self, mock_generate, mock_jev_gate, mock_extract):
        # Setup mocks
        mock_extract.return_value = (
            [{"page": 1, "text": "Titan revenue was 5000 crore in FY2025-26."}],
            []
        )
        mock_jev_gate.return_value = (
            [pipeline.Chunk(chunk_id=0, page=1, section=None, source_type="text",
                            text="Titan revenue was 5000 crore in FY2025-26.")],
            [{"chunk_id": 0, "score": 0.95, "index": 0}]
        )
        mock_generate.return_value = ("Titan revenue was 5000 crore.", [{"page": 1}])

        result = pipeline.run_pipeline(
            self.pdf_path,
            "What was Titan's revenue?",
            self.config,
            rebuild=True
        )

        self.assertTrue(result.grounded)
        self.assertIsNotNone(result.jev_scores)
        self.assertEqual(len(result.jev_scores), 1)
        self.assertEqual(result.jev_scores[0]["score"], 0.95)

    @patch("rag_pipeline_jev.extract_pdf_content")
    @patch("rag_pipeline_jev.Retriever.jev_gate")
    def test_pipeline_jev_refuses_when_no_evidence(self, mock_jev_gate, mock_extract):
        mock_extract.return_value = ([{"page": 1, "text": "Some text."}], [])
        mock_jev_gate.return_value = ([], [])  # All chunks filtered

        result = pipeline.run_pipeline(
            self.pdf_path,
            "What was the revenue?",
            self.config,
            rebuild=True
        )

        self.assertFalse(result.grounded)
        self.assertEqual(result.answer, "I don't know from the document.")
        self.assertIn("Jev", result.refusal_reason)
        self.assertEqual(result.jev_scores, [])


if __name__ == "__main__":
    unittest.main()