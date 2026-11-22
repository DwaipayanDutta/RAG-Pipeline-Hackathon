#!/usr/bin/env python3
"""
Unit tests for rag_pipeline.py (Standard RAG over the Titan annual report).

These target the functions rag_pipeline.py actually defines. The previous version of this file
imported names that do not exist (clean_text, build_chunks_from_pages, get_retriever, ...), so
it failed at collection time. No model weights are downloaded: only pure functions and the CLI
helpers are exercised.

Run:  cd RAG_Pipeline_Simple && python -m pytest test_rag_pipeline.py -v
"""
import json
import os
import tempfile
import unittest

import rag_pipeline as rp
from rag_pipeline import (Chunk, EvidenceRecord, QueryResult, chunk_text, classify_query, enrich_chunks,
                          extract_entities, extract_fiscal_years, extract_metrics, load_config, load_queries,
                          normalize_text, parse_financial_value, parse_threshold, validate_answer)


class TestTextUtilities(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  Revenue   GREW\n\t21% "), "revenue grew 21%")

    def test_fiscal_years_are_canonicalised(self):
        self.assertEqual(set(extract_fiscal_years("FY2025-26 vs FY 24-25 and 2023-24")),
                         {"FY2025-26", "FY2024-25", "FY2023-24"})

    def test_entities_and_metrics(self):
        self.assertIn("CaratLane", extract_entities("Tanishq and CaratLane grew"))
        self.assertEqual(extract_metrics("revenue and profit before tax"), ["revenue", "profit_before_tax"])


class TestParseFinancialValue(unittest.TestCase):
    def test_thousands_are_not_truncated(self):
        # Regression: commas were stripped before a \d{1,3}(?:,\d{3})* match, so 51,084 became 510.
        self.assertEqual(parse_financial_value("₹51,084 crore"), (51084.0, "INR crore", False))
        self.assertEqual(parse_financial_value("5000 crore")[0], 5000.0)

    def test_percentage_and_empty(self):
        self.assertEqual(parse_financial_value("21.5%"), (21.5, "percent", True))
        self.assertEqual(parse_financial_value(""), (None, None, False))


class TestParseThreshold(unittest.TestCase):
    def test_exceeding_with_unit(self):
        result = parse_threshold("exceeding 5000 crore")
        self.assertEqual((result["operator"], result["threshold"], result["unit"]), (">", 5000.0, "INR crore"))
        self.assertEqual(parse_threshold("exceeding INR 5,000 crore")["threshold"], 5000.0)

    def test_percentages_and_operators(self):
        self.assertEqual(parse_threshold("above 10%")["unit"], "percent")
        self.assertEqual(parse_threshold("at least 1000 crore")["operator"], ">=")
        self.assertEqual(parse_threshold("below 5%")["operator"], "<")

    def test_invalid(self):
        self.assertIsNone(parse_threshold("some random text"))


class TestClassifyQuery(unittest.TestCase):
    def test_routes(self):
        cases = {
            "exceeding 5000 crore": "THRESHOLD",
            "greater than 10%": "THRESHOLD",
            "Which divisions recorded double-digit growth?": "THRESHOLD",
            "compare revenue of Titan and CaratLane": "COMPARISON",
            "which business had highest revenue": "RANKING",
            "what is the total revenue": "AGGREGATION",
            "what is the growth rate": "NUMERIC",
            "what were the key factors": "FACTUAL",
        }
        for query, expected in cases.items():
            self.assertEqual(classify_query(query), expected, query)


class TestChunking(unittest.TestCase):
    def test_paragraphs_are_packed_with_page_metadata(self):
        chunks = chunk_text([{"page": 3, "text": "Para one.\n\nPara two."}, {"page": 4, "text": ""}], 400, 80)
        self.assertEqual([(c.chunk_id, c.page, c.text) for c in chunks], [(0, 3, "Para one.\nPara two.")])
        self.assertEqual(chunks[0].metadata["page"], 3)

    def test_long_text_is_split_with_overlap(self):
        # NOTE: chunk_size/overlap are applied to *characters* in this pipeline.
        paras = "\n\n".join(f"Paragraph {i} " + "word " * 15 for i in range(6))
        chunks = chunk_text([{"page": 1, "text": paras}], chunk_size=200, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertEqual([c.chunk_id for c in chunks], list(range(len(chunks))))
        self.assertTrue(chunks[1].text.startswith(chunks[0].text[-20:].strip()[:5]))


class TestValidateAnswer(unittest.TestCase):
    def setUp(self):
        self.chunks = enrich_chunks([Chunk(0, 1, None, "text", "Titan revenue was 10,000 crore in FY2025-26.")])

    def test_grounded_answer(self):
        grounded, validation, _ = validate_answer("Titan revenue was 10,000 crore in FY2025-26.", self.chunks)
        self.assertTrue(grounded)
        self.assertTrue(validation["numeric_grounding"])

    def test_unsupported_number(self):
        grounded, _, reason = validate_answer("Titan revenue was 5,000 crore in FY2025-26.", self.chunks)
        self.assertFalse(grounded)
        self.assertIn("Numeric", reason)

    def test_un_comma_d_number_is_one_claim(self):
        chunks = enrich_chunks([Chunk(0, 1, None, "text", "Titan revenue was 5000 crore in FY2025-26.")])
        self.assertTrue(validate_answer("Titan revenue was 5000 crore in FY2025-26.", chunks)[0])


class TestDataStructuresAndConfig(unittest.TestCase):
    def test_dataclasses(self):
        rec = EvidenceRecord(entity="Titan", metric="revenue", fiscal_year="FY2025-26", value=10000.0,
                             unit="INR crore", page=1, chunk_id=0)
        self.assertEqual((rec.entity, rec.value, rec.source_type), ("Titan", 10000.0, "text"))
        res = QueryResult(query="q", query_type="FACTUAL", answer="a", grounded=True, grounding_status="grounded",
                          confidence=1.0, citations=[], retrieved_chunks=[], validation={})
        self.assertIsNone(res.refusal_reason)

    def test_load_config_overrides(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"final_top_k": 3}, f)
        try:
            cfg = load_config(f.name)
            self.assertEqual(cfg["final_top_k"], 3)
            self.assertEqual(cfg["chunk_size"], rp.DEFAULT_CONFIG["chunk_size"])
        finally:
            os.unlink(f.name)


class TestCliHelpers(unittest.TestCase):
    def test_load_queries_accepts_strings_and_objects(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(["q1", {"query": "q2"}], f)
        try:
            self.assertEqual(load_queries(f.name), ["q1", "q2"])
        finally:
            os.unlink(f.name)

    def test_repo_queries_file_and_default_pdf_exist(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertEqual(len(load_queries(os.path.join(repo, "Data and Question", "queries.json"))), 11)
        self.assertTrue(os.path.isfile(rp.DEFAULT_PDF))


if __name__ == "__main__":
    unittest.main()
