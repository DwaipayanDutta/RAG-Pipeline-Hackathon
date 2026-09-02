import unittest
import tempfile
import os
from rag_pipeline import (
    normalize_text, safe_float, extract_fiscal_years, extract_entities,
    extract_metrics, extract_numeric_value, parse_threshold,
    classify_query, handle_threshold_query, EvidenceRecord, Chunk
)

class TestUtilities(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  Hello  World!  "), "hello world!")

    def test_safe_float(self):
        self.assertEqual(safe_float("1,234.56"), 1234.56)
        self.assertEqual(safe_float("₹5,000 crore"), 5000.0)
        self.assertIsNone(safe_float("abc"))

    def test_extract_fiscal_years(self):
        text = "FY2025-26 and 2024-25"
        years = extract_fiscal_years(text)
        self.assertIn("FY2025-26", years)
        self.assertIn("FY2024-25", years)

    def test_extract_entities(self):
        text = "Titan Company and Tanishq reported growth."
        ents = extract_entities(text)
        self.assertIn("Titan Company", ents)
        self.assertIn("Tanishq", ents)

    def test_extract_metrics(self):
        text = "Revenue grew 10% and PBT increased."
        metrics = extract_metrics(text)
        self.assertIn("revenue", metrics)
        self.assertIn("profit_before_tax", metrics)

    def test_extract_numeric_value(self):
        self.assertEqual(extract_numeric_value("Revenue 5,000 crore"), 5000.0)
        self.assertEqual(extract_numeric_value("growth 12.5%"), 12.5)

    def test_parse_threshold(self):
        parsed = parse_threshold("exceeding 5,000 crore")
        self.assertEqual(parsed["operator"], ">")
        self.assertEqual(parsed["threshold"], 5000.0)
        self.assertEqual(parsed["unit"], "INR crore")

        parsed = parse_threshold("above 10%")
        self.assertEqual(parsed["operator"], ">")
        self.assertEqual(parsed["threshold"], 10.0)
        self.assertEqual(parsed["unit"], "percent")

class TestQueryClassification(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify_query("exceeding 5000 crore"), "THRESHOLD")
        self.assertEqual(classify_query("compare FY2025 and FY2024"), "COMPARISON")
        self.assertEqual(classify_query("which business had highest revenue"), "RANKING")
        self.assertEqual(classify_query("total revenue"), "AGGREGATION")
        self.assertEqual(classify_query("what is the revenue of Titan?"), "NUMERIC")
        self.assertEqual(classify_query("Tell me about the company"), "FACTUAL")

class TestThresholdQuery(unittest.TestCase):
    def test_threshold(self):
        # Create mock chunks with evidence
        rec1 = EvidenceRecord(entity="Titan", metric="revenue", fiscal_year="FY2025-26", value=10000, unit="INR crore", page=1, chunk_id=1)
        rec2 = EvidenceRecord(entity="CaratLane", metric="revenue", fiscal_year="FY2025-26", value=3000, unit="INR crore", page=2, chunk_id=2)
        chunk1 = Chunk(chunk_id=1, page=1, section=None, source_type="table", text="", metadata={}, evidence_records=[rec1])
        chunk2 = Chunk(chunk_id=2, page=2, section=None, source_type="table", text="", metadata={}, evidence_records=[rec2])
        answer, citations, ev = handle_threshold_query("exceeding 5000 crore", [chunk1, chunk2])
        self.assertIsNotNone(answer)
        self.assertIn("Titan", answer)
        self.assertNotIn("CaratLane", answer)

if __name__ == "__main__":
    unittest.main()