from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import citation_ids, validate_citations
from retriever import ArchiveIndex, Article


class RetrievalTests(unittest.TestCase):
    def test_specific_words_rank_matching_article(self):
        # These short strings test ranking code only. Course evidence comes
        # from the downloaded BBC archive and the three evaluation cases in app.py.
        articles = [
            Article("article-0000", "tech", "mobile phone virus bluetooth"),
            Article("article-0001", "sport", "football cup final"),
        ]
        results = ArchiveIndex(articles).search("bluetooth phone virus", 1)
        self.assertEqual(results[0][0].article_id, "article-0000")


class CitationTests(unittest.TestCase):
    def test_extracts_stable_citations(self):
        self.assertEqual(
            citation_ids("The archive says this [article-0000]."),
            {"article-0000"},
        )

    def test_rejects_unretrieved_citation(self):
        valid, invalid = validate_citations(
            "See [article-9999].", {"article-0000"}
        )
        self.assertFalse(valid)
        self.assertEqual(invalid, {"article-9999"})

    def test_requires_at_least_one_citation(self):
        valid, invalid = validate_citations("No citation.", {"article-0000"})
        self.assertFalse(valid)
        self.assertEqual(invalid, set())


if __name__ == "__main__":
    unittest.main()

