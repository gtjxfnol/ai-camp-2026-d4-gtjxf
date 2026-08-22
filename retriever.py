"""TF-IDF retrieval over the real BBC news archive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Article:
    article_id: str
    category: str
    text: str


def load_articles(path: Path) -> list[Article]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Real BBC archive not found at {path}. "
            "Follow the real-data section of the starter README and download tfidf_dataset.csv from Kaggle."
        )
    frame = pd.read_csv(path)
    missing = {"text", "category"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    if len(frame) != 2225:
        raise ValueError(
            f"Expected 2,225 BBC documents but found {len(frame)}. "
            "Check the named Kaggle file."
        )
    categories = set(frame["category"].astype(str))
    expected_categories = {"business", "entertainment", "politics", "sport", "tech"}
    if categories != expected_categories:
        raise ValueError(
            f"Expected BBC categories {sorted(expected_categories)} but found "
            f"{sorted(categories)}"
        )
    if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("The archive contains a missing or empty article text")
    articles = [
        Article(
            article_id=f"article-{int(source_row):04d}",
            category=str(row["category"]),
            text=str(row["text"]),
        )
        for source_row, row in frame.iterrows()
    ]
    return articles


class ArchiveIndex:
    def __init__(self, articles: list[Article]) -> None:
        if not articles:
            raise ValueError("The archive contains no articles")
        self.articles = articles
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(
            article.text for article in articles
        )

    @classmethod
    def from_csv(cls, path: Path) -> "ArchiveIndex":
        return cls(load_articles(path))

    def search(
        self, question: str, top_k: int = 4
    ) -> list[tuple[Article, float]]:
        """Return top articles in descending similarity order."""
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked = sorted(
            enumerate(scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]
        return [
            (self.articles[index], float(score)) for index, score in ranked
        ]
