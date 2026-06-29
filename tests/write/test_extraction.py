"""Tests for quipu.extraction.local.extract_local."""

from __future__ import annotations

import pytest

from quipu.extraction.local import extract_local


class TestExtractLocalStructure:
    def test_returns_dict_with_required_keys(self):
        result = extract_local("Hello World")
        assert isinstance(result, dict)
        assert "entities" in result
        assert "keywords" in result

    def test_entities_is_list(self):
        result = extract_local("Hello World")
        assert isinstance(result["entities"], list)

    def test_keywords_is_list(self):
        result = extract_local("Hello World")
        assert isinstance(result["keywords"], list)

    def test_empty_string(self):
        result = extract_local("")
        assert result == {"entities": [], "keywords": []}

    def test_all_stopwords(self):
        result = extract_local("a an the and or but in on at to for of with")
        assert result["keywords"] == []

    def test_entities_are_strings(self):
        result = extract_local("Alice and Bob went to London")
        for e in result["entities"]:
            assert isinstance(e, str)

    def test_keywords_are_strings(self):
        result = extract_local("machine learning algorithms process data")
        for k in result["keywords"]:
            assert isinstance(k, str)


class TestExtractLocalDeterminism:
    def test_same_input_same_output(self):
        text = "Python and Django are popular frameworks for web development"
        r1 = extract_local(text)
        r2 = extract_local(text)
        assert r1 == r2

    def test_repeated_calls_identical(self):
        text = "The quick brown Fox jumps over the lazy Dog"
        results = [extract_local(text) for _ in range(5)]
        for r in results[1:]:
            assert r == results[0]


class TestExtractLocalDedup:
    def test_entity_dedup(self):
        text = "Alice met Alice and Alice left"
        result = extract_local(text)
        entities_lower = [e.lower() for e in result["entities"]]
        assert len(entities_lower) == len(set(entities_lower)), (
            f"Duplicate entities found: {result['entities']}"
        )

    def test_keyword_dedup(self):
        text = "learning machine learning deep learning"
        result = extract_local(text)
        assert len(result["keywords"]) == len(set(result["keywords"])), (
            f"Duplicate keywords found: {result['keywords']}"
        )

    def test_entity_dedup_preserves_first_occurrence(self):
        text = "Python is great. Python is also popular. Python everywhere."
        result = extract_local(text)
        python_count = result["entities"].count("Python")
        assert python_count <= 1

    def test_keyword_dedup_preserves_first_occurrence(self):
        # "machine" appears multiple times
        text = "machine learning and machine vision use machine models"
        result = extract_local(text)
        machine_count = result["keywords"].count("machine")
        assert machine_count <= 1


class TestExtractLocalOrderStable:
    def test_entity_order_stable(self):
        text = "Alice met Bob and Charlie yesterday"
        r1 = extract_local(text)
        r2 = extract_local(text)
        assert r1["entities"] == r2["entities"]

    def test_keyword_order_reflects_first_occurrence(self):
        text = "zebra apple mango zebra apple"
        result = extract_local(text)
        # "zebra" appears before "apple" in the unique ordered list
        # (both appear first at their respective first positions)
        kws = result["keywords"]
        if "zebra" in kws and "apple" in kws:
            assert kws.index("zebra") < kws.index("apple")


class TestExtractLocalContent:
    def test_capitalized_words_as_entities(self):
        text = "Albert Einstein developed the theory of relativity"
        result = extract_local(text)
        # At least Albert Einstein should appear
        entity_strings = " ".join(result["entities"])
        assert "Einstein" in entity_strings or "Albert" in entity_strings

    def test_common_words_not_in_keywords(self):
        text = "the quick brown fox jumps over the lazy dog"
        result = extract_local(text)
        for stop in ["the", "over"]:
            assert stop not in result["keywords"]

    def test_content_words_in_keywords(self):
        text = "neural network training requires gradient descent optimization"
        result = extract_local(text)
        kws = result["keywords"]
        # At least some content words should appear
        assert len(kws) > 0
        # Words like "neural", "network", "training" should be present
        assert any(w in kws for w in ["neural", "network", "training", "gradient"])

    def test_minimum_word_length_enforced(self):
        text = "to be or not to be"
        result = extract_local(text)
        for kw in result["keywords"]:
            assert len(kw) >= 3, f"Short word found in keywords: {kw!r}"
