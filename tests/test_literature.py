"""Tests for src/dvc_behavior/literature.py (fully offline)."""

from __future__ import annotations

import json

import pytest

from dvc_behavior import literature as lit


def _payload(tables: dict | None = None, highlights: dict | None = None) -> dict:
    return {"tables": tables or {}, "highlights": highlights or {}}


class TestBuildQueries:
    def test_always_includes_domain_anchor(self):
        queries = lit.build_literature_queries(_payload())
        assert queries[0] == lit._DOMAIN_ANCHOR
        assert all(isinstance(q, str) and q for q in queries)

    def test_circadian_and_light_dark_terms_added_when_present(self):
        payload = _payload(
            tables={"circadian_summary": {}, "light_dark_summary": {}, "activity_bouts": {}}
        )
        queries = lit.build_literature_queries(payload, max_queries=10)
        joined = " | ".join(queries)
        assert "circadian" in joined
        assert "light dark" in joined
        assert "bout" in joined

    def test_rhythm_detected_via_highlights_only(self):
        payload = _payload(highlights={"circadian_summary": {"rhythms": [{"amplitude": 0.4}]}})
        queries = lit.build_literature_queries(payload)
        assert any("circadian" in q for q in queries)

    def test_respects_max_queries_and_dedupes(self):
        payload = _payload(
            tables={"circadian_summary": {}, "light_dark_summary": {}, "stats_summary": {}}
        )
        queries = lit.build_literature_queries(payload, max_queries=2)
        assert len(queries) == 2
        assert len(set(queries)) == len(queries)

    def test_no_study_specifics_leak_into_queries(self):
        # Group names / ids in the payload must never appear in the queries.
        payload = _payload(
            highlights={"light_dark_summary": {"groups": ["SECRET_KO_LINE"], "light_dark": [{}]}}
        )
        queries = lit.build_literature_queries(payload)
        assert all("SECRET_KO_LINE" not in q for q in queries)


class TestParseEuropePMC:
    SAMPLE = {
        "resultList": {
            "result": [
                {
                    "id": "12345",
                    "source": "MED",
                    "pmid": "12345",
                    "doi": "10.1/abc",
                    "title": "Circadian locomotor rhythms in mice.",
                    "authorString": "Doe J, Smith A.",
                    "journalTitle": "J Biol Rhythms",
                    "pubYear": "2020",
                },
                {
                    "id": "PPR1",
                    "source": "PPR",
                    "title": "A preprint without DOI",
                    "pubYear": "2023",
                },
            ]
        }
    }

    def test_parses_fields_and_builds_url(self):
        refs = lit._parse_europepmc_result(self.SAMPLE)
        assert len(refs) == 2
        first = refs[0]
        assert first.title == "Circadian locomotor rhythms in mice"  # trailing dot stripped
        assert first.doi == "10.1/abc"
        assert first.url == "https://doi.org/10.1/abc"
        assert first.year == "2020"

    def test_empty_or_malformed_is_safe(self):
        assert lit._parse_europepmc_result({}) == []
        assert lit._parse_europepmc_result({"resultList": {"result": [None, 5]}}) == []


class _FakeProvider:
    name = "fake"

    def __init__(self, mapping: dict[str, list[lit.Reference]]):
        self.mapping = mapping
        self.calls: list[str] = []

    def search(self, query: str, max_results: int) -> list[lit.Reference]:
        self.calls.append(query)
        return self.mapping.get(query, [])


class TestFindSupportingLiterature:
    def test_offline_default_returns_queries_no_refs(self):
        result = lit.find_supporting_literature(_payload(tables={"circadian_summary": {}}))
        assert result.provider == "offline"
        assert result.references == []
        assert result.queries  # queries still recorded for transparency
        assert result.disclaimer == lit.LITERATURE_DISCLAIMER

    def test_dedupes_across_queries_and_caps(self):
        shared = lit.Reference(title="Shared", doi="10.1/x")
        payload = _payload(tables={"circadian_summary": {}, "light_dark_summary": {}})
        queries = lit.build_literature_queries(payload)
        mapping = {
            queries[0]: [shared, lit.Reference(title="A", pmid="1")],
            queries[1]: [shared, lit.Reference(title="B", pmid="2")],
        }
        provider = _FakeProvider(mapping)
        result = lit.find_supporting_literature(payload, provider, max_results=10)
        keys = [r.key() for r in result.references]
        assert len(keys) == len(set(keys))  # no duplicate "Shared"
        assert any(r.title == "Shared" for r in result.references)

    def test_max_results_cap(self):
        payload = _payload(tables={"circadian_summary": {}})
        q = lit.build_literature_queries(payload)[0]
        many = [lit.Reference(title=f"T{i}", pmid=str(i)) for i in range(10)]
        provider = _FakeProvider({q: many})
        result = lit.find_supporting_literature(payload, provider, max_results=3)
        assert len(result.references) == 3

    def test_result_is_json_serializable(self):
        provider = _FakeProvider({})
        result = lit.find_supporting_literature(_payload(), provider)
        json.dumps(result.to_dict())  # must not raise
        assert "Related literature" in result.to_markdown()


class TestEuropePMCProvider:
    def test_search_uses_http_seam(self, monkeypatch):
        captured = {}

        def fake_get(url, params, timeout=15.0):
            captured["url"] = url
            captured["params"] = params
            return TestParseEuropePMC.SAMPLE

        monkeypatch.setattr(lit, "_http_get_json", fake_get)
        provider = lit.EuropePMCProvider()
        refs = provider.search("circadian rhythm rodent", max_results=5)

        assert captured["url"] == lit.EUROPEPMC_SEARCH_URL
        assert captured["params"]["query"] == "circadian rhythm rodent"
        assert captured["params"]["format"] == "json"
        assert len(refs) == 2

    def test_http_failure_raises_runtime_error(self, monkeypatch):
        def boom(url, params, timeout=15.0):
            raise RuntimeError("Literature search request failed: boom")

        monkeypatch.setattr(lit, "_http_get_json", boom)
        provider = lit.EuropePMCProvider()
        with pytest.raises(RuntimeError):
            lit.find_supporting_literature(
                {"tables": {"circadian_summary": {}}}, provider
            )
