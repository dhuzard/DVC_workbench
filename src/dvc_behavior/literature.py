"""Optional literature grounding for DVC insight summaries.

This module turns the *kinds of findings* in an analysis payload into a few
generic, domain-level search queries (e.g. "circadian rhythm locomotor activity
rodent") and looks up related open-access references via Europe PMC.

Privacy model (consistent with ``insights.py``):

* It is **opt-in** and **off by default** — nothing here runs unless the caller
  explicitly invokes it with a network provider.
* Only **generic keyword queries** leave the machine. No raw data, no per-animal
  values, and no study-specific labels (group names, file names, subject ids)
  are ever sent.
* The default :class:`NullLiteratureProvider` is fully offline (returns no
  references) so the feature degrades gracefully and the tests need no network.
* Results are framed as **automated suggestions to verify**, never as evidence
  that confirms a finding.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

LITERATURE_DISCLAIMER = (
    "Automated literature suggestions based on generic topic keywords only. They "
    "are a starting point for a manual search, not evidence for or against your "
    "result. Read and judge each reference yourself before citing it."
)

# Generic, domain-level anchor always included so a query is never study-specific.
_DOMAIN_ANCHOR = "mouse home-cage locomotor activity"

EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Reference:
    """A single literature reference (open metadata only)."""

    title: str
    authors: str = ""
    year: str = ""
    journal: str = ""
    doi: str = ""
    pmid: str = ""
    source: str = ""
    url: str = ""

    def key(self) -> str:
        """Stable identity for de-duplication."""
        return (self.doi or self.pmid or self.title).strip().lower()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiteratureResult:
    """Outcome of a literature lookup, with the queries used for transparency."""

    queries: list[str]
    references: list[Reference] = field(default_factory=list)
    provider: str = "offline"
    generated_at: str = ""
    disclaimer: str = LITERATURE_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "queries": list(self.queries),
            "references": [r.to_dict() for r in self.references],
            "provider": self.provider,
            "generated_at": self.generated_at,
            "disclaimer": self.disclaimer,
        }

    def to_markdown(self) -> str:
        return format_references_markdown(self)


# --------------------------------------------------------------------------- #
# Query construction (pure, deterministic — no network, no study specifics)
# --------------------------------------------------------------------------- #
def build_literature_queries(payload: dict | None, *, max_queries: int = 4) -> list[str]:
    """Derive generic topic queries from the *shape* of an insight payload.

    Only the presence of certain analysis tables/highlights influences the
    queries; concrete values, group names and identifiers are never used.
    """
    tables = (payload or {}).get("tables", {}) or {}
    highlights = (payload or {}).get("highlights", {}) or {}
    present = set(tables) | set(highlights)

    def _has_rhythm() -> bool:
        if {"circadian_summary", "nonparametric_circadian", "period_estimate"} & present:
            return True
        circ = highlights.get("circadian_summary", {}) or {}
        return bool(circ.get("rhythms"))

    def _has_light_dark() -> bool:
        if "light_dark_summary" in present:
            return True
        ld = highlights.get("light_dark_summary", {}) or {}
        return bool(ld.get("light_dark"))

    queries: list[str] = [_DOMAIN_ANCHOR]
    if _has_rhythm():
        queries.append("circadian rhythm locomotor activity rodent")
    if _has_light_dark():
        queries.append("light dark phase activity mouse")
    if "activity_bouts" in present:
        queries.append("activity bout fragmentation rodent behavior")
    if {"stats_summary", "auc_summary", "daily_means"} & present:
        queries.append("home-cage activity phenotype mouse behavioral assay")

    # De-duplicate preserving order, then cap.
    seen: set[str] = set()
    unique = [q for q in queries if not (q in seen or seen.add(q))]
    return unique[: max(1, max_queries)]


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
@runtime_checkable
class LiteratureProvider(Protocol):
    name: str

    def search(
        self, query: str, max_results: int
    ) -> list[Reference]:  # pragma: no cover - protocol
        ...


class NullLiteratureProvider:
    """Offline default: records queries but fetches nothing (no egress)."""

    name = "offline"

    def search(self, query: str, max_results: int) -> list[Reference]:
        return []


def _http_get_json(url: str, params: dict[str, Any], timeout: float = 15.0) -> dict:
    """Lazy, isolated HTTP GET returning parsed JSON.

    Kept as a module-level seam so tests can monkeypatch it without network and
    so ``requests`` stays an optional, lazily-imported dependency.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "Literature search needs the optional 'requests' package. Install it "
            "with `pip install requests`, or keep the offline default."
        ) from exc
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # noqa: BLE001 - normalize to a guidance error
        raise RuntimeError(f"Literature search request failed: {exc}") from exc


class EuropePMCProvider:
    """Query Europe PMC (open, no API key). Sends only the keyword query string."""

    name = "europepmc"

    def __init__(self, *, endpoint: str = EUROPEPMC_SEARCH_URL, timeout: float = 15.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def search(self, query: str, max_results: int) -> list[Reference]:
        params = {
            "query": query,
            "format": "json",
            "pageSize": int(max(1, max_results)),
            "resultType": "lite",
        }
        data = _http_get_json(self.endpoint, params, timeout=self.timeout)
        return _parse_europepmc_result(data)


def _parse_europepmc_result(data: dict) -> list[Reference]:
    """Parse a Europe PMC search response into :class:`Reference` objects."""
    results = (((data or {}).get("resultList") or {}).get("result")) or []
    references: list[Reference] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        doi = str(item.get("doi") or "")
        pmid = str(item.get("pmid") or item.get("id") or "")
        if doi:
            url = f"https://doi.org/{doi}"
        elif pmid and str(item.get("source", "")).upper() == "MED":
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        else:
            url = ""
        references.append(
            Reference(
                title=str(item.get("title") or "").strip().rstrip("."),
                authors=str(item.get("authorString") or "").strip(),
                year=str(item.get("pubYear") or "").strip(),
                journal=str(item.get("journalTitle") or "").strip(),
                doi=doi,
                pmid=pmid,
                source=str(item.get("source") or "").strip(),
                url=url,
            )
        )
    return references


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def find_supporting_literature(
    payload: dict | None,
    provider: LiteratureProvider | None = None,
    *,
    max_results: int = 5,
    max_queries: int = 4,
) -> LiteratureResult:
    """Run the (opt-in) literature lookup and return a de-duplicated result.

    Defaults to the offline :class:`NullLiteratureProvider`.  Pass an
    :class:`EuropePMCProvider` to actually fetch references; only the generic
    keyword queries are sent.
    """
    provider = provider or NullLiteratureProvider()
    queries = build_literature_queries(payload, max_queries=max_queries)

    seen: set[str] = set()
    references: list[Reference] = []
    for query in queries:
        try:
            hits = provider.search(query, max_results)
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep one bad query from aborting all
            raise RuntimeError(f"Literature search failed for '{query}': {exc}") from exc
        for ref in hits:
            k = ref.key()
            if k and k not in seen:
                seen.add(k)
                references.append(ref)

    references = references[: max(1, max_results)]
    return LiteratureResult(
        queries=queries,
        references=references,
        provider=getattr(provider, "name", "unknown"),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def format_references_markdown(result: LiteratureResult) -> str:
    """Render a literature result as Markdown for display or export."""
    lines = ["## Related literature (automated suggestions)", ""]
    lines.append("**Queries used:** " + "; ".join(f"`{q}`" for q in result.queries))
    lines.append("")
    if not result.references:
        lines.append("_No references fetched (offline mode, or the search returned nothing)._")
    else:
        for ref in result.references:
            cite = ref.title or "(untitled)"
            meta = ", ".join(p for p in (ref.authors, ref.journal, ref.year) if p)
            link = f" <{ref.url}>" if ref.url else ""
            lines.append(f"- **{cite}**" + (f" — {meta}" if meta else "") + link)
    lines.append("")
    lines.append(f"_{result.disclaimer}_")
    return "\n".join(lines)


def literature_payload_json(result: LiteratureResult) -> str:
    """Serialize a literature result for the export bundle."""
    return json.dumps(result.to_dict(), indent=2, sort_keys=True)


__all__ = [
    "EUROPEPMC_SEARCH_URL",
    "EuropePMCProvider",
    "LITERATURE_DISCLAIMER",
    "LiteratureProvider",
    "LiteratureResult",
    "NullLiteratureProvider",
    "Reference",
    "build_literature_queries",
    "find_supporting_literature",
    "format_references_markdown",
    "literature_payload_json",
]
