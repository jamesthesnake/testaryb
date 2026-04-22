from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List

import openai
import pandas as pd
from pydantic import BaseModel, Field

from backend.mana.config.env import FRED_API_KEY, OPENAI_API_KEY, PINECONE_API_KEY
from backend.mana.config.openai import GPT4_O
from backend.mana.llms.openai_model import OpenAI

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
PINECONE_INDEX_NAME = "takehome-macro-james-hennessy"


@dataclass
class FREDSeriesConfig:
    fred_id: str
    description: str
    keywords: list[str]
    search_text: str


SERIES_REGISTRY: dict[str, FREDSeriesConfig] = {
    "GDP": FREDSeriesConfig(
        fred_id="GDP",
        description="US Nominal Gross Domestic Product (Billions of Dollars, SAAR)",
        keywords=["gdp", "nominal gdp", "current gdp", "output", "gross domestic", "economic output"],
        search_text=(
            "United States GDP gross domestic product economic output recession expansion "
            "quarterly nominal GDP current dollar national accounts level"
        ),
    ),
    "REAL_GDP": FREDSeriesConfig(
        fred_id="GDPC1",
        description="US Real Gross Domestic Product (Billions of chained 2017 Dollars, SAAR)",
        keywords=["real gdp", "gdp growth", "growth", "inflation adjusted gdp", "recession", "expansion"],
        search_text=(
            "United States real GDP gross domestic product growth inflation adjusted "
            "quarterly chained dollars recession expansion national accounts"
        ),
    ),
    "CPI": FREDSeriesConfig(
        fred_id="CPIAUCSL",
        description="US Consumer Price Index — All Urban Consumers",
        keywords=["inflation", "cpi", "price", "consumer price", "cost of living", "deflation"],
        search_text=(
            "United States inflation consumer price index CPI cost of living prices "
            "deflation monthly price level"
        ),
    ),
    "EUROZONE_INFLATION": FREDSeriesConfig(
        fred_id="FPCPITOTLZGEMU",
        description="Euro Area Inflation, Consumer Prices (Annual Percent)",
        keywords=["eurozone inflation", "euro area inflation", "euro inflation", "european inflation"],
        search_text=(
            "Eurozone Euro Area EMU Europe inflation consumer prices CPI annual percent "
            "European Central Bank price stability"
        ),
    ),
    "UNEMPLOYMENT": FREDSeriesConfig(
        fred_id="UNRATE",
        description="US Unemployment Rate (Percent)",
        keywords=["unemployment", "jobs", "labor", "employment", "jobless", "workforce"],
        search_text=(
            "United States unemployment rate jobs labor market employment jobless workforce "
            "monthly percent"
        ),
    ),
    "JAPAN_UNEMPLOYMENT": FREDSeriesConfig(
        fred_id="LRUN64TTJPM156S",
        description="Japan Unemployment Rate, Ages 15 to 64 (Percent)",
        keywords=["japan unemployment", "japanese unemployment", "japan jobs", "japan labor", "japan employment"],
        search_text=(
            "Japan Japanese unemployment rate jobs labor market employment jobless workforce "
            "OECD monthly percent"
        ),
    ),
    "FEDERAL_FUNDS_RATE": FREDSeriesConfig(
        fred_id="FEDFUNDS",
        description="Federal Funds Effective Rate",
        keywords=["interest rate", "federal funds", "fed rate", "federal reserve", "monetary policy", "rate hike"],
        search_text=(
            "Federal funds effective rate Federal Reserve interest rate monetary policy "
            "rate hikes rate cuts overnight bank lending"
        ),
    ),
    "USD_EUR": FREDSeriesConfig(
        fred_id="DEXUSEU",
        description="US Dollar to Euro Exchange Rate",
        keywords=["exchange rate", "dollar", "currency", "forex", "usd", "eur/usd", "usd/eur"],
        search_text=(
            "United States dollar euro exchange rate foreign exchange forex currency USD EUR "
            "dollars per euro"
        ),
    ),
}

_EUROZONE_TERMS = ("eurozone", "euro area", "euro-area", "emu")
_JAPAN_TERMS = ("japan", "japanese")
_INFLATION_TERMS = ("inflation", "cpi", "consumer price", "prices")
_LABOR_TERMS = ("unemployment", "jobs", "labor", "employment", "jobless", "workforce")
_GDP_GROWTH_TERMS = ("growth", "grow", "real", "inflation-adjusted", "inflation adjusted", "plot")


class Citation(BaseModel):
    source: str
    series_id: str
    description: str
    url: str


class DataRetrievalIssue(BaseModel):
    source: str
    series_key: str
    series_id: str
    description: str
    error: str


class DataRetrievalError(RuntimeError):
    def __init__(self, message: str, issues: list[DataRetrievalIssue]):
        super().__init__(message)
        self.issues = issues


class MacroMetric(BaseModel):
    metric_key: str
    label: str
    series_key: str
    series_id: str
    value: float
    unit: str
    date: str
    calculation: str
    description: str


class MacroAnalysisResponse(BaseModel):
    answer: str
    citations: List[Citation]
    metrics: list[MacroMetric] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    series_key: str
    score: float
    source: str


def _fetch_fred(series_id: str, limit: int, api_key: str) -> pd.DataFrame:
    params = urllib.parse.urlencode({
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    })
    with urllib.request.urlopen(f"{FRED_BASE_URL}?{params}", timeout=15) as resp:
        data = json.loads(resp.read().decode())

    if data.get("error_message"):
        raise RuntimeError(f"FRED error: {data['error_message']}")
    if "observations" not in data:
        raise RuntimeError("FRED response did not include observations")

    df = pd.DataFrame(data["observations"])[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _series_document(key: str) -> str:
    cfg = SERIES_REGISTRY[key]
    keywords = " ".join(cfg.keywords)
    return (
        f"Series key: {key}\n"
        f"FRED series id: {cfg.fred_id}\n"
        f"Description: {cfg.description}\n"
        f"Keywords: {keywords}\n"
        f"Search text: {cfg.search_text}"
    )


def _lexical_score(query: str, document: str) -> float:
    query_terms = {term.strip(".,?!:;()[]{}").lower() for term in query.split()}
    query_terms.discard("")
    if not query_terms:
        return 0.0

    document_terms = {term.strip(".,?!:;()[]{}").lower() for term in document.split()}
    matches = query_terms.intersection(document_terms)
    return len(matches) / len(query_terms)


def _summarize_error(error: Exception) -> str:
    message = str(error).strip()
    if not message:
        return error.__class__.__name__
    return f"{error.__class__.__name__}: {message}"


def _clean_observations(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)


def _pct_change(current: float, previous: float) -> float:
    if previous == 0:
        raise ValueError("Cannot compute percent change from zero")
    return ((current / previous) - 1) * 100


def _annualized_qoq_growth(current: float, previous: float) -> float:
    if previous <= 0:
        raise ValueError("Cannot compute annualized growth from a non-positive value")
    return (((current / previous) ** 4) - 1) * 100


def _metric_value(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.2f}%"
    if unit == "USD per EUR":
        return f"{value:.4f} {unit}"
    return f"{value:,.2f} {unit}"


class MacroSpecialist:
    """
    Orchestrates FRED data retrieval and macroeconomic Q&A via OpenAI.

    Usage:
        specialist = MacroSpecialist()
        result = specialist.ask("What is the current US unemployment rate?")
        print(result.answer)
        for c in result.citations:
            print(c.url)
    """

    def __init__(self):
        self._llm = OpenAI(model=GPT4_O, stream=False, temperature=0.2)
        self._cache: dict[str, pd.DataFrame] = {}
        self._embedding_client: openai.OpenAI | None = None
        self._series_documents = {key: _series_document(key) for key in SERIES_REGISTRY}
        self._embedding_index: dict[str, list[float]] | None = None
        self._pinecone_index = None

        if PINECONE_API_KEY:
            self._init_pinecone()

    # ------------------------------------------------------------------
    # Pinecone
    # ------------------------------------------------------------------

    def _init_pinecone(self) -> None:
        """Connect to Pinecone and ensure the series index is populated."""
        try:
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=PINECONE_API_KEY)
            existing = {idx.name for idx in pc.list_indexes()}
            if PINECONE_INDEX_NAME not in existing:
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            self._pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            self._sync_pinecone_vectors()
        except Exception:
            self._pinecone_index = None

    def _sync_pinecone_vectors(self) -> None:
        """Upsert series embeddings into Pinecone if the index is empty or incomplete."""
        stats = self._pinecone_index.describe_index_stats()
        if stats.total_vector_count >= len(SERIES_REGISTRY):
            return
        keys = list(self._series_documents)
        embeddings = self._embed_texts([self._series_documents[k] for k in keys])
        self._pinecone_index.upsert(vectors=[
            {"id": key, "values": emb}
            for key, emb in zip(keys, embeddings)
        ])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_client is None:
            self._embedding_client = openai.OpenAI(max_retries=0, api_key=OPENAI_API_KEY)
        response = self._embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def _ensure_embedding_index(self) -> dict[str, list[float]]:
        if self._embedding_index is None:
            keys = list(self._series_documents)
            embeddings = self._embed_texts([self._series_documents[key] for key in keys])
            self._embedding_index = dict(zip(keys, embeddings))
        return self._embedding_index

    def _query_embedding_index(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        """Return top-k series by embedding similarity, using Pinecone when available."""
        if self._pinecone_index is not None:
            try:
                response = self._pinecone_index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_values=False,
                )
                return [
                    RetrievalResult(series_key=match.id, score=match.score, source="pinecone")
                    for match in response.matches
                ]
            except Exception:
                pass  # fall through to in-memory

        index = self._ensure_embedding_index()
        return sorted(
            (
                RetrievalResult(
                    series_key=key,
                    score=_cosine_similarity(query_embedding, emb),
                    source="embedding",
                )
                for key, emb in index.items()
            ),
            key=lambda r: r.score,
            reverse=True,
        )[:top_k]

    def _guardrail_series(self, query: str) -> list[str]:
        q = query.lower()
        matched: list[str] = []

        if "gdp" in q:
            if any(term in q for term in _GDP_GROWTH_TERMS):
                matched.append("REAL_GDP")
            else:
                matched.append("GDP")

        # Geography-specific routes must win over the generic US CPI/labor routes.
        if any(term in q for term in _EUROZONE_TERMS) and any(term in q for term in _INFLATION_TERMS):
            matched.append("EUROZONE_INFLATION")
        if any(term in q for term in _JAPAN_TERMS) and any(term in q for term in _LABOR_TERMS):
            matched.append("JAPAN_UNEMPLOYMENT")

        return matched

    def retrieve_series(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Retrieve relevant FRED series using embedding search (Pinecone or in-memory)."""
        guardrails = self._guardrail_series(query)
        try:
            query_embedding = self._embed_texts([query])[0]
            ranked = self._query_embedding_index(query_embedding, top_k=len(SERIES_REGISTRY))
        except Exception:
            ranked = sorted(
                (
                    RetrievalResult(
                        series_key=key,
                        score=_lexical_score(query, document),
                        source="lexical-fallback",
                    )
                    for key, document in self._series_documents.items()
                ),
                key=lambda result: result.score,
                reverse=True,
            )

        selected: list[RetrievalResult] = []
        for key in guardrails:
            result = next(
                (item for item in ranked if item.series_key == key),
                RetrievalResult(series_key=key, score=1.0, source="guardrail"),
            )
            selected.append(result.model_copy(update={"source": f"{result.source}+guardrail"}))

        for result in ranked:
            if result.series_key not in {item.series_key for item in selected}:
                selected.append(result)
            if len(selected) >= top_k:
                break

        return selected

    def _relevant_series(self, query: str) -> list[str]:
        return [result.series_key for result in self.retrieve_series(query)]

    def _load_series(self, key: str, limit: int = 10) -> pd.DataFrame:
        cache_key = f"{key}:{limit}"
        if cache_key not in self._cache:
            cfg = SERIES_REGISTRY[key]
            self._cache[cache_key] = _fetch_fred(cfg.fred_id, limit, api_key=FRED_API_KEY or "")
        return self._cache[cache_key]

    def _retrieval_issue(self, key: str, error: Exception) -> DataRetrievalIssue:
        cfg = SERIES_REGISTRY[key]
        return DataRetrievalIssue(
            source="FRED — Federal Reserve Bank of St. Louis",
            series_key=key,
            series_id=cfg.fred_id,
            description=cfg.description,
            error=_summarize_error(error),
        )

    def _metric(
            self,
            key: str,
            metric_key: str,
            label: str,
            value: float,
            unit: str,
            date: str,
            calculation: str,
    ) -> MacroMetric:
        cfg = SERIES_REGISTRY[key]
        return MacroMetric(
            metric_key=metric_key,
            label=label,
            series_key=key,
            series_id=cfg.fred_id,
            value=float(value),
            unit=unit,
            date=date,
            calculation=calculation,
            description=cfg.description,
        )

    def _series_metrics(self, key: str, df: pd.DataFrame) -> list[MacroMetric]:
        clean = _clean_observations(df)
        if clean.empty:
            return []

        latest = clean.iloc[-1]
        latest_value = float(latest["value"])
        latest_date = str(latest["date"])
        metrics: list[MacroMetric] = []

        if key == "GDP":
            metrics.append(self._metric(
                key,
                "nominal_gdp_level",
                "US nominal GDP level",
                latest_value,
                "billions of dollars, SAAR",
                latest_date,
                "Latest FRED GDP observation; nominal dollars, seasonally adjusted annual rate.",
            ))
            if len(clean) >= 5:
                previous = clean.iloc[-5]
                metrics.append(self._metric(
                    key,
                    "nominal_gdp_yoy_growth",
                    "US nominal GDP growth, YoY",
                    _pct_change(latest_value, float(previous["value"])),
                    "%",
                    latest_date,
                    f"Percent change from {previous['date']} to {latest_date}.",
                ))
            return metrics

        if key == "REAL_GDP":
            metrics.append(self._metric(
                key,
                "real_gdp_level",
                "US real GDP level",
                latest_value,
                "billions of chained 2017 dollars, SAAR",
                latest_date,
                "Latest FRED GDPC1 observation; inflation-adjusted dollars.",
            ))
            if len(clean) >= 2:
                previous = clean.iloc[-2]
                metrics.append(self._metric(
                    key,
                    "real_gdp_qoq_annualized_growth",
                    "US real GDP growth, QoQ annualized",
                    _annualized_qoq_growth(latest_value, float(previous["value"])),
                    "%",
                    latest_date,
                    f"Annualized percent change from {previous['date']} to {latest_date}.",
                ))
            if len(clean) >= 5:
                previous = clean.iloc[-5]
                metrics.append(self._metric(
                    key,
                    "real_gdp_yoy_growth",
                    "US real GDP growth, YoY",
                    _pct_change(latest_value, float(previous["value"])),
                    "%",
                    latest_date,
                    f"Percent change from {previous['date']} to {latest_date}.",
                ))
            return metrics

        if key == "CPI":
            metrics.append(self._metric(
                key,
                "cpi_index_level",
                "US CPI index level",
                latest_value,
                "index",
                latest_date,
                "Latest CPIAUCSL observation.",
            ))
            if len(clean) >= 13:
                previous = clean.iloc[-13]
                metrics.append(self._metric(
                    key,
                    "us_cpi_yoy_inflation",
                    "US CPI inflation rate, YoY",
                    _pct_change(latest_value, float(previous["value"])),
                    "%",
                    latest_date,
                    f"Percent change from {previous['date']} to {latest_date}.",
                ))
            return metrics

        if key == "EUROZONE_INFLATION":
            return [self._metric(
                key,
                "eurozone_inflation_rate",
                "Euro area inflation rate",
                latest_value,
                "%",
                latest_date,
                "Latest annual consumer price inflation observation from FRED.",
            )]

        if key in {"UNEMPLOYMENT", "JAPAN_UNEMPLOYMENT", "FEDERAL_FUNDS_RATE"}:
            metric_key = {
                "UNEMPLOYMENT": "us_unemployment_rate",
                "JAPAN_UNEMPLOYMENT": "japan_unemployment_rate",
                "FEDERAL_FUNDS_RATE": "federal_funds_effective_rate",
            }[key]
            label = {
                "UNEMPLOYMENT": "US unemployment rate",
                "JAPAN_UNEMPLOYMENT": "Japan unemployment rate",
                "FEDERAL_FUNDS_RATE": "Federal funds effective rate",
            }[key]
            return [self._metric(
                key,
                metric_key,
                label,
                latest_value,
                "%",
                latest_date,
                "Latest FRED rate observation.",
            )]

        if key == "USD_EUR":
            return [self._metric(
                key,
                "usd_per_euro_exchange_rate",
                "US dollar to euro exchange rate",
                latest_value,
                "USD per EUR",
                latest_date,
                "Latest FRED DEXUSEU observation.",
            )]

        return [self._metric(
            key,
            f"{key.lower()}_latest_value",
            SERIES_REGISTRY[key].description,
            latest_value,
            "value",
            latest_date,
            "Latest FRED observation.",
        )]

    def _build_context(self, query: str) -> tuple[str, list[Citation], list[MacroMetric], list[str]]:
        parts: list[str] = []
        citations: list[Citation] = []
        metrics: list[MacroMetric] = []
        issues: list[DataRetrievalIssue] = []

        for key in self._relevant_series(query):
            cfg = SERIES_REGISTRY[key]
            try:
                df = self._load_series(key, limit=24)
                clean = _clean_observations(df)
                if clean.empty:
                    issues.append(self._retrieval_issue(key, ValueError("FRED returned no numeric observations")))
                    continue
                series_metrics = self._series_metrics(key, clean)
                metrics.extend(series_metrics)
                metric_rows = "\n".join(
                    f"  {metric.label}: {_metric_value(metric.value, metric.unit)} as of {metric.date}. "
                    f"Calculation: {metric.calculation}"
                    for metric in series_metrics
                )
                recent = clean.tail(5).sort_values("date", ascending=False)
                rows = "\n".join(f"  {r['date']}: {r['value']}" for _, r in recent.iterrows())
                parts.append(
                    f"{cfg.description} ({cfg.fred_id}):\n"
                    f"Derived metrics:\n{metric_rows}\n"
                    f"Recent observations:\n{rows}"
                )
                citations.append(Citation(
                    source="FRED — Federal Reserve Bank of St. Louis",
                    series_id=cfg.fred_id,
                    description=cfg.description,
                    url=f"https://fred.stlouisfed.org/series/{cfg.fred_id}",
                ))
            except Exception as exc:
                issues.append(self._retrieval_issue(key, exc))

        if not parts:
            raise DataRetrievalError(
                "No FRED data could be retrieved for the selected macroeconomic series.",
                issues,
            )

        warnings = [
            f"{issue.description} ({issue.series_id}) could not be retrieved: {issue.error}"
            for issue in issues
        ]
        return "\n\n".join(parts), citations, metrics, warnings

    # ------------------------------------------------------------------
    # Q&A
    # ------------------------------------------------------------------

    def ask(self, query: str) -> MacroAnalysisResponse:
        """Answer a macroeconomic question using live FRED data and OpenAI."""
        context, citations, metrics, warnings = self._build_context(query)
        system_prompt = (
            "You are a macroeconomics specialist with deep expertise in economic indicators, "
            "monetary policy, and global financial markets. Use the provided deterministic "
            "metrics and FRED observations to give precise, data-driven answers. Treat derived "
            "metrics as authoritative for rates and growth calculations. Reference specific "
            "values, units, and dates when relevant. "
            "If data warnings are provided, mention the missing sources and do not infer values "
            "for unavailable series."
        )
        warning_context = ""
        if warnings:
            warning_context = "Data Warnings:\n" + "\n".join(f"- {warning}" for warning in warnings) + "\n\n"
        user_prompt = (
            f"Economic Data (FRED):\n{context}\n\n"
            f"{warning_context}"
            f"Question: {query}"
        )
        answer = self._llm.complete(user_prompt=user_prompt, system_prompt=system_prompt)
        return MacroAnalysisResponse(answer=answer, citations=citations, metrics=metrics, warnings=warnings)

    # ------------------------------------------------------------------
    # Data access (for visualizations)
    # ------------------------------------------------------------------

    def get_series_df(self, series_key: str, limit: int = 60) -> pd.DataFrame:
        """
        Return a time-ordered DataFrame for a named FRED series.

        Args:
            series_key: One of the keys in SERIES_REGISTRY
            limit: Number of observations to return (most recent first, then sorted asc)
        Returns:
            DataFrame with columns: date (str), value (float)
        """
        cfg = SERIES_REGISTRY[series_key]
        df = _fetch_fred(cfg.fred_id, limit=limit, api_key=FRED_API_KEY or "")
        return df.sort_values("date").reset_index(drop=True)

    def list_series(self) -> dict[str, tuple[str, str]]:
        """Return the available series as {key: (fred_id, description)}."""
        return {key: (cfg.fred_id, cfg.description) for key, cfg in SERIES_REGISTRY.items()}
