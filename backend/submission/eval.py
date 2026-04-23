"""
Evaluation harness for MacroSpecialist.

Runs the 7 canonical spec questions, scores each response with a GPT-4o judge,
and writes results to backend/eval_results/<timestamp>.json.

Usage:
    # from repo root
    PYTHONPATH=. poetry --directory backend run python -m backend.submission.eval
    PYTHONPATH=. poetry --directory backend run python -m backend.submission.eval --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from backend.mana.config.openai import GPT4_O
from backend.mana.llms.openai_model import InstructorOpenAI
from backend.submission.macro_specialist import MacroAnalysisResponse, MacroSpecialist

RESULTS_DIR = Path(__file__).parent.parent / "eval_results"


@dataclass
class EvalCase:
    question: str
    expected_series: list[str]
    criteria: list[str]


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        question="What is the current GDP of the US?",
        expected_series=["GDP", "REAL_GDP"],
        criteria=[
            "Includes a specific dollar value for US GDP",
            "States the unit (billions of dollars or similar)",
            "References a recent date or quarter",
        ],
    ),
    EvalCase(
        question="What is the inflation rate in the Eurozone?",
        expected_series=["EUROZONE_INFLATION"],
        criteria=[
            "States a specific inflation percentage for the Eurozone or Euro Area",
            "References consumer prices or CPI",
            "Includes a recent date",
        ],
    ),
    EvalCase(
        question="What is the unemployment rate in Japan?",
        expected_series=["JAPAN_UNEMPLOYMENT"],
        criteria=[
            "States a specific unemployment percentage for Japan",
            "Does not confuse Japan figures with US unemployment",
            "Includes a recent date",
        ],
    ),
    EvalCase(
        question="What is the current interest rate set by the Federal Reserve?",
        expected_series=["FEDERAL_FUNDS_RATE"],
        criteria=[
            "States the Federal Funds Rate as a specific percentage",
            "Attributes the rate to the Federal Reserve",
            "Includes a recent date",
        ],
    ),
    EvalCase(
        question="What is the current exchange rate between the US Dollar and the Euro?",
        expected_series=["USD_EUR"],
        criteria=[
            "States a specific USD/EUR exchange rate value",
            "Correctly identifies the direction (USD per EUR or EUR per USD)",
            "Includes a recent date",
        ],
    ),
    EvalCase(
        question="What has been the GDP growth rate in the US over the past 10 years?",
        expected_series=["REAL_GDP"],
        criteria=[
            "Discusses real (inflation-adjusted) GDP growth rather than nominal",
            "Includes specific growth percentages or a trend description",
            "References multiple time periods or a trend over time",
        ],
    ),
    EvalCase(
        question="What has been the US unemployment rate trend over the past 20 years?",
        expected_series=["UNEMPLOYMENT"],
        criteria=[
            "Describes the trend in US unemployment over an extended period",
            "Mentions notable turning points such as the financial crisis or COVID",
            "Includes specific rate values at key points",
        ],
    ),
]


class JudgeScore(BaseModel):
    score: int = Field(
        ..., ge=1, le=5,
        description="Overall quality: 1=poor, 2=weak, 3=adequate, 4=good, 5=excellent",
    )
    grounded_in_data: bool = Field(
        ...,
        description="Answer draws on the provided FRED metrics and citations rather than hallucinating values",
    )
    cites_correct_series: bool = Field(
        ...,
        description="Answer references the expected FRED series or an equivalent",
    )
    contains_specific_values: bool = Field(
        ...,
        description="Answer includes specific numeric values with units and dates",
    )
    reasoning: str = Field(
        ...,
        description="One or two sentences explaining the score and any shortcomings",
    )


class EvalResult(BaseModel):
    question: str
    expected_series: list[str]
    cited_series: list[str]
    answer: str
    metrics_count: int
    warnings: list[str]
    judge: JudgeScore
    specialist_model: str
    judge_model: str
    timestamp: str


class EvalSummary(BaseModel):
    specialist_model: str
    judge_model: str
    timestamp: str
    mean_score: float
    pass_rate: float
    results: list[EvalResult]


def _build_judge_prompt(case: EvalCase, response: MacroAnalysisResponse) -> str:
    metrics_text = "\n".join(
        f"  {m.label}: {m.value} {m.unit} as of {m.date}"
        for m in response.metrics
    ) or "  (none)"

    citations_text = "\n".join(
        f"  {c.series_id}: {c.description}"
        for c in response.citations
    ) or "  (none)"

    criteria_text = "\n".join(f"  - {c}" for c in case.criteria)

    return f"""You are evaluating a response from an AI macroeconomics assistant.

Question: {case.question}

Expected FRED series: {", ".join(case.expected_series)}

Data provided to the assistant:
Citations:
{citations_text}
Metrics:
{metrics_text}

Assistant answer:
{response.answer}

Evaluation criteria (what a good answer must satisfy):
{criteria_text}

Score the response 1–5 and fill in the boolean fields."""


def _score(
    case: EvalCase,
    response: MacroAnalysisResponse,
    judge_model: str,
) -> JudgeScore:
    llm = InstructorOpenAI(
        model=judge_model,
        response_model=JudgeScore,
        temperature=0.0,
    )
    return llm.complete(user_prompt=_build_judge_prompt(case, response))


def run_eval(
    specialist_model: str = GPT4_O,
    judge_model: str = GPT4_O,
) -> EvalSummary:
    specialist = MacroSpecialist(model=specialist_model)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results: list[EvalResult] = []

    for i, case in enumerate(EVAL_CASES, 1):
        print(f"[{i}/{len(EVAL_CASES)}] {case.question}")
        try:
            response = specialist.ask(case.question)
            judge = _score(case, response, judge_model)
            result = EvalResult(
                question=case.question,
                expected_series=case.expected_series,
                cited_series=[c.series_id for c in response.citations],
                answer=response.answer,
                metrics_count=len(response.metrics),
                warnings=response.warnings,
                judge=judge,
                specialist_model=specialist_model,
                judge_model=judge_model,
                timestamp=timestamp,
            )
            print(f"  score={judge.score}/5  grounded={judge.grounded_in_data}  specific_values={judge.contains_specific_values}")
            print(f"  {judge.reasoning}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            result = EvalResult(
                question=case.question,
                expected_series=case.expected_series,
                cited_series=[],
                answer=f"ERROR: {exc}",
                metrics_count=0,
                warnings=[str(exc)],
                judge=JudgeScore(
                    score=1,
                    grounded_in_data=False,
                    cites_correct_series=False,
                    contains_specific_values=False,
                    reasoning=f"Evaluation could not complete due to error: {exc}",
                ),
                specialist_model=specialist_model,
                judge_model=judge_model,
                timestamp=timestamp,
            )
        results.append(result)

    scores = [r.judge.score for r in results]
    summary = EvalSummary(
        specialist_model=specialist_model,
        judge_model=judge_model,
        timestamp=timestamp,
        mean_score=round(sum(scores) / len(scores), 2),
        pass_rate=round(sum(1 for s in scores if s >= 4) / len(scores), 2),
        results=results,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{timestamp.replace(':', '-')}.json"
    out_path.write_text(summary.model_dump_json(indent=2))

    print(f"\nMean score : {summary.mean_score:.2f}/5")
    print(f"Pass rate  : {summary.pass_rate:.0%}  (score >= 4)")
    print(f"Results    : {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MacroSpecialist response quality")
    parser.add_argument("--model", default=GPT4_O, help="Model used by MacroSpecialist (default: gpt-4o)")
    parser.add_argument("--judge-model", default=GPT4_O, help="Model used by the judge (default: gpt-4o)")
    args = parser.parse_args()
    run_eval(specialist_model=args.model, judge_model=args.judge_model)
