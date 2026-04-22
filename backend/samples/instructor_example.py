from pydantic import BaseModel, Field

from backend.mana.config.openai import GPT4_O
from backend.mana.llms.openai_model import InstructorOpenAI
from backend.mana.data.load_data import load_sec_filing



class SecFilingSummaryModel(BaseModel):
    ticker: str = Field(..., description="Ticker of interest")
    filing_type: str = Field(..., description="SEC filing form type of interest")
    summary: str = Field(..., description="Summary of SEC Filing")


def generate_secfiling_summary(ticker, filing_type) -> SecFilingSummaryModel:
    filings = load_sec_filing(ticker, filing_type)
    instructions = """Summarize the latest sec filing about the ticker below. List timeline and topics covered.
                        Ticker: {}
                        Filing Type: {}
                        SEC Filings: {}
                    """.format(ticker, filing_type, filings[:50000])

    llm = InstructorOpenAI(model=GPT4_O, response_model=SecFilingSummaryModel)
    return llm.complete(system_prompt=instructions, user_prompt="Summarize sec filings of the ticker")


if __name__ == "__main__":
    ticker = "NVDA"
    filing_type = "10-K"
    summary = generate_secfiling_summary(ticker, filing_type)
    print(summary)