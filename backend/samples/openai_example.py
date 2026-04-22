from backend.mana.config.openai import GPT4_O
from backend.mana.llms.openai_model import OpenAI


def sample_openai_usage():
    llm = OpenAI(model=GPT4_O, temperature=0.0, stream=True)
    return llm.complete("What is r*star?")



if __name__ == "__main__":
    result = sample_openai_usage()
    print(result)