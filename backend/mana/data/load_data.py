import pandas.io.json

from backend.mana.config.env import DATA_FOLDER


def load_sec_filing(ticker, type) -> pandas.DataFrame:
    path = DATA_FOLDER / "filings" / ticker / f"secfilings_{ticker}_{type}.json"
    return pandas.io.json.read_json(path)


def load_one_sided_estimates() -> str:
    path = DATA_FOLDER / "natural_rate" / "one_sided_estimates.csv"
    return path.read_text()

def load_two_sided_estimates() -> str:
    path = DATA_FOLDER / "natural_rate" / "two_sided_estimates.csv"
    return path.read_text()




if __name__ == '__main__':
    print(load_sec_filing("AAPL", "10-K"))
    print(load_one_sided_estimates())
    print(load_two_sided_estimates())