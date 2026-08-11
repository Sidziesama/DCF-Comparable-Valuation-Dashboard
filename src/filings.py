import os
import requests
import pandas as pd


SEC_SUBMISSIONS_URL = (
    "https://data.sec.gov/submissions/CIK{cik}.json"
)


def _sec_headers():
    user_agent = os.getenv("SEC_USER_AGENT")

    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is missing from .env"
        )

    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def fetch_submissions(cik: str) -> dict:
    """
    Fetch SEC filing metadata for a company.
    """

    url = SEC_SUBMISSIONS_URL.format(
        cik=str(cik).zfill(10)
    )

    response = requests.get(
        url,
        headers=_sec_headers(),
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def build_recent_filings(cik: str) -> pd.DataFrame:
    """
    Return recent SEC filings as a normalized dataframe.
    """

    submissions = fetch_submissions(cik)

    recent = submissions[
        "filings"
    ]["recent"]

    filings = pd.DataFrame(recent)

    columns = [
        "accessionNumber",
        "filingDate",
        "reportDate",
        "form",
        "primaryDocument",
    ]

    filings = filings[
        [c for c in columns if c in filings.columns]
    ].copy()

    filings["filingDate"] = pd.to_datetime(
        filings["filingDate"],
        errors="coerce",
    )

    filings["reportDate"] = pd.to_datetime(
        filings["reportDate"],
        errors="coerce",
    )

    return filings.sort_values(
        "filingDate",
        ascending=False,
    )


def get_latest_filing(
    cik: str,
    form: str,
) -> dict:
    """
    Get latest filing of requested form.
    """

    filings = build_recent_filings(cik)

    matches = filings[
        filings["form"] == form
    ]

    if matches.empty:
        raise RuntimeError(
            f"No {form} filing found."
        )

    return matches.iloc[0].to_dict()


def get_latest_10q(cik: str) -> dict:

    return get_latest_filing(
        cik,
        "10-Q",
    )


def get_latest_10k(cik: str) -> dict:

    return get_latest_filing(
        cik,
        "10-K",
    )


def print_filing_summary(cik: str):

    latest_10q = get_latest_10q(cik)
    latest_10k = get_latest_10k(cik)

    print("\n==============================")
    print("LATEST SEC FILINGS")
    print("==============================")

    print(
        f"Latest 10-Q report date : "
        f"{latest_10q['reportDate'].date()}"
    )

    print(
        f"Latest 10-Q filing date : "
        f"{latest_10q['filingDate'].date()}"
    )

    print(
        f"Latest 10-Q accession   : "
        f"{latest_10q['accessionNumber']}"
    )

    print()

    print(
        f"Latest 10-K report date : "
        f"{latest_10k['reportDate'].date()}"
    )

    print(
        f"Latest 10-K filing date : "
        f"{latest_10k['filingDate'].date()}"
    )


if __name__ == "__main__":

    from dotenv import load_dotenv

    load_dotenv()

    print_filing_summary(
        "0001403161"
    )