import re
import requests
import pandas as pd
from sec_data import ALIASES

import warnings

from bs4 import (
    BeautifulSoup,
    XMLParsedAsHTMLWarning,
)

warnings.filterwarnings(
    "ignore",
    category=XMLParsedAsHTMLWarning,
)

# =========================================================
# SEC filing URL
# =========================================================

def build_filing_url(
    cik: str,
    accession_number: str,
    primary_document: str,
) -> str:
    """
    Construct the SEC EDGAR URL for a filing's primary document.
    """

    cik_clean = str(int(cik))

    accession_clean = accession_number.replace(
        "-",
        "",
    )

    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_clean}/"
        f"{accession_clean}/"
        f"{primary_document}"
    )


# =========================================================
# Fetch filing
# =========================================================

def fetch_filing_html(
    filing: dict,
    cik: str,
    user_agent: str,
) -> str:

    url = build_filing_url(
        cik=cik,
        accession_number=filing["accessionNumber"],
        primary_document=filing["primaryDocument"],
    )

    response = requests.get(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.text


# =========================================================
# XBRL utility functions
# =========================================================

def local_name(name):
    """
    Convert:
        us-gaap:RevenueFromContract...
    into:
        RevenueFromContract...
    """

    if name is None:
        return None

    return str(name).split(":")[-1]


def clean_numeric_value(text):
    """
    Convert XBRL text into float.
    """

    if text is None:
        return None

    text = (
        str(text)
        .replace(",", "")
        .replace("$", "")
        .strip()
    )

    if text in {
        "",
        "-",
        "—",
        "–",
    }:
        return 0.0

    # Parentheses indicate negatives.
    negative = (
        text.startswith("(")
        and text.endswith(")")
    )

    text = text.replace(
        "(",
        "",
    ).replace(
        ")",
        "",
    )

    try:
        value = float(text)

    except ValueError:
        return None

    if negative:
        value *= -1

    return value


def apply_scale(
    value,
    scale,
    sign,
):
    """
    Inline XBRL values can carry scale attributes.

    Example:
        value=40
        scale=9

    means:
        40 * 10^9
    """

    if value is None:
        return None

    if scale is not None:

        try:
            value *= 10 ** int(scale)

        except (ValueError, TypeError):
            pass

    if sign == "-":
        value *= -1

    return value


# =========================================================
# Context extraction
# =========================================================

def extract_contexts(soup):
    """
    Extract XBRL reporting contexts.

    Also detect dimensional contexts. For core financial
    statement metrics we generally want consolidated facts,
    not disaggregated facts such as geography, revenue type,
    client incentives, etc.
    """

    contexts = {}

    for tag in soup.find_all():

        tag_name = (
            tag.name.lower()
            if tag.name
            else ""
        )

        if not tag_name.endswith("context"):
            continue

        context_id = (
            tag.get("id")
            or tag.get("ID")
        )

        if not context_id:
            continue

        start_date = None
        end_date = None
        instant = None

        has_dimensions = False

        for child in tag.find_all():

            child_name = (
                child.name.lower()
                if child.name
                else ""
            )

            if child_name.endswith("startdate"):
                start_date = child.get_text(
                    strip=True
                )

            elif child_name.endswith("enddate"):
                end_date = child.get_text(
                    strip=True
                )

            elif child_name.endswith("instant"):
                instant = child.get_text(
                    strip=True
                )

            # XBRL dimensional members
            elif (
                child_name.endswith(
                    "explicitmember"
                )
                or child_name.endswith(
                    "typedmember"
                )
            ):
                has_dimensions = True

        contexts[context_id] = {
            "start": start_date,
            "end": end_date,
            "instant": instant,
            "has_dimensions": has_dimensions,
        }

    return contexts

# =========================================================
# Metric map
# =========================================================

def build_tag_to_metric_map():

    mapping = {}

    for metric, aliases in ALIASES.items():

        for alias in aliases:

            mapping[alias] = metric

    return mapping


# =========================================================
# Filing metadata
# =========================================================

def extract_dei_value(
    soup,
    target_name,
):
    """
    Extract DEI filing metadata such as:
      DocumentFiscalYearFocus
      DocumentFiscalPeriodFocus
    """

    for tag in soup.find_all():

        name = tag.get("name")

        if not name:
            continue

        if local_name(name) != target_name:
            continue

        value = tag.get_text(
            strip=True
        )

        if value:
            return value

    return None


# =========================================================
# Inline XBRL parser
# =========================================================

def extract_inline_xbrl(
    html: str,
    filing: dict,
) -> pd.DataFrame:

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    contexts = extract_contexts(
        soup
    )

    tag_to_metric = (
        build_tag_to_metric_map()
    )

    fiscal_year = extract_dei_value(
        soup,
        "DocumentFiscalYearFocus",
    )

    fiscal_period = extract_dei_value(
        soup,
        "DocumentFiscalPeriodFocus",
    )

    try:
        fiscal_year = int(
            fiscal_year
        )

    except (
        ValueError,
        TypeError,
    ):
        fiscal_year = None

    observations = []

    for tag in soup.find_all():

        tag_name = (
            tag.name.lower()
            if tag.name
            else ""
        )

        # Inline numeric XBRL fact
        if not (
            tag_name.endswith(
                "nonfraction"
            )
            or tag_name.endswith(
                "fraction"
            )
        ):
            continue

        xbrl_name = tag.get(
            "name"
        )

        if not xbrl_name:
            continue

        sec_tag = local_name(
            xbrl_name
        )

        metric = tag_to_metric.get(
            sec_tag
        )

        if metric is None:
            continue

        context_ref = (
            tag.get("contextref")
            or tag.get("contextRef")
        )

        context = contexts.get(
            context_ref,
            {},
        )
        # Skip dimensional / disaggregated facts.
        #
        # Example:
        # Visa may use the same accounting concept inside
        # contexts describing client incentives, geography,
        # or revenue categories. Those should not replace
        # consolidated financial statement values.

        if context.get(
            "has_dimensions",
            False,
        ):
            continue

        value = clean_numeric_value(
            tag.get_text(
                strip=True
            )
        )

        value = apply_scale(
            value=value,
            scale=tag.get("scale"),
            sign=tag.get("sign"),
        )

        if value is None:
            continue

        start = context.get(
            "start"
        )

        end = (
            context.get("end")
            or context.get("instant")
        )

        observations.append(
            {
                "metric": metric,
                "value": value,
                "unit": tag.get(
                    "unitref",
                    "USD",
                ),
                "fy": fiscal_year,
                "fp": fiscal_period,
                "form": filing["form"],
                "filed": filing[
                    "filingDate"
                ],
                "start": start,
                "end": end,
                "frame": None,
                "xbrl_tag": sec_tag,
                "source": "filing_xbrl",
            }
        )

    if not observations:
        raise RuntimeError(
            "No mapped Inline XBRL financial facts "
            "were extracted from the filing."
        )

    df = pd.DataFrame(
        observations
    )

    # -----------------------------------------------------
    # Values into $ millions
    # -----------------------------------------------------

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df["value"] = (
        df["value"]
        / 1_000_000
    )

    return df


# =========================================================
# Remove duplicate facts
# =========================================================

def deduplicate_filing_facts(
    df: pd.DataFrame,
) -> pd.DataFrame:

    data = df.copy()

    data["start"] = pd.to_datetime(
        data["start"],
        errors="coerce",
    )

    data["end"] = pd.to_datetime(
        data["end"],
        errors="coerce",
    )

    data["filed"] = pd.to_datetime(
        data["filed"],
        errors="coerce",
    )

    data = (
        data
        .sort_values(
            [
                "metric",
                "start",
                "end",
                "filed",
            ]
        )
        .drop_duplicates(
            subset=[
                "metric",
                "start",
                "end",
                "value",
            ],
            keep="last",
        )
    )

    return data.reset_index(
        drop=True
    )

def fetch_latest_filing_facts(
    cik: str,
    filing: dict,
    user_agent: str,
) -> pd.DataFrame:

    print(
        "\nFetching direct filing XBRL..."
    )

    print(
        "Filing:",
        filing["form"],
    )

    print(
        "Report date:",
        filing["reportDate"],
    )

    html = fetch_filing_html(
        filing=filing,
        cik=cik,
        user_agent=user_agent,
    )

    facts = extract_inline_xbrl(
        html=html,
        filing=filing,
    )

    facts = deduplicate_filing_facts(
        facts
    )

    print(
        f"Extracted {len(facts)} "
        "mapped filing facts."
    )

    return facts

if __name__ == "__main__":

    import os

    from dotenv import load_dotenv

    from filings import get_latest_10q

    load_dotenv()

    cik = "0001403161"

    user_agent = os.getenv(
        "SEC_USER_AGENT"
    )

    latest_10q = get_latest_10q(
        cik
    )

    facts = fetch_latest_filing_facts(
        cik=cik,
        filing=latest_10q,
        user_agent=user_agent,
    )

    recent = facts[
        facts["metric"].isin(
            [
                "revenue",
                "operating_income",
                "net_income",
                "tax_expense",
                "capex",
                "cfo",
                "cash",
                "total_assets",
                "long_term_debt",
            ]
        )
    ]

    print(
        "\n=============================="
    )

    print(
        "LATEST FILING FACTS"
    )

    print(
        "=============================="
    )

    print(
        recent[
            [
                "metric",
                "value",
                "fy",
                "fp",
                "start",
                "end",
                "xbrl_tag",
            ]
        ]
        .sort_values(
            [
                "metric",
                "end",
            ]
        )
        .to_string(
            index=False
        )
    )