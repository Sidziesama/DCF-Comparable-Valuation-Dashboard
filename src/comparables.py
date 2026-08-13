from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import yaml


# =========================================================
# PATHS
# =========================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# =========================================================
# MULTIPLE CONFIGURATION
# =========================================================

MULTIPLE_COLUMNS = [
    "ev_revenue",
    "ev_ebitda",
    "ev_ebit",
    "pe",
]


# Multiple eligibility by company.
#
# AXP is treated as P/E-only because its lending/balance-sheet
# structure makes enterprise-value multiples less comparable
# to Visa's asset-light payment-network model.

MULTIPLE_ELIGIBILITY = {

    "MA": {
        "ev_revenue",
        "ev_ebitda",
        "ev_ebit",
        "pe",
    },

    "PYPL": {
        "ev_revenue",
        "ev_ebitda",
        "ev_ebit",
        "pe",
    },

    "FISV": {
        "ev_revenue",
        "ev_ebitda",
        "ev_ebit",
        "pe",
    },

    "GPN": {
        "ev_revenue",
        "ev_ebitda",
        "ev_ebit",
        "pe",
    },

    "AXP": {
        "pe",
    },
}


# =========================================================
# BASIC HELPERS
# =========================================================

def to_millions(value):
    """
    Convert raw USD value to USD millions.
    """

    if value is None:
        return np.nan

    try:
        return float(value) / 1_000_000

    except (TypeError, ValueError):
        return np.nan


def safe_divide(
    numerator,
    denominator,
):
    """
    Safe multiple calculation.
    """

    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator <= 0
    ):
        return np.nan

    return (
        numerator
        / denominator
    )


# =========================================================
# TTM STATEMENT FALLBACK
# =========================================================

def get_ttm_statement_value(
    company,
    row_names,
):
    """
    Attempt to extract a TTM income-statement value
    from yfinance using multiple possible row labels.

    Returns USD millions.
    """

    try:

        statement = (
            company.ttm_income_stmt
        )

        if (
            statement is None
            or statement.empty
        ):
            return np.nan

        for row_name in row_names:

            if (
                row_name
                not in statement.index
            ):
                continue

            values = (
                statement.loc[
                    row_name
                ]
                .dropna()
            )

            if values.empty:
                continue

            return (
                float(
                    values.iloc[0]
                )
                / 1_000_000
            )

    except Exception:
        pass

    return np.nan


# =========================================================
# MARKET / FUNDAMENTAL DATA
# =========================================================

def fetch_comparable_data(
    ticker: str,
) -> dict:
    """
    Fetch market and trailing fundamental data
    for a comparable company.

    Monetary values are returned in USD millions.
    """

    company = yf.Ticker(
        ticker
    )

    info = company.info


    # -----------------------------------------------------
    # Market data
    # -----------------------------------------------------

    price = info.get(
        "currentPrice"
    )

    market_cap_raw = info.get(
        "marketCap"
    )

    enterprise_value_raw = info.get(
        "enterpriseValue"
    )

    shares_raw = info.get(
        "sharesOutstanding"
    )

    total_debt_raw = info.get(
        "totalDebt"
    )

    cash_raw = info.get(
        "totalCash"
    )


    # -----------------------------------------------------
    # Fundamental data from Yahoo info
    # -----------------------------------------------------

    revenue_raw = info.get(
        "totalRevenue"
    )

    ebitda_raw = info.get(
        "ebitda"
    )

    net_income_raw = info.get(
        "netIncomeToCommon"
    )

    trailing_eps = info.get(
        "trailingEps"
    )


    # -----------------------------------------------------
    # Convert market values
    # -----------------------------------------------------

    market_cap = to_millions(
        market_cap_raw
    )

    enterprise_value = to_millions(
        enterprise_value_raw
    )

    shares_outstanding = (
        to_millions(
            shares_raw
        )
    )

    total_debt = to_millions(
        total_debt_raw
    )

    cash = to_millions(
        cash_raw
    )
    if pd.isna(total_debt):
        total_debt = (
        get_balance_sheet_value(
            company,
            [
                "Total Debt",
                "Long Term Debt And Capital Lease Obligation",
            ],
        )
    )


    if pd.isna(cash):

        cash = (
            get_balance_sheet_value(
                company,
                [
                    "Cash Cash Equivalents And Short Term Investments",
                    "Cash And Cash Equivalents",
                ],
            )
        )

    # -----------------------------------------------------
    # Revenue
    # -----------------------------------------------------

    revenue = to_millions(
        revenue_raw
    )

    if pd.isna(revenue):

        revenue = (
            get_ttm_statement_value(
                company,
                [
                    "Total Revenue",
                    "Operating Revenue",
                    "Revenue",
                ],
            )
        )


    # -----------------------------------------------------
    # EBITDA
    # -----------------------------------------------------

    ebitda = to_millions(
        ebitda_raw
    )

    if pd.isna(ebitda):

        ebitda = (
            get_ttm_statement_value(
                company,
                [
                    "EBITDA",
                    "Normalized EBITDA",
                ],
            )
        )


    # -----------------------------------------------------
    # EBIT
    # -----------------------------------------------------

    ebit = (
        get_ttm_statement_value(
            company,
            [
                "EBIT",
                "Operating Income",
            ],
        )
    )


    # -----------------------------------------------------
    # Net income
    # -----------------------------------------------------

    net_income = to_millions(
        net_income_raw
    )

    if pd.isna(net_income):

        net_income = (
            get_ttm_statement_value(
                company,
                [
                    "Net Income",
                    "Net Income Common Stockholders",
                    "Net Income Including Noncontrolling Interests",
                ],
            )
        )


    # -----------------------------------------------------
    # EV fallback
    # -----------------------------------------------------

    if pd.isna(
        enterprise_value
    ):

        if (
            pd.notna(market_cap)
            and pd.notna(total_debt)
            and pd.notna(cash)
        ):

            enterprise_value = (
                market_cap
                + total_debt
                - cash
            )


    return {

        "ticker":
            ticker,

        "company_name":
            info.get(
                "shortName",
                ticker,
            ),

        "price":
            price,

        "market_cap":
            market_cap,

        "enterprise_value":
            enterprise_value,

        "revenue":
            revenue,

        "ebitda":
            ebitda,

        "ebit":
            ebit,

        "net_income":
            net_income,

        "shares_outstanding":
            shares_outstanding,

        "total_debt":
            total_debt,

        "cash":
            cash,

        "eps":
            trailing_eps,
    }


# =========================================================
# MULTIPLES
# =========================================================

def calculate_multiples(
    row,
):
    """
    Calculate standard trading multiples.
    """

    return {

        "ev_revenue":
            safe_divide(
                row.get(
                    "enterprise_value"
                ),
                row.get(
                    "revenue"
                ),
            ),

        "ev_ebitda":
            safe_divide(
                row.get(
                    "enterprise_value"
                ),
                row.get(
                    "ebitda"
                ),
            ),

        "ev_ebit":
            safe_divide(
                row.get(
                    "enterprise_value"
                ),
                row.get(
                    "ebit"
                ),
            ),

        "pe":
            safe_divide(
                row.get(
                    "market_cap"
                ),
                row.get(
                    "net_income"
                ),
            ),
    }


# =========================================================
# COMPARABLE TABLE
# =========================================================

def build_comparable_table(
    tickers,
):
    """
    Build normalized trading-comparable table.
    """

    rows = []

    for ticker in tickers:

        print(
            f"Fetching comparable: "
            f"{ticker}"
        )

        try:

            data = (
                fetch_comparable_data(
                    ticker
                )
            )

            multiples = (
                calculate_multiples(
                    data
                )
            )

            data.update(
                multiples
            )

            rows.append(
                data
            )

        except Exception as error:

            print(
                f"Failed {ticker}: "
                f"{error}"
            )

    if not rows:

        raise RuntimeError(
            "No comparable companies "
            "were successfully loaded."
        )

    comps = (
        pd.DataFrame(
            rows
        )
        .set_index(
            "ticker"
        )
    )

    return comps


# =========================================================
# MULTIPLE ELIGIBILITY
# =========================================================

def apply_multiple_eligibility(
    comps,
):
    """
    Remove multiples that should not be used
    for particular peer companies.
    """

    output = comps.copy()

    for ticker in output.index:

        eligible = (
            MULTIPLE_ELIGIBILITY.get(
                ticker,
                set(),
            )
        )

        for metric in MULTIPLE_COLUMNS:

            if metric not in eligible:

                output.loc[
                    ticker,
                    metric,
                ] = np.nan

    return output


# =========================================================
# PEER STATISTICS
# =========================================================

def calculate_peer_statistics(
    comps,
):
    """
    Calculate peer trading statistics using only
    eligible multiples.
    """

    eligible_comps = (
        apply_multiple_eligibility(
            comps
        )
    )

    statistics = pd.DataFrame(
        index=[
            "Minimum",
            "25th Percentile",
            "Median",
            "75th Percentile",
            "Maximum",
        ],
        columns=MULTIPLE_COLUMNS,
        dtype=float,
    )

    for metric in MULTIPLE_COLUMNS:

        values = (
            eligible_comps[
                metric
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .dropna()
        )

        if values.empty:
            continue

        statistics.loc[
            "Minimum",
            metric,
        ] = values.min()

        statistics.loc[
            "25th Percentile",
            metric,
        ] = values.quantile(
            0.25
        )

        statistics.loc[
            "Median",
            metric,
        ] = values.median()

        statistics.loc[
            "75th Percentile",
            metric,
        ] = values.quantile(
            0.75
        )

        statistics.loc[
            "Maximum",
            metric,
        ] = values.max()

    return statistics


# =========================================================
# CORE DIRECT COMP
# =========================================================

def build_core_comp_view(
    comps,
):
    """
    Mastercard is Visa's closest public network comp.
    """

    if "MA" not in comps.index:

        return pd.DataFrame()

    return comps.loc[
        ["MA"]
    ].copy()


# =========================================================
# IMPLIED VALUATION
# =========================================================

def implied_valuation_from_multiple(
    multiple,
    metric_value,
    multiple_type,
    cash,
    debt,
    shares_outstanding,
):
    """
    Apply a trading multiple to Visa.

    EV multiples:
        multiple × operating metric
        -> enterprise value
        -> equity value

    P/E:
        multiple × net income
        -> equity value
    """

    if (
        pd.isna(multiple)
        or pd.isna(metric_value)
    ):

        return {
            "enterprise_value":
                np.nan,

            "equity_value":
                np.nan,

            "share_price":
                np.nan,
        }


    # -----------------------------------------------------
    # P/E
    # -----------------------------------------------------

    if multiple_type == "pe":

        equity_value = (
            multiple
            * metric_value
        )

        share_price = (
            equity_value
            / shares_outstanding
        )

        return {

            "enterprise_value":
                np.nan,

            "equity_value":
                equity_value,

            "share_price":
                share_price,
        }


    # -----------------------------------------------------
    # EV multiples
    # -----------------------------------------------------

    enterprise_value = (
        multiple
        * metric_value
    )

    equity_value = (
        enterprise_value
        + cash
        - debt
    )

    share_price = (
        equity_value
        / shares_outstanding
    )

    return {

        "enterprise_value":
            enterprise_value,

        "equity_value":
            equity_value,

        "share_price":
            share_price,
    }


def build_implied_valuation(
    peer_statistics,
    visa_metrics,
    cash,
    debt,
    shares_outstanding,
):
    """
    Apply peer 25th / median / 75th percentile
    multiples to Visa's LTM financials.
    """

    mapping = {

        "ev_revenue":
            "revenue",

        "ev_ebitda":
            "ebitda",

        "ev_ebit":
            "ebit",

        "pe":
            "net_income",
    }

    rows = []

    for multiple_type, metric in (
        mapping.items()
    ):

        target_metric = (
            visa_metrics.get(
                metric,
                np.nan,
            )
        )

        for statistic in [
            "25th Percentile",
            "Median",
            "75th Percentile",
        ]:

            multiple = (
                peer_statistics.loc[
                    statistic,
                    multiple_type,
                ]
            )

            result = (
                implied_valuation_from_multiple(
                    multiple=multiple,
                    metric_value=target_metric,
                    multiple_type=multiple_type,
                    cash=cash,
                    debt=debt,
                    shares_outstanding=(
                        shares_outstanding
                    ),
                )
            )

            rows.append(
                {

                    "multiple_type":
                        multiple_type,

                    "statistic":
                        statistic,

                    "multiple":
                        multiple,

                    "visa_metric":
                        target_metric,

                    "enterprise_value":
                        result[
                            "enterprise_value"
                        ],

                    "equity_value":
                        result[
                            "equity_value"
                        ],

                    "implied_share_price":
                        result[
                            "share_price"
                        ],
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# DIRECT MASTERCARD IMPLIED VALUATION
# =========================================================

def build_mastercard_implied_valuation(
    comps,
    visa_metrics,
    cash,
    debt,
    shares_outstanding,
):
    """
    Apply Mastercard's actual trading multiples directly
    to Visa as the closest economic comparable.
    """

    if "MA" not in comps.index:

        return pd.DataFrame()

    ma = comps.loc[
        "MA"
    ]

    mapping = {

        "ev_revenue":
            "revenue",

        "ev_ebitda":
            "ebitda",

        "ev_ebit":
            "ebit",

        "pe":
            "net_income",
    }

    rows = []

    for multiple_type, metric in (
        mapping.items()
    ):

        multiple = ma.get(
            multiple_type,
            np.nan,
        )

        target_metric = (
            visa_metrics.get(
                metric,
                np.nan,
            )
        )

        result = (
            implied_valuation_from_multiple(
                multiple=multiple,
                metric_value=target_metric,
                multiple_type=multiple_type,
                cash=cash,
                debt=debt,
                shares_outstanding=(
                    shares_outstanding
                ),
            )
        )

        rows.append(
            {

                "multiple_type":
                    multiple_type,

                "multiple":
                    multiple,

                "visa_metric":
                    target_metric,

                "enterprise_value":
                    result[
                        "enterprise_value"
                    ],

                "equity_value":
                    result[
                        "equity_value"
                    ],

                "implied_share_price":
                    result[
                        "share_price"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )

def get_balance_sheet_value(
    company,
    row_names,
):
    """
    Retrieve latest balance-sheet value
    from yfinance.

    Returns USD millions.
    """

    try:

        balance = (
            company.quarterly_balance_sheet
        )

        if (
            balance is None
            or balance.empty
        ):
            return np.nan

        for row_name in row_names:

            if row_name not in balance.index:
                continue

            values = (
                balance.loc[
                    row_name
                ]
                .dropna()
            )

            if values.empty:
                continue

            return (
                float(values.iloc[0])
                / 1_000_000
            )

    except Exception:
        pass

    return np.nan   

# =========================================================
# LOAD CONFIG
# =========================================================

def load_config():

    with open(
        ROOT
        / "config"
        / "company.yaml",
        "r",
    ) as f:

        return yaml.safe_load(f)

def calculate_group_statistics(
    comps,
    tickers,
):
    """
    Statistics for a selected peer subset.
    """

    subset = comps.loc[
        comps.index.intersection(
            tickers
        )
    ]

    return calculate_peer_statistics(
        subset
    )

# =========================================================
# RUNNER
# =========================================================

if __name__ == "__main__":

    config = load_config()

    peer_config = (
        config["peers"]
    )


    # -----------------------------------------------------
    # Peer universe
    # -----------------------------------------------------

    core_network = (
        peer_config.get(
            "core_network",
            [],
        )
    )

    payments_tech = (
        peer_config.get(
            "payments_tech",
            [],
        )
    )

    financial = (
        peer_config.get(
            "financial",
            [],
        )
    )

    peers = (
        core_network
        + payments_tech
        + financial
    )


    # -----------------------------------------------------
    # Build comps
    # -----------------------------------------------------

    comps = (
        build_comparable_table(
            peers
        )
    )

    eligible_comps = (
        apply_multiple_eligibility(
            comps
        )
    )

    statistics = (
        calculate_peer_statistics(
            comps
        )
    )

    core_comp = (
        build_core_comp_view(
            comps
        )
    )
    network_statistics = (
    calculate_group_statistics(
        comps,
        ["MA"],
    )
)

    payments_statistics = (
        calculate_group_statistics(
            comps,
            [
                "PYPL",
                "FISV",
                "GPN",
            ],
        )
    )

    # -----------------------------------------------------
    # Load Visa historical model
    # -----------------------------------------------------

    historical = pd.read_csv(
        ROOT
        / "data"
        / "processed"
        / "historical_model.csv",
        index_col=0,
    )

    if "LTM" not in historical.index:

        raise RuntimeError(
            "Historical model does not contain LTM."
        )

    ltm = historical.loc[
        "LTM"
    ]


    # -----------------------------------------------------
    # Load Visa latest balance sheet
    # -----------------------------------------------------

    balance = pd.read_csv(
        ROOT
        / "data"
        / "processed"
        / "latest_balance_sheet.csv",
    )

    balance_map = (
        balance
        .set_index("metric")[
            "value"
        ]
        .to_dict()
    )

    cash = float(
        balance_map.get(
            "cash",
            0.0,
        )
    )

    debt = (
        float(
            balance_map.get(
                "short_term_debt",
                0.0,
            )
        )
        +
        float(
            balance_map.get(
                "long_term_debt",
                0.0,
            )
        )
    )


    # -----------------------------------------------------
    # Visa market data
    # -----------------------------------------------------

    visa = yf.Ticker(
        config["ticker"]
    )

    visa_info = visa.info

    shares_raw = (
        visa_info.get(
            "sharesOutstanding"
        )
    )

    current_price = (
        visa_info.get(
            "currentPrice"
        )
    )

    if shares_raw is None:

        raise RuntimeError(
            "Visa shares outstanding unavailable."
        )

    shares_outstanding = (
        shares_raw
        / 1_000_000
    )


    # -----------------------------------------------------
    # Visa LTM metrics
    # -----------------------------------------------------

    visa_metrics = {

        "revenue":
            float(
                ltm["revenue"]
            ),

        "ebit":
            float(
                ltm[
                    "operating_income"
                ]
            ),

        "net_income":
            float(
                ltm["net_income"]
            ),

        # We intentionally leave EBITDA blank until
        # LTM D&A is added to historical_model.py.

        "ebitda":
        float(
            ltm["ebitda"]
        ),
    }


    # -----------------------------------------------------
    # Peer implied valuation
    # -----------------------------------------------------

    implied = (
        build_implied_valuation(
            peer_statistics=(
                statistics
            ),
            visa_metrics=(
                visa_metrics
            ),
            cash=cash,
            debt=debt,
            shares_outstanding=(
                shares_outstanding
            ),
        )
    )


    # -----------------------------------------------------
    # Direct Mastercard valuation
    # -----------------------------------------------------

    mastercard_implied = (
        build_mastercard_implied_valuation(
            comps=comps,
            visa_metrics=(
                visa_metrics
            ),
            cash=cash,
            debt=debt,
            shares_outstanding=(
                shares_outstanding
            ),
        )
    )


    # -----------------------------------------------------
    # Current price comparisons
    # -----------------------------------------------------

    implied[
        "current_price"
    ] = current_price

    implied[
        "upside_downside"
    ] = (
        implied[
            "implied_share_price"
        ]
        / current_price
        - 1
    )


    mastercard_implied[
        "current_price"
    ] = current_price

    mastercard_implied[
        "upside_downside"
    ] = (
        mastercard_implied[
            "implied_share_price"
        ]
        / current_price
        - 1
    )


    # =====================================================
    # SAVE OUTPUTS
    # =====================================================

    output_dir = (
        ROOT
        / "data"
        / "processed"
    )

    comps.to_csv(
        output_dir
        / "trading_comparables.csv"
    )

    eligible_comps.to_csv(
        output_dir
        / "trading_comparables_eligible.csv"
    )

    statistics.to_csv(
        output_dir
        / "comparable_statistics.csv"
    )

    implied.to_csv(
        output_dir
        / "comps_implied_valuation.csv",
        index=False,
    )

    mastercard_implied.to_csv(
        output_dir
        / "mastercard_implied_valuation.csv",
        index=False,
    )
    network_statistics.to_csv(
    output_dir
    / "network_comp_statistics.csv"
)

    payments_statistics.to_csv(
        output_dir
        / "payments_tech_statistics.csv"
    )


    # =====================================================
    # PRINT OUTPUT
    # =====================================================

    print(
        "\n=============================="
    )

    print(
        "TRADING COMPARABLES"
    )

    print(
        "=============================="
    )

    display_columns = [
        "company_name",
        "market_cap",
        "enterprise_value",
        "revenue",
        "ebitda",
        "ebit",
        "net_income",
        "ev_revenue",
        "ev_ebitda",
        "ev_ebit",
        "pe",
    ]

    print(
        eligible_comps[
            display_columns
        ]
        .round(2)
        .to_string()
    )


    print(
        "\n=============================="
    )

    print(
        "PEER STATISTICS"
    )

    print(
        "=============================="
    )

    print(
        statistics
        .round(2)
        .to_string()
    )


    print(
        "\n=============================="
    )

    print(
        "VISA IMPLIED PEER VALUATION"
    )

    print(
        "=============================="
    )

    print(
        implied[
            [
                "multiple_type",
                "statistic",
                "multiple",
                "visa_metric",
                "implied_share_price",
                "upside_downside",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )


    print(
        "\n=============================="
    )

    print(
        "MASTERCARD DIRECT COMP"
    )

    print(
        "=============================="
    )

    print(
        mastercard_implied[
            [
                "multiple_type",
                "multiple",
                "visa_metric",
                "implied_share_price",
                "upside_downside",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )