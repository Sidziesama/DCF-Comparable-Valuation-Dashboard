import pandas as pd


def validate_financial_data(
    df: pd.DataFrame
) -> dict:

    results = {}

    results["row_count"] = len(df)

    results["missing_values"] = (
        df["value"].isna().sum()
    )

    results["negative_revenue"] = (
        (
            (df["metric"] == "revenue")
            &
            (df["value"] < 0)
        ).sum()
    )

    results["duplicate_observations"] = (
        df.duplicated(
            subset=[
                "metric",
                "fy",
                "fp",
                "form",
                "end"
            ]
        ).sum()
    )

    return results


def print_validation_report(
    results: dict
):

    print("\n==============================")
    print("FINANCIAL DATA VALIDATION")
    print("==============================")

    for key, value in results.items():

        print(
            f"{key:30}: {value}"
        )


def validate_data_freshness(
    quarterly_df,
    latest_10q,
):
    """
    Compare our extracted quarterly financial data
    against the latest SEC 10-Q report date.
    """

    if quarterly_df.empty:
        return {
            "is_current": False,
            "latest_sec_period": None,
            "latest_data_period": None,
        }

    latest_sec_period = pd.to_datetime(
        latest_10q["reportDate"]
    )

    latest_data_period = pd.to_datetime(
        quarterly_df["end"],
        errors="coerce",
    ).max()

    is_current = (
        latest_data_period
        >= latest_sec_period
    )

    return {
        "is_current": is_current,
        "latest_sec_period": latest_sec_period,
        "latest_data_period": latest_data_period,
    }


def print_freshness_report(
    result,
):

    print("\n==============================")
    print("DATA FRESHNESS")
    print("==============================")

    print(
        "Latest SEC period     :",
        result["latest_sec_period"],
    )

    print(
        "Latest dataset period :",
        result["latest_data_period"],
    )

    print(
        "Dataset current       :",
        result["is_current"],
    )