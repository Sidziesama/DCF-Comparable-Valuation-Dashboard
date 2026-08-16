import pandas as pd
import numpy as np


# =========================================================
# FOOTBALL FIELD DATA
# =========================================================

def build_football_field(
    gordon_dcf,
    exit_dcf,
    direct_peer_comps,
    current_price,
    direct_peer_label="Direct Peer",
):
    """
    Convert valuation methodologies into
    low / base / high ranges for visualization.

    Output is dashboard-ready.
    """

    rows = []


    # =====================================================
    # GORDON GROWTH DCF
    # =====================================================

    if not gordon_dcf.empty:

        rows.append(
            {
                "method":
                    "DCF - Gordon Growth",

                "low":
                    float(
                        gordon_dcf.loc[
                            "bear",
                            "implied_share_price",
                        ]
                    ),

                "base":
                    float(
                        gordon_dcf.loc[
                            "base",
                            "implied_share_price",
                        ]
                    ),

                "high":
                    float(
                        gordon_dcf.loc[
                            "bull",
                            "implied_share_price",
                        ]
                    ),

                "current_price":
                    current_price,
            }
        )


    # =====================================================
    # EXIT MULTIPLE DCF
    # =====================================================

    if not exit_dcf.empty:

        rows.append(
            {
                "method":
                    "DCF - Exit Multiple",

                "low":
                    float(
                        exit_dcf.loc[
                            "bear",
                            "implied_share_price",
                        ]
                    ),

                "base":
                    float(
                        exit_dcf.loc[
                            "base",
                            "implied_share_price",
                        ]
                    ),

                "high":
                    float(
                        exit_dcf.loc[
                            "bull",
                            "implied_share_price",
                        ]
                    ),

                "current_price":
                    current_price,
            }
        )


    # =====================================================
    # DIRECT-PEER TRADING COMPS
    # =====================================================

    if not direct_peer_comps.empty:

        prices = (
            direct_peer_comps[
                "implied_share_price"
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

        if not prices.empty:

            rows.append(
                {
                    "method":
                        f"{direct_peer_label} Trading Comps",

                    "low":
                        float(
                            prices.min()
                        ),

                    "base":
                        float(
                            prices.median()
                        ),

                    "high":
                        float(
                            prices.max()
                        ),

                    "current_price":
                        current_price,
                }
            )


    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result


    # =====================================================
    # UPSIDE / DOWNSIDE
    # =====================================================

    result[
        "low_upside_downside"
    ] = (
        result["low"]
        / current_price
        - 1
    )

    result[
        "base_upside_downside"
    ] = (
        result["base"]
        / current_price
        - 1
    )

    result[
        "high_upside_downside"
    ] = (
        result["high"]
        / current_price
        - 1
    )


    return result


# =========================================================
# CENTRAL VALUATION RANGE
# =========================================================

def calculate_central_range(
    football_field,
    included_methods=None,
):
    """
    Calculate the central range from selected primary methods.

    Exit-multiple DCF remains visible in the football field, but is excluded
    by default because it is a terminal-value sensitivity rather than an
    equally weighted estimate of intrinsic value.
    """

    if football_field.empty:

        return {}

    if included_methods is None:
        included_methods = [
            method for method in football_field["method"].dropna().unique()
            if method != "DCF - Exit Multiple"
        ]

    selected = football_field[football_field["method"].isin(included_methods)]
    bases = (
        selected[
            "base"
        ]
        .dropna()
    )

    if bases.empty:

        return {}

    return {

        "minimum_base":
            float(
                bases.min()
            ),

        "median_base":
            float(
                bases.median()
            ),

        "maximum_base":
            float(
                bases.max()
            ),

        "mean_base":
            float(
                bases.mean()
            ),

        "method_count":
            int(len(bases)),
    }
