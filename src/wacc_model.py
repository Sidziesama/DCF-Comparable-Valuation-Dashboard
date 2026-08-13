import pandas as pd


def calculate_cost_of_equity(
    risk_free_rate,
    beta,
    equity_risk_premium,
):
    """
    CAPM:
    Ke = Rf + Beta * ERP
    """

    return (
        risk_free_rate
        + beta * equity_risk_premium
    )


def calculate_cost_of_debt(
    interest_expense,
    average_debt,
):
    """
    Approximate pre-tax cost of debt using:

        Interest Expense / Average Debt

    Later we can replace this with Visa's
    market yield / credit spread.
    """

    if average_debt <= 0:
        raise ValueError(
            "Average debt must be positive."
        )

    return (
        interest_expense
        / average_debt
    )


def calculate_wacc(
    market_cap,
    debt,
    risk_free_rate,
    beta,
    equity_risk_premium,
    pre_tax_cost_of_debt,
    tax_rate,
):

    cost_of_equity = (
        calculate_cost_of_equity(
            risk_free_rate,
            beta,
            equity_risk_premium,
        )
    )

    after_tax_cost_of_debt = (
        pre_tax_cost_of_debt
        * (1 - tax_rate)
    )

    total_capital = (
        market_cap + debt
    )

    equity_weight = (
        market_cap / total_capital
    )

    debt_weight = (
        debt / total_capital
    )

    wacc = (
        equity_weight
        * cost_of_equity

        +

        debt_weight
        * after_tax_cost_of_debt
    )

    return {
        "risk_free_rate":
            risk_free_rate,

        "beta":
            beta,

        "equity_risk_premium":
            equity_risk_premium,

        "cost_of_equity":
            cost_of_equity,

        "pre_tax_cost_of_debt":
            pre_tax_cost_of_debt,

        "after_tax_cost_of_debt":
            after_tax_cost_of_debt,

        "market_cap":
            market_cap,

        "debt":
            debt,

        "equity_weight":
            equity_weight,

        "debt_weight":
            debt_weight,

        "tax_rate":
            tax_rate,

        "wacc":
            wacc,
    }

def build_wacc_report(result):

    rows = [
        ["Risk-Free Rate",
         result["risk_free_rate"]],

        ["Beta",
         result["beta"]],

        ["Equity Risk Premium",
         result["equity_risk_premium"]],

        ["Cost of Equity",
         result["cost_of_equity"]],

        ["Pre-Tax Cost of Debt",
         result["pre_tax_cost_of_debt"]],

        ["Tax Rate",
         result["tax_rate"]],

        ["After-Tax Cost of Debt",
         result["after_tax_cost_of_debt"]],

        ["Equity Weight",
         result["equity_weight"]],

        ["Debt Weight",
         result["debt_weight"]],

        ["WACC",
         result["wacc"]],
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Metric",
            "Value",
        ],
    )