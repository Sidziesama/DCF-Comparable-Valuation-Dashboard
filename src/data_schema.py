from dataclasses import dataclass
from typing import Optional


@dataclass
class FinancialMetric:
    metric: str
    value: float
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[str] = None
    period_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    filing_date: Optional[str] = None
    form: Optional[str] = None
    source: str = "SEC"


INCOME_STATEMENT_METRICS = [
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "operating_expenses",
    "operating_income",
    "interest_expense",
    "pretax_income",
    "tax_expense",
    "net_income",
    "diluted_eps",
]


BALANCE_SHEET_METRICS = [
    "cash",
    "short_term_investments",
    "accounts_receivable",
    "current_assets",
    "ppe",
    "total_assets",
    "accounts_payable",
    "current_liabilities",
    "short_term_debt",
    "long_term_debt",
    "total_liabilities",
    "equity",
]


CASH_FLOW_METRICS = [
    "net_income",
    "depreciation",
    "stock_based_compensation",
    "change_in_working_capital",
    "cfo",
    "capex",
    "cfi",
    "cff",
    "dividends",
    "buybacks",
]