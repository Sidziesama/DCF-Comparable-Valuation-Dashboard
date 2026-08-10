from pathlib import Path
import yaml
from dotenv import load_dotenv

from sec_data import build_historical
from market_data import get_market_data
from model import forecast_financials, calculate_wacc, dcf_valuation, sensitivity_table

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    with open(ROOT / "config/company.yaml", "r") as f:
        return yaml.safe_load(f)

def run():
    load_dotenv(ROOT / ".env")
    cfg = load_config()

    hist = build_historical(
        cfg["cik"],
        years=cfg["historical_years"]
    )
    market = get_market_data(cfg["ticker"])
    a = cfg["assumptions"]

    wacc = calculate_wacc(market, a)
    forecast = forecast_financials(hist, a)

    shares = market["shares_outstanding"] / 1_000_000
    cash = (market.get("cash") or 0) / 1_000_000
    debt = (market.get("total_debt") or 0) / 1_000_000

    valuation = dcf_valuation(
        forecast,
        wacc["wacc"],
        a["terminal_growth"],
        cash,
        debt,
        shares
    )

    sens = sensitivity_table(
        forecast,
        [0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10],
        [0.015, 0.02, 0.025, 0.03, 0.035],
        cash,
        debt,
        shares
    )

    processed = ROOT / "data/processed"
    processed.mkdir(exist_ok=True)

    hist.to_csv(processed / "historical.csv")
    forecast.to_csv(processed / "forecast.csv")
    sens.to_csv(processed / "dcf_sensitivity.csv")

    print("\n=== VISA VALUATION ===")
    print(f"Price:              ${market['price']:.2f}")
    print(f"Market cap:         ${market['market_cap']/1e9:.1f}B")
    print(f"WACC:               {wacc['wacc']:.2%}")
    print(f"Terminal growth:    {a['terminal_growth']:.2%}")
    print(f"DCF intrinsic price:${valuation['intrinsic_price']:.2f}")
    print(f"Enterprise value:   ${valuation['enterprise_value']:.1f}M")
    print("\nSensitivity:")
    print(sens.round(2))

if __name__ == "__main__":
    run()
