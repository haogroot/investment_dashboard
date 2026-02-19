# Investment Portfolio Dashboard

This Python project generates a comprehensive investment portfolio dashboard in Excel from your trade logs. It automatically fetches market data using `yfinance` to calculate current valuations, P&L, and exposure analysis.

## Features

- **Automated Data Processing**: Parses trade logs (CSV) from Fubon Securities (or standard format).
- **Market Data Integration**: Fetches daily closing prices and company info (Sector, Industry, Country) via Yahoo Finance.
- **Excel Dashboard**:
    - **Positions**: Detailed breakdown of current holdings with unrealized P&L.
    - **Equity Curve**: Visualizes Net Asset Value (NAV) growth and Drawdowns over time.
    - **Dashboard**: Key Performance Indicators (KPIs) and allocation charts.
    - **Sector & Geographic Exposure**: Analyzes portfolio diversification.
- **Privacy Focused**: Processes data locally. Your trade logs are never sent to any server other than for price fetching (ticker symbols only).

## Prerequisites

- Python 3.8+
- Internet connection (for `yfinance`)

## Installation

1.  Clone this repository or download the source code.
2.  Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare Trade Log**:
    - Export your trade history from your broker as a CSV file.
    - Place the CSV file in the `trade_source` directory.
    - Ensure the file is named `fubon-trade_202602.csv` (or update `INPUT_FILE` in `build_portfolio.py`).

2.  **Run the Script**:

    **Default (uses `fubon-trade_202602.csv`):**
    ```bash
    python build_portfolio.py
    ```

    **Specify a custom file (e.g., inside `trade_source/`):**
    ```bash
    python build_portfolio.py fubon-trade_20260219.csv
    ```

3.  **View Results**:
    - The generated Excel dashboard will be in the `trade_output` directory, named with the format `portfolio_dashboard_YYYYMMDD.xlsx`.

## Daily Updates (New)

You can update your portfolio incrementally without rebuilding.

### 1. Full Update (Trades + Prices)
Syncs new trades from your CSV and fetches the latest market prices.
```bash
python update_portfolio.py
```

### 2. Quick Update (Prices Only)
Fetches only the latest market prices (skips trade check).
```bash
python update_portfolio.py quick
```

## HTML Dashboard (New) 📊

Generate a set of interactive HTML reports (Positions, Risk, Options, etc.) viewable in any browser.

1.  **Install Dependency**:
    ```bash
    pip install jinja2
    ```

2.  **Generate Reports**:
    ```bash
    python generate_html.py
    ```
    *Optional: Specify custom input file: `python generate_html.py my_trades.csv`*

3.  **View**:
    Open `trade_output/html/index.html` in your browser.

### 3. Add Trade Manually
Appends a single trade to your CSV and runs a full update.
```bash
# Usage: python update_portfolio.py trade DATE TICKER SIDE QTY PRICE
python update_portfolio.py trade 2026-02-18 NVDA BUY 10 125.50
```

## Project Structure

- `build_portfolio.py`: Main script logic.
- `update_portfolio.py`: Script for daily incremental updates.
- `requirements.txt`: Python dependencies.
- `trade_source/`: Directory for input CSV files (Git-ignored for privacy).
- `trade_output/`: Directory for generated Excel files (Git-ignored).

## License

MIT
