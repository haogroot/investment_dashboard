# Investment Portfolio Dashboard

This Python project generates a comprehensive investment portfolio dashboard from your trade logs. It automatically fetches market data using `yfinance` and provides both an Excel summary and a rich interactive HTML dashboard for current valuations, P&L, exposure analysis, and historical tracking.

## Features

- **Automated Data Processing**: Parses trade logs (CSV) from brokers (e.g., Fubon Securities).
- **Multi-Market Support (New)**: Combines US stock trades and Taiwan stock inventory.
- **Market Data Integration**: Fetches daily closing prices, company info, and Beta via Yahoo Finance.
- **Interactive HTML Dashboard 📊**:
    - **Dashboard**: Key Performance Indicators (KPIs) and Asset Allocation Widget.
    - **Unified Holdings**: Consolidated view of all global positions.
    - **Positions**: Detailed breakdowns of US and Taiwan holdings.
    - **Historical Transactions**: Track your trade history over time.
    - **Risk & Stress Testing**: Beta analysis, correlation matrices, and customizable stress-testing.
    - **Sector & Geographic Exposure**: Analyzes portfolio diversification.
- **Excel Dashboard**: Provides a supplementary Excel output with equity curves and position breakdowns.
- **Privacy Focused**: Processes data locally. Your trade logs remain on your machine.

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

### 1. Generating the HTML Dashboard (Recommended)

Generate a complete set of interactive HTML reports viewable in any modern browser.

**Basic Usage:**
Defaults to `trade_source/fubon-trade-record_20260224.csv`.
```bash
python generate_html.py
```

**Custom Input File:**
```bash
python generate_html.py trade_source/your_us_history.csv
```

**Include Taiwan Stock Inventory (New):**
Loads your Taiwan stock holdings along with US positions to generate a unified portfolio view.
```bash
python generate_html.py trade_source/fubon-trade-record.csv --tw-inventory trade_source/ctbc-inventory.csv
```

**View Results:**
Open `trade_output/html/index.html` in your browser.

### 2. Generating the Excel Dashboard

Generates an Excel (`.xlsx`) dashboard with equity curves and portfolio summary.

**Default:**
```bash
python build_portfolio.py
```

**Custom Input File:**
```bash
python build_portfolio.py trade_source/your_trade_file.csv
```

**View Results:**
The generated Excel dashboard will be in the `trade_output` directory.

### 3. Daily Updates

You can update your portfolio prices incrementally without rebuilding everything from scratch.

**Full Update (Trades + Prices):**
Syncs new trades from your CSV and fetches the latest market prices.
```bash
python update_portfolio.py
```

**Quick Update (Prices Only):**
Fetches only the latest market prices (skips trade history check).
```bash
python update_portfolio.py quick
```

**Add Trade Manually:**
Appends a single trade to your CSV and runs a full update.
```bash
# Usage: python update_portfolio.py trade DATE TICKER SIDE QTY PRICE
python update_portfolio.py trade 2026-02-18 NVDA BUY 10 125.50
```

## Project Structure

- `generate_html.py`: Generates interactive HTML reports.
- `build_portfolio.py`: Main logic for building Excel dashboards.
- `update_portfolio.py`: Script for daily incremental updates.
- `portfolio_analytics.py`: Calculates risk, exposure, and performance metrics.
- `tw_stock_loader.py`: Handles loading and processing Taiwan stock inventories.
- `templates/`: Jinja2 HTML templates for the interactive web dashboard.
- `trade_source/`: Directory for input CSV files (Git-ignored for privacy).
- `trade_output/`: Directory for generated Excel and HTML files (Git-ignored).

## License

MIT
