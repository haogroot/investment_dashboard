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

    ```bash
    python build_portfolio.py
    ```

3.  **View Results**:
    - The generated Excel dashboard will be in the `trade_output` directory, named with the format `portfolio_dashboard_YYYYMMDD.xlsx`.

## Project Structure

- `build_portfolio.py`: Main script logic.
- `requirements.txt`: Python dependencies.
- `trade_source/`: Directory for input CSV files (Git-ignored for privacy).
- `trade_output/`: Directory for generated Excel files (Git-ignored).

## License

MIT
