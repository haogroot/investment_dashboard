import os
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import sys
import jinja2

# Import project modules
import build_portfolio
import portfolio_analytics

# Configuration
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "trade_output" / "html"
STATIC_DIR = OUTPUT_DIR / "static"

def setup_directories():
    """Ensures output directories exist."""
    if os.path.exists(OUTPUT_DIR):
        # Optional: Clear old build?
        # shutil.rmtree(OUTPUT_DIR)
        pass
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "static").mkdir(exist_ok=True)
    
    # Create dummy static CSS/JS if not exist (will handle later)

def generate_html_report(input_file=None):
    print("Loading data...")
    # 1. Load Data
    df_trades, market_data, df_ref, all_dates = build_portfolio.load_process_data(input_file)
    
    if df_trades is None:
        print("No trades found. Aborting.")
        return

    # 2. Calculate State (Units, Cash)
    portfolio_tickers = df_ref['Ticker'].tolist()
    # Ensure cost_basis is returned by calculate_portfolio_state (updated build_portfolio previously)
    df_units, df_cash, cost_basis = build_portfolio.calculate_portfolio_state(df_trades, all_dates, portfolio_tickers)
    
    # 3. Calculate Analytics
    print("Calculating analytics...")
    analytics = portfolio_analytics.calculate_analytics(
        df_units, 
        df_cash, 
        market_data, 
        df_ref, 
        cost_basis,
        starting_capital=build_portfolio.STARTING_CAPITAL
    )
    
    # 4. Setup Jinja2 Environment
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        autoescape=jinja2.select_autoescape(['html', 'xml'])
    )
    
    # Add helper functions (e.g. formatting)
    def format_currency(value):
        return "${:,.2f}".format(value)
    
    def format_pct(value):
        return "{:.2%}".format(value)
    
    env.filters['currency'] = format_currency
    env.filters['pct'] = format_pct
    
    # Export JSON
    import json
    json_path = OUTPUT_DIR / "risk_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        # Convert non-serializable objects if any (e.g. numpy types)
        # Using default=str for safety
        json.dump(analytics, f, indent=4, default=str)
    print(f"Generated {json_path.name}")

    # 5. Render Templates
    pages = [
        ('index.html', 'dashboard.html'),
        ('positions.html', 'positions.html'),
        ('risk.html', 'risk.html'),
        ('options.html', 'options.html'),
        ('sector_exposure.html', 'sector_exposure.html')
    ]
    
    print("Rendering templates...")
    # Pass all analytics data to template
    # Also pass 'now' for timestamp
    context = {
        'data': analytics,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    for output_name, template_name in pages:
        try:
            template = env.get_template(template_name)
            html_out = template.render(**context)
            with open(OUTPUT_DIR / output_name, 'w', encoding='utf-8') as f:
                f.write(html_out)
            print(f"Generated {output_name}")
        except jinja2.TemplateNotFound:
            print(f"Template {template_name} not found. Skipping.")
        except Exception as e:
            print(f"Error rendering {template_name}: {e}")

if __name__ == "__main__":
    setup_directories()
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    generate_html_report(input_file)
