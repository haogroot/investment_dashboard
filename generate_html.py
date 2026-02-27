import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import sys
import jinja2

# Import project modules
import build_portfolio
import portfolio_analytics
import tw_stock_loader

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

def generate_html_report(input_file=None, tw_inventory_file=None):
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
        starting_capital=build_portfolio.STARTING_CAPITAL,
        df_trades=df_trades
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

    def format_commas(value):
        try:
            return "{:,.0f}".format(value)
        except:
            return value
    
    env.filters['currency'] = format_currency
    env.filters['pct'] = format_pct
    env.filters['commas'] = format_commas
    
    # Export JSON
    import json
    
    # Custom encoder for numpy types
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    # First serialize to json string to handle numpy types
    analytics_json_str = json.dumps(analytics, cls=NpEncoder)
    # Then parse back to native python types
    clean_analytics = json.loads(analytics_json_str)

    json_path = OUTPUT_DIR / "risk_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(clean_analytics, f, indent=4)
    print(f"Generated {json_path.name}")

    # 5. Load Taiwan Stock Data (if provided)
    tw_data = None
    if tw_inventory_file:
        tw_data = tw_stock_loader.load_tw_inventory(tw_inventory_file)
        if tw_data:
            clean_analytics['tw_positions'] = tw_data['tw_positions']
            clean_analytics['tw_summary'] = tw_data['tw_summary']
            print(f"Loaded {len(tw_data['tw_positions'])} Taiwan stock positions")

    # Combine US and TW Data
    all_positions = []
    total_market_value_twd = 0
    total_pnl_twd = 0
    total_cost_twd = 0

    # Add US positions to all_positions
    us_fx_rate = clean_analytics['dashboard'].get('fx_rate', 1.0)
    for pos in clean_analytics['positions']:
        pos_twd = pos.copy()
        pos_twd['currency'] = 'USD'
        market_value_twd = pos.get('market_value_twd', pos.get('market_value', 0) * us_fx_rate)
        pos_twd['market_value_unified'] = market_value_twd
        
        # Approximate US Cost in TWD (if cost_basis available, use it or estimate)
        us_cost_twd = (pos.get('market_value', 0) - pos.get('unrealized_pnl', 0)) * us_fx_rate
        pos_twd['total_cost_unified'] = us_cost_twd
        
        unrealized_pnl_twd = pos.get('unrealized_pnl', 0) * us_fx_rate
        pos_twd['unrealized_pnl_unified'] = unrealized_pnl_twd

        total_market_value_twd += market_value_twd
        total_pnl_twd += unrealized_pnl_twd
        total_cost_twd += us_cost_twd
        all_positions.append(pos_twd)

    # Add TW positions to all_positions
    if tw_data:
        for pos in tw_data['tw_positions']:
            pos_twd = pos.copy()
            pos_twd['currency'] = 'TWD'
            pos_twd['ticker'] = pos.get('ticker_local', pos.get('ticker', ''))
            pos_twd['name'] = pos.get('name_local', pos.get('name', ''))
            
            market_value_twd = pos.get('market_value', 0)
            pos_twd['market_value_unified'] = market_value_twd
            
            tw_cost_twd = pos.get('market_value', 0) - pos.get('unrealized_pnl', 0)
            pos_twd['total_cost_unified'] = tw_cost_twd
            
            unrealized_pnl_twd = pos.get('unrealized_pnl', 0)
            pos_twd['unrealized_pnl_unified'] = unrealized_pnl_twd

            total_market_value_twd += market_value_twd
            total_pnl_twd += unrealized_pnl_twd
            total_cost_twd += tw_cost_twd
            all_positions.append(pos_twd)

    # Recalculate weights for combined portfolio
    for pos in all_positions:
        pos['weight_unified'] = pos['market_value_unified'] / total_market_value_twd if total_market_value_twd > 0 else 0

    all_positions.sort(key=lambda x: x['market_value_unified'], reverse=True)

    clean_analytics['all_positions'] = all_positions

    total_return_pct_unified = total_pnl_twd / total_cost_twd if total_cost_twd > 0 else 0
    clean_analytics['all_summary'] = {
        'total_value_twd': total_market_value_twd,
        'total_pnl_twd': total_pnl_twd,
        'total_cost_twd': total_cost_twd,
        'total_return_pct': total_return_pct_unified,
        'position_count': len(all_positions)
    }

    # 7. Goal Tracking
    goal_config_path = BASE_DIR / 'goal_config.json'
    if goal_config_path.exists():
        with open(goal_config_path, 'r', encoding='utf-8') as f:
            goal_config = json.load(f)
        goal_tracking = portfolio_analytics.calculate_goal_tracking(
            total_asset_twd=total_market_value_twd,
            goal_config=goal_config
        )
        clean_analytics['goal_tracking'] = goal_tracking
        print(f"Calculated goal tracking for {len(goal_tracking['goals'])} goals")
    else:
        print("goal_config.json not found, skipping goal tracking")

    # 6. Render Templates
    pages = [
        ('index.html', 'dashboard.html'),
        ('positions.html', 'positions.html'),
        ('correlation_matrix.html', 'correlation_matrix.html'),
        ('risk.html', 'risk.html'),
        ('stress_testing.html', 'stress_testing.html'),
        ('options.html', 'options.html'),
        ('sector_exposure.html', 'sector_exposure.html'),
        ('history.html', 'history.html'),
        ('goals.html', 'goals.html')
    ]
    
    # Conditionally add TW positions page
    if tw_data:
        pages.append(('tw_positions.html', 'tw_positions.html'))
    
    # Always generate all_positions.html
    pages.append(('all_positions.html', 'all_positions.html'))
    
    print("Rendering templates...")
    # Pass all analytics data to template
    # Also pass 'now' for timestamp
    context = {
        'data': clean_analytics,
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
    import argparse
    parser = argparse.ArgumentParser(description='Build HTML Dashboard')
    parser.add_argument('input_file', nargs='?', default=None, help='Input trade CSV filename (default: configured INPUT_FILE)')
    parser.add_argument('--tw-inventory', dest='tw_inventory', default=None,
                        help='Taiwan stock inventory CSV file path (e.g. trade_source/ctbc-trade-record_20260225.csv)')
    args = parser.parse_args()

    # Resolve Input File Path if provided
    input_path = None
    if args.input_file:
        input_path = Path(args.input_file)

    tw_inventory_path = None
    if args.tw_inventory:
        tw_inventory_path = Path(args.tw_inventory)

    setup_directories()
    generate_html_report(input_path, tw_inventory_file=tw_inventory_path)
