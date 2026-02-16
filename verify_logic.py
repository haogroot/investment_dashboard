import pandas as pd
from datetime import datetime, timedelta

# --- Mock Data ---
START_DATE = "2024-06-01"
STARTING_CAPITAL = 100000.0

# Mock Trades
# Note: 2024-06-01 is Saturday. 2024-06-03 is Monday.
trades_data = [
    {'Date': '2024-06-03', 'Ticker': 'AAPL', 'Side': 'Buy', 'Price': 100.0, 'Qty': 10, 'Fee': 0, 'Tax': 0}, # Normal Business Day
    {'Date': '2024-06-04', 'Ticker': 'GOOGL', 'Side': 'Buy', 'Price': 200.0, 'Qty': 5, 'Fee': 0, 'Tax': 0}, # Normal Business Day
    {'Date': '2024-06-08', 'Ticker': 'NVDA', 'Side': 'Buy', 'Price': 1000.0, 'Qty': 1, 'Fee': 0, 'Tax': 0}, # Saturday Trade
]

df_trades = pd.DataFrame(trades_data)
df_trades['Date'] = pd.to_datetime(df_trades['Date']).dt.normalize()

# Mock Date Range (Business Days)
# 6/3 (Mon) to 6/10 (Mon)
all_dates = pd.date_range(start='2024-06-03', end='2024-06-10', freq='B')

print("--- Data Setup ---")
print("Trades:")
print(df_trades)
print("\nDate Grid:")
print(all_dates)

# --- Logic from build_portfolio.py (Pasted & Adapted) ---

def calculate_portfolio_state_debug(df_trades, all_dates, tickers):
    df_units = pd.DataFrame(0, index=all_dates, columns=tickers)
    df_cash = pd.DataFrame(0.0, index=all_dates, columns=['Cash_Balance'])
    
    cost_basis_tracker = {t: {'total_shares': 0, 'total_cost': 0.0, 'avg_cost': 0.0} for t in tickers}
    
    current_cash = STARTING_CAPITAL
    current_units = {t: 0 for t in tickers}
    
    df_trades = df_trades.sort_values('Date')
    n_trades = len(df_trades)
    trade_idx = 0
    
    print("\n--- Starting Calculation Loop ---")
    
    for date in all_dates:
        print(f"\nProcessing Date Grid: {date.date()}")
        
        # DEBUG: Print status of trade processing
        if trade_idx < n_trades:
            next_trade_date = df_trades.iloc[trade_idx]['Date']
            print(f"  Next Trade Date: {next_trade_date.date()} (Idx: {trade_idx})")
            print(f"  Condition (Next Trade <= Current Grid): {next_trade_date <= date}")
        else:
            print("  No more trades pending.")

        while trade_idx < n_trades and df_trades.iloc[trade_idx]['Date'] <= date:
            trade = df_trades.iloc[trade_idx]
            print(f"    -> SUPPORTED TRADE FOUND: {trade['Date'].date()} {trade['Ticker']} {trade['Side']} {trade['Qty']}")
            
            ticker = trade['Ticker']
            side = str(trade['Side']).lower()
            qty = float(trade['Qty'])
            price = float(trade['Price'])
            fee = float(trade.get('Fee', 0))
            tax = float(trade.get('Tax', 0))
            total_val = price * qty
            
            if 'buy' in side:
                current_cash -= (total_val + fee)
                current_units[ticker] += qty
                print(f"       Action: Bought {qty} {ticker}. New Unit Count: {current_units[ticker]}")
                
            elif 'sell' in side:
                current_cash += (total_val - fee - tax)
                current_units[ticker] -= qty
                print(f"       Action: Sold {qty} {ticker}. New Unit Count: {current_units[ticker]}")

            trade_idx += 1
        
        # Snapshot for the day
        for t in tickers:
            df_units.at[date, t] = current_units[t]
        
        df_cash.at[date, 'Cash_Balance'] = current_cash
        print(f"  End of Day State: Cash={current_cash}, Units={current_units}")
        
    return df_units, df_cash

# --- Run Debug ---
tickers = ['AAPL', 'GOOGL', 'NVDA']
df_units, df_cash = calculate_portfolio_state_debug(df_trades, all_dates, tickers)

print("\n--- Final Results ---")
print("Daily Units Head:")
print(df_units.head())
print("\nDaily Units Tail:")
print(df_units.tail())
