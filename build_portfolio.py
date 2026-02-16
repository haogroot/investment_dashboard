import pandas as pd
import yfinance as yf
import xlsxwriter
from datetime import datetime, timedelta
import os

from pathlib import Path

# --- Configuration ---
# Use relative paths based on the script directory to handle Windows/WSL mapping automatically
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "trade_source" / "fubon-trade_202602.csv"
OUTPUT_DIR = BASE_DIR / "trade_output"
OUTPUT_FILE = OUTPUT_DIR / f"portfolio_dashboard_{datetime.today().strftime('%Y%m%d')}.xlsx"
STARTING_CAPITAL = 84240.32  # USD
START_DATE = "2024-06-06"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_process_data():
    """Reads trade log and fetches market data."""
    print(f"Reading trade log from {INPUT_FILE}...")
    try:
        df_trades = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        df_trades = pd.read_csv(INPUT_FILE, encoding='big5')
    except FileNotFoundError:
        print("Input file not found. Creating dummy data for structure verification.")
        df_trades = pd.DataFrame([
            {'Date': '2024-06-06', 'Ticker': 'NVDA', 'Side': 'Buy', 'Price': 120.0, 'Qty': 10, 'Fee': 5.0, 'Tax': 0.0},
            {'Date': '2024-06-07', 'Ticker': 'AAPL', 'Side': 'Buy', 'Price': 200.0, 'Qty': 5, 'Fee': 2.0, 'Tax': 0.0}
        ])
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None, None

    # Clean columns
    df_trades.columns = [c.strip() for c in df_trades.columns]
    
    # Common Column Mapping (Fubon Chinese Headers -> Standard English)
    column_mapping = {
        '成交時間': 'Date', '成交日期': 'Date', '交易日期': 'Date', '日期': 'Date',
        '商品名稱': 'Ticker', '股票代號': 'Ticker', '股票代碼': 'Ticker', '代號': 'Ticker', '證券代號': 'Ticker',
        '買賣別': 'Side', '交易類別': 'Side', '買賣': 'Side',
        '成交價格': 'Price', '成交單價': 'Price', '成交價': 'Price', '單價': 'Price',
        '成交股數': 'Qty', '股數': 'Qty', '數量': 'Qty',
        '手續費': 'Fee',
        '交易稅': 'Tax', '稅足': 'Tax'
    }
    
    # Apply mapping for found columns
    df_trades = df_trades.rename(columns=column_mapping)

    # Clean Ticker: Extract symbol from "CompanyName(SYMBOL)" format
    if 'Ticker' in df_trades.columns:
        # Handle both half-width () and full-width （） parentheses
        df_trades['Ticker'] = df_trades['Ticker'].astype(str).apply(lambda x: x.replace('（', '(').replace('）', ')'))
        df_trades['Ticker'] = df_trades['Ticker'].apply(lambda x: x.split('(')[-1].split(')')[0] if '(' in x else x)
        df_trades['Ticker'] = df_trades['Ticker'].str.strip().str.upper()
    
    # Requirement Check
    required_cols = ['Date', 'Ticker', 'Side', 'Price', 'Qty']
    missing = [c for c in required_cols if c not in df_trades.columns]
    if missing:
        print(f"Error: Missing columns {missing}")
        print(f"Detected columns: {df_trades.columns.tolist()}")
        print("Please ensure your CSV header matches standard Fubon export or contains Date, Ticker, Side, Price, Qty.")
        return None, None, None, None

    # Ensure Date format and normalize to midnight (remove time component for daily matching)
    df_trades['Date'] = pd.to_datetime(df_trades['Date']).dt.normalize()
    
    # Get unique tickers
    tickers = df_trades['Ticker'].unique().tolist()
    
    if not tickers:
        print("No tickers found.")
        return None, None, None, None

    print(f"Fetching market data for: {tickers}...")
    
    # Fetch OHLCV
    # Using threads=True for faster download
    market_data = yf.download(tickers, start=START_DATE, progress=False, threads=True)['Close']
    
    # Handle single ticker case
    if isinstance(market_data, pd.Series):
        market_data = market_data.to_frame(name=tickers[0])
    
    # Fill missing days (weekends/holidays) to ensure alignment with Daily_Units
    end_date_dt = datetime.today()
    all_dates = pd.date_range(start=START_DATE, end=end_date_dt, freq='B')
    market_data = market_data.reindex(all_dates).ffill()

    # Fetch Reference Data
    ref_data = []
    print("Fetching company details...")
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            ref_data.append({
                'Ticker': t,
                'Name': info.get('shortName', t),
                'Sector': info.get('sector', 'Unknown'),
                'Industry': info.get('industry', 'Unknown'),
                'Country': info.get('country', 'Unknown'),
                'Beta': info.get('beta', 1.0)
            })
        except Exception as e:
            print(f"Could not fetch info for {t}: {e}")
            ref_data.append({'Ticker': t, 'Name': t, 'Sector': 'Unknown', 'Industry': 'Unknown', 'Country': 'Unknown', 'Beta': 1.0})
    df_ref = pd.DataFrame(ref_data)

    return df_trades, market_data, df_ref, all_dates

def calculate_portfolio_state(df_trades, all_dates, tickers):
    """Calculates daily units, daily cash, and average cost basis."""
    
    df_units = pd.DataFrame(0, index=all_dates, columns=tickers)
    df_cash = pd.DataFrame(0.0, index=all_dates, columns=['Cash_Balance'])
    
    # Cost Basis Dictionary: {Ticker: {'total_shares': 0, 'total_cost': 0.0, 'avg_cost': 0.0}}
    cost_basis_tracker = {t: {'total_shares': 0, 'total_cost': 0.0, 'avg_cost': 0.0} for t in tickers}
    
    current_cash = STARTING_CAPITAL
    current_units = {t: 0 for t in tickers}
    
    df_trades = df_trades.sort_values('Date')
    n_trades = len(df_trades)
    trade_idx = 0
    
    for date in all_dates:
        # Process trades occurring on or before this date that haven't been processed
        # Note: If multiple trades on same day, process all.
        # But 'date' in all_dates might match trade 'Date'.
        


        while trade_idx < n_trades and df_trades.iloc[trade_idx]['Date'] <= date:
            trade = df_trades.iloc[trade_idx]
            ticker = trade['Ticker']
            
            # --- 買賣別判定 ---
            raw_side = str(trade['Side']).strip()
            # 優先檢查常見中文標籤
            if '買' in raw_side:
                side = 'buy'
            elif '賣' in raw_side:
                side = 'sell'
            # 備援檢查英文標籤
            elif 'buy' in raw_side.lower():
                side = 'buy'
            elif 'sell' in raw_side.lower():
                side = 'sell'
            else:
                side = 'unknown'

            qty = float(trade['Qty'])
            price = float(trade['Price'])
            fee = float(trade['Fee']) if pd.notnull(trade.get('Fee')) else 0.0
            tax = float(trade['Tax']) if pd.notnull(trade.get('Tax')) else 0.0
            total_val = price * qty
            
            if side == 'buy':
                # 買進：減少現金，增加庫存
                current_cash -= (total_val + fee)
                current_units[ticker] += qty
                
                # 更新平均成本 (加權平均)
                prev_shares = cost_basis_tracker[ticker]['total_shares']
                prev_cost = cost_basis_tracker[ticker]['total_cost']
                new_cost = prev_cost + total_val + fee # 成本包含手續費
                new_shares = prev_shares + qty
                
                cost_basis_tracker[ticker]['total_shares'] = new_shares
                cost_basis_tracker[ticker]['total_cost'] = new_cost
                cost_basis_tracker[ticker]['avg_cost'] = new_cost / new_shares if new_shares > 0 else 0
                

                
            elif side == 'sell':
                # 賣出：增加現金，減少庫存
                current_cash += (total_val - fee - tax)
                current_units[ticker] -= qty
                
                # 賣出不改變平均成本，但需按比例減少總成本庫存
                avg_c = cost_basis_tracker[ticker]['avg_cost']
                cost_basis_tracker[ticker]['total_shares'] -= qty
                cost_basis_tracker[ticker]['total_cost'] -= (qty * avg_c)
                

                
            trade_idx += 1
        
        # Snapshot for the day
        for t in tickers:
            df_units.at[date, t] = current_units[t]
        
        df_cash.at[date, 'Cash_Balance'] = current_cash
        
    return df_units, df_cash, cost_basis_tracker

def create_excel_dashboard(df_trades, market_data, df_ref, df_units, df_cash, cost_basis, output_path):
    print(f"Generating Excel file at {output_path}...")
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    workbook = writer.book

    # --- Formats ---
    # Colors
    bg_header = '#4472C4'
    font_header = '#FFFFFF'
    
    fmt_currency = workbook.add_format({'num_format': '$#,##0.00'})
    fmt_pct = workbook.add_format({'num_format': '0.00%'})
    fmt_date = workbook.add_format({'num_format': 'yyyy-mm-dd'})
    fmt_header = workbook.add_format({'bold': True, 'bg_color': bg_header, 'font_color': font_header, 'border': 1})
    fmt_header_centered = workbook.add_format({'bold': True, 'bg_color': bg_header, 'font_color': font_header, 'border': 1, 'align': 'center'})
    
    # --- 1. Data Lake Sheets ---
    
    # Trade_Log
    df_trades.to_excel(writer, sheet_name='Trade_Log', index=False)
    
    # Market_Data 
    # Transpose might be better if many tickers? No, dates as rows is standard.
    market_data.to_excel(writer, sheet_name='Market_Data')
    
    # Reference_Data
    df_ref.to_excel(writer, sheet_name='Reference_Data', index=False)
    
    # Daily_Units
    df_units.to_excel(writer, sheet_name='Daily_Units')
    
    # Daily_Cash
    df_cash.to_excel(writer, sheet_name='Daily_Cash')
    
    # --- 2. Analytics Sheets ---
    
    tickers = df_units.columns.tolist()
    n_tickers = len(tickers)
    
    # --- Positions Sheet ---
    ws_pos = workbook.add_worksheet('Positions')
    headers_pos = ['Ticker', 'Company', 'Sector', 'Country', 'Shares', 'Avg Cost', 'Current Price', 'Day Change %', 'Market Value', 'Unrealized P&L', 'P&L %', 'Portfolio Weight']
    ws_pos.write_row(0, 0, headers_pos, fmt_header_centered)
    
    # Start Row index
    start_row = 1
    
    for i, ticker in enumerate(tickers):
        row = start_row + i
        # 0 Ticker
        ws_pos.write(row, 0, ticker)
        
        # 1 Company (Lookup)
        ws_pos.write_formula(row, 1, f'=XLOOKUP(A{row+1},Reference_Data!A:A,Reference_Data!B:B)', value=df_ref[df_ref['Ticker']==ticker]['Name'].values[0] if not df_ref.empty else '')
        
        # 2 Sector (Lookup)
        ws_pos.write_formula(row, 2, f'=XLOOKUP(A{row+1},Reference_Data!A:A,Reference_Data!C:C)', value=df_ref[df_ref['Ticker']==ticker]['Sector'].values[0] if not df_ref.empty else '')
        
        # 3 Country (Lookup)
        ws_pos.write_formula(row, 3, f'=XLOOKUP(A{row+1},Reference_Data!A:A,Reference_Data!E:E)', value=df_ref[df_ref['Ticker']==ticker]['Country'].values[0] if not df_ref.empty else '')
        
        # 4 Shares (From Daily_Units last row)
        col_letter = xlsxwriter.utility.xl_col_to_name(i + 1) # B, C... (Date is A)
        ws_pos.write_formula(row, 4, f'=INDEX(Daily_Units!{col_letter}:{col_letter},COUNT(Daily_Units!A:A)+1)')
        
        # 5 Avg Cost (Static from Python calculation)
        avg = cost_basis[ticker]['avg_cost']
        ws_pos.write(row, 5, avg, fmt_currency)
        
        # 6 Current Price (Index Match on Market Data)
        ws_pos.write_formula(row, 6, f'=INDEX(Market_Data!{col_letter}:{col_letter},COUNT(Market_Data!A:A)+1)', fmt_currency)
        
        # 7 Day Change %
        # (Today - Yesterday) / Yesterday
        # Market Data has Headers in Row 1. Date in A.
        ws_pos.write_formula(row, 7, f'=(G{row+1}-INDEX(Market_Data!{col_letter}:{col_letter},COUNT(Market_Data!A:A)))/INDEX(Market_Data!{col_letter}:{col_letter},COUNT(Market_Data!A:A))', fmt_pct)
        
        # 8 Market Value
        ws_pos.write_formula(row, 8, f'=E{row+1}*G{row+1}', fmt_currency)
        
        # 9 Unrealized P&L
        ws_pos.write_formula(row, 9, f'=I{row+1}-(E{row+1}*F{row+1})', fmt_currency)
        
        # 10 P&L %
        ws_pos.write_formula(row, 10, f'=IF(E{row+1}*F{row+1}<>0, J{row+1}/(E{row+1}*F{row+1}), 0)', fmt_pct)
        
        # 11 Weight
        ws_pos.write_formula(row, 11, f'=I{row+1}/SUM(I$2:I${n_tickers+1})', fmt_pct)

    ws_pos.autofit()

    # --- Equity Curve Sheet ---
    ws_eq = workbook.add_worksheet('Equity_Curve')
    headers_eq = ['Date', 'Cash', 'Invested Value', 'Total NAV', 'Daily Return', 'Cumulative Return', 'Drawdown']
    ws_eq.write_row(0, 0, headers_eq, fmt_header_centered)
    
    n_days = len(df_units)
    
    # We write formulas for all days
    for r in range(n_days):
        row = r + 1
        excel_r = row + 1
        
        # Date
        ws_eq.write_formula(row, 0, f'=Market_Data!A{excel_r}', fmt_date)
        
        # Cash
        ws_eq.write_formula(row, 1, f'=Daily_Cash!B{excel_r}', fmt_currency)
        
        # Invested Value: SUMPRODUCT(Units_Row, Price_Row)
        # Units sheet: B{row}..End{row}
        # Market sheet: B{row}..End{row}
        last_col = xlsxwriter.utility.xl_col_to_name(n_tickers)
        ws_eq.write_formula(row, 2, f'=SUMPRODUCT(Daily_Units!B{excel_r}:{last_col}{excel_r},Market_Data!B{excel_r}:{last_col}{excel_r})', fmt_currency)
        
        # Total NAV
        ws_eq.write_formula(row, 3, f'=B{excel_r}+C{excel_r}', fmt_currency)
        
        # Daily Return
        if r == 0:
            ws_eq.write(row, 4, 0, fmt_pct)
        else:
            ws_eq.write_formula(row, 4, f'=(D{excel_r}-D{excel_r-1})/D{excel_r-1}', fmt_pct)
            
        # Cum Return
        ws_eq.write_formula(row, 5, f'=(D{excel_r}-{STARTING_CAPITAL})/{STARTING_CAPITAL}', fmt_pct)
        
        # Drawdown
        ws_eq.write_formula(row, 6, f'=D{excel_r}/MAX(D$2:D{excel_r})-1', fmt_pct)

    ws_eq.autofit()
    
    # --- Dashboard Sheet ---
    ws_dash = workbook.add_worksheet('Dashboard')
    ws_dash.hide_gridlines(2)
    
    # Summary Table
    ws_dash.write('B2', "Current NAV", fmt_header)
    ws_dash.write_formula('C2', f'=Equity_Curve!D{n_days+1}', fmt_currency)
    
    ws_dash.write('B3', "Total Return", fmt_header)
    ws_dash.write_formula('C3', f'=Equity_Curve!F{n_days+1}', fmt_pct)
    
    ws_dash.write('B4', "Max Drawdown", fmt_header)
    ws_dash.write_formula('C4', f'=MIN(Equity_Curve!G:G)', fmt_pct)
    
    ws_dash.write('B5', "Invested %", fmt_header)
    ws_dash.write_formula('C5', f'=Equity_Curve!C{n_days+1}/Equity_Curve!D{n_days+1}', fmt_pct)
    
    # Charts
    chart_nav = workbook.add_chart({'type': 'line'})
    chart_nav.add_series({
        'name': 'NAV',
        'categories': ['Equity_Curve', 1, 0, n_days, 0],
        'values':     ['Equity_Curve', 1, 3, n_days, 3],
        'line':       {'color': '#4472C4', 'width': 2.25}
    })
    chart_nav.set_title({'name': 'Portfolio Construction'})
    chart_nav.set_x_axis({'name': 'Date', 'date_axis': True})
    chart_nav.set_legend({'none': True})
    ws_dash.insert_chart('E2', chart_nav, {'x_scale': 1.5, 'y_scale': 1.5})


    chart_alloc = workbook.add_chart({'type': 'pie'})
    chart_alloc.add_series({
        'name': 'Allocation',
        'categories': ['Positions', 1, 0, n_tickers, 0],
        'values':     ['Positions', 1, 8, n_tickers, 8], # Market Value col
        'data_labels': {'percentage': True, 'position': 'outside_end'}
    })
    chart_alloc.set_title({'name': 'Current Allocation'})
    ws_dash.insert_chart('E25', chart_alloc)


    # --- Sector Exposure ---
    ws_sec = workbook.add_worksheet('Sector_Exposure')
    ws_sec.write_row(0, 0, ['Sector', 'Value', 'Weight'], fmt_header_centered)
    
    sectors = sorted(df_ref['Sector'].unique()) if not df_ref.empty else ['Unknown']
    for i, sec in enumerate(sectors):
        r = i + 1
        ws_sec.write(r, 0, sec)
        # SumIf from Positions. Sector is Col C (Index 3rd, Ref !C:C)
        # Market Value is Col I (Index 9th, Ref !I:I)
        ws_sec.write_formula(r, 1, f'=SUMIF(Positions!C:C, A{r+1}, Positions!I:I)', fmt_currency)
        ws_sec.write_formula(r, 2, f'=B{r+1}/SUM(B$2:B${len(sectors)+1})', fmt_pct)
    ws_sec.autofit()

    # --- Geographic Exposure ---
    ws_geo = workbook.add_worksheet('Geographic_Exposure')
    ws_geo.write_row(0, 0, ['Country', 'Value', 'Weight'], fmt_header_centered)
    
    countries = sorted(df_ref['Country'].unique()) if not df_ref.empty else ['Unknown']
    for i, ctry in enumerate(countries):
        r = i + 1
        ws_geo.write(r, 0, ctry)
        # Country is Col D in Positions (Index 4th, Ref !D:D)
        ws_geo.write_formula(r, 1, f'=SUMIF(Positions!D:D, A{r+1}, Positions!I:I)', fmt_currency)
        ws_geo.write_formula(r, 2, f'=B{r+1}/SUM(B$2:B${len(countries)+1})', fmt_pct)
    ws_geo.autofit()

    writer.close()
    print("Dashboard created successfully.")

if __name__ == "__main__":
    df_trades, market_data, df_ref, all_dates = load_process_data()
    if df_trades is not None:
        df_units, df_cash, cost_basis = calculate_portfolio_state(df_trades, all_dates, market_data.columns.tolist())
        create_excel_dashboard(df_trades, market_data, df_ref, df_units, df_cash, cost_basis, OUTPUT_FILE)
