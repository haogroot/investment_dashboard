import pandas as pd
import yfinance as yf
import xlsxwriter
from datetime import datetime, timedelta
import os

import argparse
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

def load_process_data(input_file=None):
    """Reads trade log and fetches market data."""
    target_file = input_file if input_file else INPUT_FILE
    print(f"Reading trade log from {target_file}...")
    try:
        df_trades = pd.read_csv(target_file, encoding='utf-8')
    except UnicodeDecodeError:
        df_trades = pd.read_csv(target_file, encoding='big5')
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

    # Fetch OHLCV + Benchmark
    # Using threads=True for faster download
    tickers_to_fetch = tickers + ['^GSPC', 'TWD=X']
    print(f"Fetching market data for: {tickers_to_fetch}...")
    
    market_data = yf.download(tickers_to_fetch, start=START_DATE, progress=False, threads=True)['Close']
    
    # Handle single ticker case (yf returns Series if only 1 ticker asked, DataFrame if more)
    # But now we always fetch at least 2 (Ticker + GSPC), so it should return DataFrame.
    # However, if Ticker IS GSPC (duplicated), might need care.
    if isinstance(market_data, pd.Series):
        market_data = market_data.to_frame()
    
    # Rename ^GSPC column to 'SP500', easier for Excel references
    if '^GSPC' in market_data.columns:
        market_data = market_data.rename(columns={'^GSPC': 'SP500'})
    
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
                'Beta': info.get('beta', 1.0),
                'Type': info.get('quoteType', 'EQUITY'),
                'Strike': info.get('strikePrice', 0),
                'Expiry': info.get('expireDate', 0)
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
        ref_row = df_ref[df_ref['Ticker']==ticker]
        comp_name = ref_row['Name'].values[0] if not ref_row.empty else ''
        ws_pos.write_formula(row, 1, f'=XLOOKUP(A{row+1},Reference_Data!A:A,Reference_Data!B:B)', value=comp_name)
        
        # 2 Sector (Lookup)
        sector = ref_row['Sector'].values[0] if not ref_row.empty else ''
        ws_pos.write_formula(row, 2, f'=XLOOKUP(A{row+1},Reference_Data!A:A,Reference_Data!C:C)', value=sector)
        
        # 3 Country (Lookup)
        country = ref_row['Country'].values[0] if not ref_row.empty else ''
        ws_pos.write_formula(row, 3, f'=XLOOKUP(A{row+1},Reference_Data!A:A,Reference_Data!E:E)', value=country)
        
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
    chart_nav.set_y_axis({'name': 'Net Asset Value', 'num_format': '$#,##0'})
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

    print("Starting Excel generation...")
    try:
        # --- 3. Risk Analytics Sheets ---

        # --- Correlation Matrix ---
        print("Building Correlation Matrix...")
        ws_corr = workbook.add_worksheet('Correlation')
        # Headers: Tickers + SP500
        corr_tickers = tickers + ['SP500'] if 'SP500' in market_data.columns else tickers
        ws_corr.write_row(0, 1, corr_tickers, fmt_header_centered)
        ws_corr.write_column(1, 0, corr_tickers, fmt_header_centered)
        
        # Pre-calculate column letters for Market Data
        # Market Data Cols: Date(A), Tickers(B, C...), SP500(Last)
        md_col_map = {}
        # Adjust index because Market_Data sheet has Date in Col A (Index 0)
        # So market_data.columns[0] (Ticker1) is at Excel Col B (Index 1)
        for i, t in enumerate(market_data.columns):
            md_col_map[t] = xlsxwriter.utility.xl_col_to_name(i+1)

        n_data_rows = len(market_data)
        
        for r, t_row in enumerate(corr_tickers):
            for c, t_col in enumerate(corr_tickers):
                if t_row in md_col_map and t_col in md_col_map:
                    col1 = md_col_map[t_row]
                    col2 = md_col_map[t_col]
                    # Use whole column references (e.g. B:B) to allow appending new data without breaking formulas
                    # CORREL ignores text headers.
                    formula = f'=CORREL(Market_Data!{col1}:{col1}, Market_Data!{col2}:{col2})'
                    ws_corr.write_formula(r+1, c+1, formula, workbook.add_format({'num_format': '0.00'}))

        ws_corr.conditional_format(1, 1, len(corr_tickers), len(corr_tickers), {
            'type': '3_color_scale',
            'min_color': '#63BE7B', 'mid_color': '#FFEB84', 'max_color': '#F8696B'
        })
        ws_corr.autofit()

        # --- Helper: Daily Returns Sheet ---
        print("Building Daily Returns...")
        ws_ret = workbook.add_worksheet('Daily_Returns')
        ws_ret.hide()
        ws_ret.write_row(0, 0, ['Date'] + corr_tickers, fmt_header_centered)
        
        for day_i in range(n_data_rows - 1): # Returns are N-1
            row = day_i + 1
            excel_row = row + 1
            # Date
            ws_ret.write_formula(row, 0, f'=Market_Data!A{excel_row+1}', fmt_date)
            
            for c, t in enumerate(corr_tickers):
                if t in md_col_map:
                    col = md_col_map[t]
                    # LN(P_t / P_t-1)
                    ws_ret.write_formula(row, c+1, f'=LN(Market_Data!{col}{excel_row+1}/Market_Data!{col}{excel_row})', fmt_pct)

        # --- Helper: Daily Drawdowns Sheet ---
        print("Building Daily Drawdowns...")
        ws_dd = workbook.add_worksheet('Daily_Drawdowns')
        ws_dd.hide()
        ws_dd.write_row(0, 0, ['Date'] + corr_tickers, fmt_header_centered)
        
        for day_i in range(n_data_rows):
            row = day_i + 1
            excel_row = row + 1
            # Date
            ws_dd.write_formula(row, 0, f'=Market_Data!A{excel_row}', fmt_date)
            
            for c, t in enumerate(corr_tickers):
                if t in md_col_map:
                    col = md_col_map[t]
                    # Formula: Price / MAX(Price_Start:Price_Today) - 1
                    # Note: We still need dynamic range for MAX because it's "running max".
                    # But the Update script will just copy the previous formula row and adjust cell refs.
                    # Standard relative references work fine for row-by-row calc.
                    # However, to be "update-friendly", we ensure the formula relies on relative row refs.
                    # The current formula uses $2 constant start: MAX(Col$2:Col{row}). This is fine.
                    formula = f'=Market_Data!{col}{excel_row}/MAX(Market_Data!{col}$2:Market_Data!{col}{excel_row}) - 1'
                    ws_dd.write_formula(row, c+1, formula, fmt_pct)

        # --- Risk Metrics ---
        print("Building Risk Metrics...")
        ws_risk = workbook.add_worksheet('Risk_Metrics')
        headers_risk = ['Ticker', 'Ann Return', 'Ann Volatility', 'Sharpe Ratio', 'Beta (vs SP500)', 'Max Drawdown', 'VaR (95%)', 'VaR (99%)', 'CVaR (95%)']
        ws_risk.write_row(0, 0, headers_risk, fmt_header_centered)
        
        rf_rate = 0.04 # Risk Free Rate assumption
        
        ret_col_map = {t: xlsxwriter.utility.xl_col_to_name(i+1) for i, t in enumerate(corr_tickers)}
        n_ret_rows = n_data_rows - 1

        for i, t in enumerate(tickers):
            if t in ret_col_map:
                r = i + 1
                c_ret = ret_col_map[t]
                sp_ret = ret_col_map.get('SP500', c_ret)
                
                # Use whole column references (e.g. C:C) for Risk Metrics to auto-include new daily returns
                ret_rng = f'Daily_Returns!{c_ret}:{c_ret}'
                bench_rng = f'Daily_Returns!{sp_ret}:{sp_ret}'
                
                ws_risk.write(r, 0, t)
                ws_risk.write_formula(r, 1, f'=AVERAGE({ret_rng})*252', fmt_pct)
                ws_risk.write_formula(r, 2, f'=STDEV.P({ret_rng})*SQRT(252)', fmt_pct)
                ws_risk.write_formula(r, 3, f'=(B{r+1}-{rf_rate})/C{r+1}', workbook.add_format({'num_format': '0.00'}))
                ws_risk.write_formula(r, 4, f'=SLOPE({ret_rng}, {bench_rng})', workbook.add_format({'num_format': '0.00'}))
                
                # Max Drawdown: MIN(Daily_Drawdowns!Col)
                dd_col = ret_col_map[t]
                # MIN works with whole columns
                ws_risk.write_formula(r, 5, f'=MIN(Daily_Drawdowns!{dd_col}:{dd_col})', fmt_pct)
                ws_risk.write_formula(r, 6, f'=PERCENTILE.INC({ret_rng}, 0.05)', fmt_pct)
                ws_risk.write_formula(r, 7, f'=PERCENTILE.INC({ret_rng}, 0.01)', fmt_pct)
                ws_risk.write_formula(r, 8, f'=AVERAGEIF({ret_rng}, "<"&G{r+1})', fmt_pct)
                
        ws_risk.autofit()
        
        # --- Stress Testing ---
        print("Building Stress Testing...")
        ws_stress = workbook.add_worksheet('Stress_Testing')
        headers_stress = ['Scenario', 'Market Change', 'Portfolio Impact (Est.)', 'Est. P&L']
        ws_stress.write_row(0, 0, headers_stress, fmt_header_centered)
        
        scenarios = [
            ('Market Crash', -0.20),
            ('Correction', -0.10),
            ('Flash Crash', -0.05),
            ('Rally', 0.10)
        ]
        
        for i, (name, chg) in enumerate(scenarios):
            r = i + 1
            ws_stress.write(r, 0, name)
            ws_stress.write(r, 1, chg, fmt_pct)
            beta_formula = f'SUMPRODUCT(Positions!L2:L{n_tickers+1}, Risk_Metrics!E2:E{n_tickers+1})'
            ws_stress.write_formula(r, 2, f'={beta_formula}*B{r+1}', fmt_pct)
            ws_stress.write_formula(r, 3, f'=C{r+1}*Dashboard!C2', fmt_currency)
        
        ws_stress.autofit()
        
        # --- Investment Theses ---
        print("Building Investment Theses...")
        ws_thesis = workbook.add_worksheet('Investment_Theses')
        headers_thesis = ['Ticker', 'Weight', 'Investment Thesis', 'Catalyst', 'Edge']
        ws_thesis.write_row(0, 0, headers_thesis, fmt_header_centered)
        
        # Pre-defined Theses Data
        theses_data = {
            'AAPL': {
                'Thesis': "iPhone 安裝基數接近 20 億，創造巨大的 Services 收入機會。硬體已成熟，但 Services 每年增長 15% 以上。",
                'Catalyst': "印度製造擴張減少中國風險，解鎖下一個 10 億用戶。",
                'Edge': "市場將其視為無增長的硬體公司，忽視了 Services 的複利效應。"
            },
            'MSFT': {
                'Thesis': "Cloud + AI 領導地位。Azure 增長速度快於 AWS。Copilot 貨幣化才剛開始。",
                'Catalyst': "Enterprise Copilot 的採用在 2026 年達到轉折點。",
                'Edge': "隨著市場意識到 AI 收入是真實的而非炒作，估值倍數擴張。"
            },
            'GOOGL': {
                'Thesis': "佔主導地位的 Search 壟斷，YouTube 潛力尚未充分發揮。Search (SGE) 中的 AI 整合捍衛了護城河。",
                'Catalyst': "Gemini Ultra 發布縮小與 GPT-4 的差距；Waymo 擴張。",
                'Edge': "相對於其他 Mag-7 同業的估值斷層；Cloud 獲利能力出現轉機。"
            },
            'NVDA': {
                'Thesis': "AI 革命的默認基礎設施層。CUDA 軟體護城河形成高轉換成本。",
                'Catalyst': "B100/Blackwell 發布維持 ASP 主導地位；Sovereign AI 需求。",
                'Edge': "供應鏈掌控能力 (CoWoS) 創造領先競爭對手多年的優勢。"
            },
            'TSLA': {
                'Thesis': "不僅是一家汽車公司：Energy storage + FSD/Robotaxi 期權價值。EV 製造的成本領導地位。",
                'Catalyst': "FSD V12 廣泛發布；第 3 代平台（廉價車）公告；Optimus 進展。",
                'Edge': "在真實世界 AI 駕駛里程中的數據優勢是不可逾越的。"
            },
            'VTI': {
                'Thesis': "核心投資組合重心。以極低費用 (0.03%) 捕捉美國股票市場的總報酬。",
                'Catalyst': "美國經濟韌性；Fed 轉向降息帶動全體市場。",
                'Edge': "保證市場回報；零經理人風險。"
            },
            'QQQ': {
                'Thesis': "押注創新經濟。Nasdaq-100 基於規則的方法論自動捕捉贏家。",
                'Catalyst': "AI 生產力爆發使科技密集型指數不成比例地獲益。",
                'Edge': "被動動能策略，數十年來表現優於主動管理。"
            },
            'VT': {
                'Thesis': "全球多元化避險，對抗美元主導地位或估值萎縮。",
                'Catalyst': "新興市場均值回歸；美元走弱週期。",
                'Edge': "對全球資本主義的最徹底不可知論押注。"
            },
            'BND': {
                'Thesis': "收入來源與波動緩衝。在衰退中與股票的相關性回歸正常。",
                'Catalyst': "Fed 降息週期於 2024/2025 開始將推升債券價格（存續期間效應）。",
                'Edge': "在經歷十年的 ZIRP 後，殖利率終於具有吸引力。"
            },
            'COHR': { # Coherent
                'Thesis': "SiC 與光通訊材料科學領導者。AI Networking 的鏟子與鋤頭。",
                'Catalyst': "AI 集群驅動的 Datacenter 收發器升級週期 (800G/1.6T)。",
                'Edge': "雷射元件的垂直整合提供邊際耐用性。"
            },
            'LITE': { # Lumentum
                'Thesis': "光子學應用。電信與 Datacenter 互連的關鍵供應商。",
                'Catalyst': "庫存修正週期結束；AI 後端網路建設 (Active cables)。",
                'Edge': "高速收發器市場的雙寡頭結構。"
            },
            'AVGO': { # Broadcom
                'Thesis': "定製矽晶片之王。超大規模雲端運算商的 Custom ASICs + 高利潤軟體 (VMware)。",
                'Catalyst': "AI ASIC 訂單增長；VMware 成本協同效應實現。",
                'Edge': "一流的資本配置與 M&A 整合記錄。"
            },
            'MRVL': { # Marvell
                'Thesis': "純基礎設施矽晶片。雲端巨頭的光學 DSPs 與定製運算。",
                'Catalyst': "加速運算轉向定製矽晶片 (ASIC)，有利於 Marvell 的設計獲勝管道。",
                'Edge': "在高速數據移動 (PAM4 DSPs) 領域擁有獨特的 IP 組合。"
            },
            'SPY': {
                'Thesis': "標準基準風險敞口。高流動性和期權靈活性。",
                'Catalyst': "企業盈利復甦；美國例外論。",
                'Edge': "美國股市的標準計量單位。"
            }
        }

        # Formats for Thesis sheet
        fmt_text_wrap = workbook.add_format({'text_wrap': True, 'valign': 'top', 'border': 1})
        fmt_zebra_odd = workbook.add_format({'text_wrap': True, 'valign': 'top', 'border': 1, 'bg_color': '#F2F2F2'})
        fmt_pct_wrap = workbook.add_format({'num_format': '0.00%', 'valign': 'top', 'border': 1})
        fmt_pct_zebra = workbook.add_format({'num_format': '0.00%', 'valign': 'top', 'border': 1, 'bg_color': '#F2F2F2'})
        
        # Set Column Widths
        ws_thesis.set_column('A:A', 10) # Ticker
        ws_thesis.set_column('B:B', 10) # Weight
        ws_thesis.set_column('C:C', 50) # Thesis
        ws_thesis.set_column('D:D', 40) # Catalyst
        ws_thesis.set_column('E:E', 40) # Edge
        
        row_height = 55
        
        for i, t in enumerate(tickers):
            row = i + 1
            # Zebra Striping
            fmt_txt = fmt_zebra_odd if row % 2 == 1 else fmt_text_wrap
            fmt_p = fmt_pct_zebra if row % 2 == 1 else fmt_pct_wrap
            
            ws_thesis.set_row(row, row_height)
            
            # Ticker
            ws_thesis.write(row, 0, t, fmt_txt)
            
            # Weight (Lookup from Positions)
            # Positions Sheet: A=Ticker, L=Weight.
            # XLOOKUP(Ticker, Positions!A:A, Positions!L:L)
            ws_thesis.write_formula(row, 1, f'=XLOOKUP(A{row+1},Positions!A:A,Positions!L:L)', fmt_p)
            
            # Content
            data = theses_data.get(t, {'Thesis': '', 'Catalyst': '', 'Edge': ''})
            ws_thesis.write(row, 2, data['Thesis'], fmt_txt)
            ws_thesis.write(row, 3, data['Catalyst'], fmt_txt)
            ws_thesis.write(row, 4, data['Edge'], fmt_txt)

        writer.close()
        print("Dashboard created successfully.")
    except Exception as e:
        print(f"ERROR in create_excel_dashboard: {e}")
        import traceback
        traceback.print_exc()
        try:
            writer.close()
        except:
            pass

if __name__ == "__main__":
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description='Build Portfolio Dashboard')
    parser.add_argument('input_file', nargs='?', default=None, help='Input trade CSV filename (default: configured INPUT_FILE)')
    args = parser.parse_args()

    # Resolve Input File Path if provided
    input_path = None
    if args.input_file:
        arg_path = Path(args.input_file)
        if arg_path.is_absolute():
            input_path = arg_path
        else:
            # Assume relative to trade_source if just filename, or relative to cwd?
            # User said "python3 build_portfolio.py fubin-trade_20260219.csv"
            # It's safer to check current dir first, then trade_source?
            # Or assume trade_source as default location for data.
            input_path = BASE_DIR / "trade_source" / args.input_file

    df_trades, market_data, df_ref, all_dates = load_process_data(input_file=input_path) 
    
    if df_trades is not None:
        # Tickers for portfolio are those in df_ref (excluding SP500 if it was added separately, but df_ref is built from df_trades tickers)
        # However, df_ref is built in load_process_data from df_trades.Ticker.unique() BEFORE SP500 is fetched attached to market_data.
        # So df_ref['Ticker'] is safe.
        portfolio_tickers = df_ref['Ticker'].tolist()
        df_units, df_cash, cost_basis = calculate_portfolio_state(df_trades, all_dates, portfolio_tickers)
        create_excel_dashboard(df_trades, market_data, df_ref, df_units, df_cash, cost_basis, OUTPUT_FILE)
