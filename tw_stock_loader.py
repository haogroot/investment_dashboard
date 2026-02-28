"""
Taiwan Stock Inventory Loader
Reads CTBC (中信) securities inventory CSV and fetches market data via yfinance.
Returns structured position data compatible with the existing positions template format.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path


def load_tw_inventory(csv_path, owner='Unknown'):
    """
    Reads a Taiwan stock inventory CSV (CTBC format) or trade history CSV and enriches it with yfinance data.

    Args:
        csv_path: Path to the CSV file.
        owner: Name of the family member who owns these positions.

    Returns:
        dict with keys:
            'tw_positions': list of position dicts (same schema as US positions)
            'tw_summary': dict with summary stats (total_value, count, total_pnl, total_return_pct)
        Returns None if loading fails.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[TW] Inventory file not found: {csv_path}")
        return None

    # --- 1. Read CSV (auto-detect encoding) ---
    print(f"[TW] Reading inventory from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='big5')

    # Clean column names (strip whitespace and tab characters)
    df.columns = [c.strip().strip('\t') for c in df.columns]

    KNOWN_TICKERS = {
        '中信金': '2891', '京城銀': '2809', '元大台灣50': '0050', '冠德': '2520',
        '台新新光金': '2887', '台積電': '2330', '國泰金': '2882', '富華新': '3056',
        '富邦台50': '006208', '復華日本龍頭': '00949', '技嘉': '2376', '新潤': '6186',
        '新美齊': '2442', '智基': '6294', '欣陸': '3703', '永豐金': '2890',
        '聯上發': '2537', '長虹': '5534', '順天': '5525',
    }

    if '淨收付' in df.columns and '成交股數' in df.columns:
        # Trade History format
        print(f"[TW] Detected trade history format for {owner}.")
        df['Date_Parsed'] = pd.to_datetime(df['日期'])
        df = df.sort_values('Date_Parsed')

        inventory = {}
        for _, row in df.iterrows():
            name = str(row['股名']).strip()
            shares_str = str(row['成交股數']).replace(',', '')
            shares = pd.to_numeric(shares_str, errors='coerce') or 0
            net_amt_str = str(row['淨收付']).replace(',', '')
            net_amt = pd.to_numeric(net_amt_str, errors='coerce') or 0
            
            # Apply 1-to-4 stock split for 0050 on trades prior to or on 2025/6/17
            is_0050 = (name == '元大台灣50' or '0050' in name)
            if is_0050 and pd.notna(row['Date_Parsed']):
                if row['Date_Parsed'] <= pd.to_datetime('2025-06-17'):
                    shares *= 4
                    
            if name not in inventory:
                inventory[name] = {'shares': 0, 'total_cost': 0}
                
            if net_amt < 0: # Buy
                cost = abs(net_amt)
                inventory[name]['shares'] += shares
                inventory[name]['total_cost'] += cost
            elif net_amt > 0: # Sell
                if inventory[name]['shares'] > 0:
                    avg_c = inventory[name]['total_cost'] / inventory[name]['shares']
                    inventory[name]['shares'] -= shares
                    inventory[name]['total_cost'] -= (shares * avg_c)
                    if inventory[name]['shares'] <= 0:
                        inventory[name]['shares'] = 0
                        inventory[name]['total_cost'] = 0

        inventory_rows = []
        for name, data in inventory.items():
            if data['shares'] > 0:
                avg_c = data['total_cost'] / data['shares'] if data['shares'] > 0 else 0
                ticker = KNOWN_TICKERS.get(name, name)
                raw_name = f"{name}({ticker})" if ticker != name else name
                inventory_rows.append({
                    'RawName': raw_name,
                    'Shares': data['shares'],
                    'AvgCost': avg_c,
                    'TotalCost': data['total_cost']
                })
        df = pd.DataFrame(inventory_rows)

    else:
        # Inventory format
        # --- 2. Column Mapping ---
        column_mapping = {
            '股票名稱': 'RawName',
            '庫存股數': 'Shares',
            '市價': 'Price',
            '市值': 'MarketValue',
            '成交均價': 'AvgCost',
            '損益兩平價': 'BreakEvenPrice',
            '投入成本': 'TotalCost',
            '投資損益': 'PnL',
            '參考淨值': 'RefNAV',
            '報酬率(%)': 'ReturnPct',
            '委託別': 'OrderType',
        }

        df = df.rename(columns=column_mapping)

    # --- 3. Filter valid rows (drop summary row and NaN rows) ---
    # The last row is a summary row with '合計' in the Price column
    df = df.dropna(subset=['RawName'])

    # Ensure Shares is numeric
    df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce')
    df = df.dropna(subset=['Shares'])
    df = df[df['Shares'] > 0]

    if df.empty:
        print("[TW] No valid positions found in inventory file.")
        return None

    # --- 4. Extract ticker from name format like "台積電(2330)" ---
    def extract_ticker(raw_name):
        raw_name = str(raw_name).strip()
        # Handle both half-width () and full-width （）
        raw_name = raw_name.replace('（', '(').replace('）', ')')
        if '(' in raw_name and ')' in raw_name:
            ticker = raw_name.split('(')[-1].split(')')[0].strip()
            name = raw_name.split('(')[0].strip()
            return ticker, name
        return raw_name, raw_name

    df[['Ticker', 'LocalName']] = df['RawName'].apply(
        lambda x: pd.Series(extract_ticker(x))
    )

    # Add .TW suffix for yfinance
    df['YFTicker'] = df['Ticker'].apply(lambda x: f"{x}.TW")

    # --- 5. Fetch yfinance data ---
    tickers_tw = df['YFTicker'].tolist()
    print(f"[TW] Fetching market data for: {tickers_tw}...")

    # Fetch latest prices
    try:
        yf_prices = yf.download(tickers_tw, period='5d', progress=False, threads=True)['Close']
        if isinstance(yf_prices, pd.Series):
            yf_prices = yf_prices.to_frame()
    except Exception as e:
        print(f"[TW] Warning: Could not fetch prices from yfinance: {e}")
        yf_prices = pd.DataFrame()

    # Fetch company info
    print("[TW] Fetching company details...")
    ref_data = {}
    for yf_ticker in tickers_tw:
        try:
            ticker_obj = yf.Ticker(yf_ticker)
            info = ticker_obj.info
            ref_data[yf_ticker] = {
                'name': info.get('shortName', info.get('longName', yf_ticker)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Taiwan'),
                'beta': info.get('beta'),  # None if not available — preserved for display
                'quoteType': info.get('quoteType', 'EQUITY'),
            }
        except Exception as e:
            print(f"[TW] Could not fetch info for {yf_ticker}: {e}")
            ref_data[yf_ticker] = {
                'name': yf_ticker,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'country': 'Taiwan',
                'beta': None,  # yfinance not reachable — mark as no data
                'quoteType': 'EQUITY',
            }

    # --- 6. Build positions list ---
    positions = []
    total_market_value = 0
    total_cost = 0
    stock_count = 0
    etf_count = 0

    for _, row in df.iterrows():
        yf_ticker = row['YFTicker']
        ticker_str = row['Ticker']
        shares = float(row['Shares'])

        # Get latest price from yfinance, fallback to CSV market price
        csv_price = pd.to_numeric(row.get('Price', 0), errors='coerce') or 0
        yf_price = csv_price  # default fallback
        if not yf_prices.empty and yf_ticker in yf_prices.columns:
            latest = yf_prices[yf_ticker].dropna()
            if not latest.empty:
                yf_price = float(latest.iloc[-1])

        # Avg cost from CSV
        avg_cost_raw = row.get('AvgCost', 0)
        avg_cost = pd.to_numeric(avg_cost_raw, errors='coerce') or 0

        # Market value
        market_value = shares * yf_price

        # Cost & PnL
        total_cost_pos = shares * avg_cost if avg_cost > 0 else 0
        unrealized_pnl = market_value - total_cost_pos if total_cost_pos > 0 else 0
        pnl_pct = (unrealized_pnl / total_cost_pos) if total_cost_pos > 0 else 0

        # Reference data from yfinance
        info = ref_data.get(yf_ticker, {})
        quote_type = info.get('quoteType', 'EQUITY')

        # Use yfinance name if available, otherwise use local name from CSV
        display_name = info.get('name', row['LocalName'])
        if display_name == yf_ticker:
            display_name = row['LocalName']

        # Count stock vs ETF
        if quote_type == 'ETF':
            etf_count += 1
        else:
            stock_count += 1

        total_market_value += market_value
        total_cost += total_cost_pos

        positions.append({
            'ticker': yf_ticker,
            'ticker_local': ticker_str,
            'name': display_name,
            'name_local': row['LocalName'],
            'sector': info.get('sector', 'Unknown'),
            'country': info.get('country', 'Taiwan'),
            'shares': shares,
            'avg_cost': avg_cost,
            'price': yf_price,
            'market_value': market_value,
            'unrealized_pnl': unrealized_pnl,
            'pnl_pct': pnl_pct,
            'weight': 0,  # calculated below
            'type': quote_type,
            'beta': info.get('beta'),  # None means no data; template will display '—'
            'owner': owner,
        })

    # Calculate weights
    for pos in positions:
        pos['weight'] = pos['market_value'] / total_market_value if total_market_value > 0 else 0

    # Sort by market value descending
    positions.sort(key=lambda x: x['market_value'], reverse=True)

    # Summary stats
    total_pnl = sum(p['unrealized_pnl'] for p in positions)
    total_return_pct = (total_pnl / total_cost) if total_cost > 0 else 0

    summary = {
        'total_value': total_market_value,
        'total_cost': total_cost,
        'total_pnl': total_pnl,
        'total_return_pct': total_return_pct,
        'stock_count': stock_count,
        'etf_count': etf_count,
        'position_count': len(positions),
    }

    print(f"[TW] Loaded {len(positions)} positions, total value: NT$ {total_market_value:,.0f}")
    return {
        'tw_positions': positions,
        'tw_summary': summary,
    }
