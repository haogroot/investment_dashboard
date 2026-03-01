"""
tw_trade_loader.py — 台股交易歷史 CSV 解析器

解析如 Debby_TW_trade_20260228.csv 格式的台股交易紀錄，
輸出與 portfolio_analytics.py 中 history_grouped 結構一致的分組資料。
"""
import pandas as pd
from pathlib import Path
from itertools import groupby


def load_tw_trade_history(file_path, owner='Unknown'):
    """
    解析台股交易 CSV 檔案，回傳按月份分組的交易列表。

    Args:
        file_path: CSV 檔案路徑
        owner: 持有人名稱（如 'Debby'）

    Returns:
        list of dicts, 每個 dict 含 'month' 和 'trades' key，
        結構與 portfolio_analytics.py 的 trade_history_grouped 相同。
        若讀取失敗則回傳空 list。
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Warning: TW trade file not found: {file_path}")
        return []

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error reading TW trade CSV {file_path}: {e}")
        return []

    # Expected columns: 股名, 日期, 成交股數, 淨收付, 成交單價, 成交價金, 手續費, 交易稅
    required_cols = ['股名', '日期', '成交股數', '淨收付', '成交單價', '成交價金']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Missing column '{col}' in {file_path.name}. Skipping.")
            return []

    history_list = []

    for _, row in df.iterrows():
        try:
            # Parse date (YYYY/MM/DD)
            date_str = str(row['日期']).strip()
            date_obj = pd.to_datetime(date_str, format='%Y/%m/%d')

            # Parse numeric values (remove commas)
            def parse_num(val):
                if pd.isna(val):
                    return 0.0
                s = str(val).replace(',', '').strip()
                try:
                    return float(s)
                except ValueError:
                    return 0.0

            net_amount = parse_num(row['淨收付'])  # 正 = 賣出收入, 負 = 買入支出
            qty = parse_num(row['成交股數'])
            price = parse_num(row['成交單價'])
            trade_value = parse_num(row['成交價金'])
            fee = parse_num(row.get('手續費', 0))
            tax = parse_num(row.get('交易稅', 0))

            stock_name = str(row['股名']).strip()

            # Determine buy/sell from 淨收付 sign
            if net_amount < 0:
                side = 'Buy'
                side_zh = '買入'
            else:
                side = 'Sell'
                side_zh = '賣出'

            total_val = abs(net_amount)

            month_key = date_obj.strftime('%Y-%m')

            history_list.append({
                'date_str': date_obj.strftime('%Y-%m-%d'),
                'month_str': date_obj.strftime('%b'),
                'day_str': date_obj.strftime('%d'),
                'month_key': month_key,
                'ticker': stock_name,
                'name': stock_name,
                'side': side,
                'side_zh': side_zh,
                'qty': qty,
                'price': price,
                'total_val': total_val,
                'fee': fee,
                'tax': tax,
                'currency': 'TWD',
                'owner': owner,
            })
        except Exception as e:
            print(f"Warning: Skipping row in {file_path.name}: {e}")
            continue

    # Sort descending by date
    history_list.sort(key=lambda x: x['date_str'], reverse=True)

    # Group by month
    grouped = []
    for k, g in groupby(history_list, key=lambda x: x['month_key']):
        grouped.append({
            'month': k,
            'trades': list(g)
        })

    print(f"Loaded {len(history_list)} TW trades from {file_path.name} (owner: {owner})")
    return grouped
