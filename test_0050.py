from tw_stock_loader import load_tw_inventory
import glob

for f in glob.glob('trade_source/*.csv'):
    if 'debby' in f.lower() or 'fubon' in f.lower() or True:
        try:
            res = load_tw_inventory(f, 'user')
            if res and res['tw_positions']:
                for p in res['tw_positions']:
                    if p['ticker'] == '0050.TW':
                        print(f"File {f}, owner user: 0050 PnL {p['pnl_pct']:.2%}, Shares: {p['shares']}, Cost: {p['avg_cost']}, Price: {p['price']}")
        except Exception as e:
            pass
