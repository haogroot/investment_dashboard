import unittest
import pandas as pd
from datetime import datetime
from build_portfolio import calculate_portfolio_state, STARTING_CAPITAL

class TestPortfolioCalculation(unittest.TestCase):
    
    def setUp(self):
        # Common setup
        self.tickers = ['AAPL', 'GOOGL']
        self.start_date = '2024-01-01'
        self.end_date = '2024-01-10'
        self.all_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')

    def test_simple_buy(self):
        """Test simple Buy logic: Cash decreases, Units increase."""
        trades_data = [
            {'Date': '2024-01-02', 'Ticker': 'AAPL', 'Side': 'Buy', 'Price': 100.0, 'Qty': 10, 'Fee': 0, 'Tax': 0}
        ]
        df_trades = pd.DataFrame(trades_data)
        df_trades['Date'] = pd.to_datetime(df_trades['Date']).dt.normalize()
        
        df_units, df_cash, cost_basis = calculate_portfolio_state(df_trades, self.all_dates, self.tickers)
        
        # Check Final Units
        self.assertEqual(df_units.loc['2024-01-02', 'AAPL'], 10.0)
        self.assertEqual(df_units.loc['2024-01-10', 'AAPL'], 10.0)
        
        # Check Cash
        expected_cash = STARTING_CAPITAL - (100.0 * 10)
        self.assertAlmostEqual(df_cash.loc['2024-01-02', 'Cash_Balance'], expected_cash)
        
        # Check Cost Basis
        self.assertEqual(cost_basis['AAPL']['total_shares'], 10.0)
        self.assertEqual(cost_basis['AAPL']['avg_cost'], 100.0)

    def test_chinese_side_mapping(self):
        """Test that Chinese characters '買進' and '賣出' are correctly mapped."""
        trades_data = [
            {'Date': '2024-01-02', 'Ticker': 'AAPL', 'Side': '買進', 'Price': 100.0, 'Qty': 10, 'Fee': 0, 'Tax': 0},
            {'Date': '2024-01-03', 'Ticker': 'AAPL', 'Side': '賣出', 'Price': 110.0, 'Qty': 5, 'Fee': 0, 'Tax': 0}
        ]
        df_trades = pd.DataFrame(trades_data)
        df_trades['Date'] = pd.to_datetime(df_trades['Date']).dt.normalize()
        
        df_units, df_cash, cost_basis = calculate_portfolio_state(df_trades, self.all_dates, self.tickers)
        
        # After Buy 10, Sell 5 -> Should have 5 left
        self.assertEqual(df_units.loc['2024-01-03', 'AAPL'], 5.0)
        self.assertEqual(df_units.loc['2024-01-10', 'AAPL'], 5.0)
        
        # Cash Check
        # Start - 1000 + 550 = Start - 450
        expected_cash = STARTING_CAPITAL - 1000 + 550
        self.assertAlmostEqual(df_cash.loc['2024-01-03', 'Cash_Balance'], expected_cash)

    def test_cost_basis_weighted_avg(self):
        """Test Weighted Average Cost Basis calculation."""
        # 1. Buy 10 @ 100 = 1000
        # 2. Buy 10 @ 200 = 2000. Total Cost 3000. Total Shares 20. Avg = 150.
        # 3. Sell 5 @ 300. Shares 15. Avg Cost should REMAIN 150.
        trades_data = [
            {'Date': '2024-01-02', 'Ticker': 'AAPL', 'Side': 'Buy', 'Price': 100.0, 'Qty': 10, 'Fee': 0, 'Tax': 0},
            {'Date': '2024-01-03', 'Ticker': 'AAPL', 'Side': 'Buy', 'Price': 200.0, 'Qty': 10, 'Fee': 0, 'Tax': 0},
            {'Date': '2024-01-04', 'Ticker': 'AAPL', 'Side': 'Sell', 'Price': 300.0, 'Qty': 5, 'Fee': 0, 'Tax': 0}
        ]
        df_trades = pd.DataFrame(trades_data)
        df_trades['Date'] = pd.to_datetime(df_trades['Date']).dt.normalize()
        
        _, _, cost_basis = calculate_portfolio_state(df_trades, self.all_dates, self.tickers)
        
        self.assertEqual(cost_basis['AAPL']['total_shares'], 15.0)
        self.assertEqual(cost_basis['AAPL']['avg_cost'], 150.0) # Avg cost shouldn't change on sell
        # Total cost logic: 3000 - (5 * 150) = 3000 - 750 = 2250
        self.assertEqual(cost_basis['AAPL']['total_cost'], 2250.0)

    def test_weekend_trade_handling(self):
        """Test that trades on non-business days (weekends) are processed correctly on the timeline."""
        # 2024-01-06 is Saturday. 2024-01-07 Sunday. 2024-01-08 Monday.
        trades_data = [
            {'Date': '2024-01-06', 'Ticker': 'GOOGL', 'Side': 'Buy', 'Price': 100.0, 'Qty': 10, 'Fee': 0, 'Tax': 0}
        ]
        df_trades = pd.DataFrame(trades_data)
        df_trades['Date'] = pd.to_datetime(df_trades['Date']).dt.normalize()
        
        # all_dates is Business Days, so 01-06/07 are skipped in the index
        df_units, _, _ = calculate_portfolio_state(df_trades, self.all_dates, self.tickers)
        
        # On Monday 2024-01-08, the units should appear
        if pd.Timestamp('2024-01-08') in df_units.index:
            self.assertEqual(df_units.loc['2024-01-08', 'GOOGL'], 10.0)
        else:
            print("Warning: 2024-01-08 not in test date range, cannot verify weekend accumulation")

if __name__ == '__main__':
    unittest.main()
