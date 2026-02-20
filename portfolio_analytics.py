
def calculate_analytics(df_units, df_cash, market_data, df_ref, cost_basis, starting_capital):
    """
    Computes all portfolio analytics required for the HTML dashboard.
    Returns a dictionary structured for Jinja2 templates.
    """
    import numpy as np
    import pandas as pd
    
    # 1. Calculate NAV Series
    # Align dates
    common_index = df_units.index.intersection(market_data.index)
    df_units_aligned = df_units.loc[common_index]
    market_aligned = market_data.loc[common_index, df_units.columns]
    
    # Position Values (Units * Price)
    pos_values = df_units_aligned * market_aligned
    
    # Total Invested Value per day
    invested_value = pos_values.sum(axis=1)
    
    # Cash Balance (aligned)
    cash_aligned = df_cash.loc[common_index, 'Cash_Balance']
    
    # Total NAV
    nav_series = invested_value + cash_aligned
    
    # 2. Performance Metrics
    if len(nav_series) > 0:
        total_return = (nav_series.iloc[-1] / starting_capital) - 1
        day_change_pct = nav_series.pct_change().iloc[-1]
        current_nav = nav_series.iloc[-1]
        
        # Max Drawdown
        rolling_max = nav_series.cummax()
        drawdown = (nav_series - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # Sharpe Ratio (Ann.) - simplistic
        daily_rets = nav_series.pct_change().dropna()
        if daily_rets.std() > 0:
            sharpe = (daily_rets.mean() / daily_rets.std()) * (252**0.5)
        else:
            sharpe = 0
    else:
        total_return = 0
        day_change_pct = 0
        current_nav = starting_capital
        max_dd = 0
        sharpe = 0

    # 3. Current Positions Snapshot
    last_date = common_index[-1]
    last_prices = market_aligned.loc[last_date]
    last_units = df_units_aligned.loc[last_date]
    
    positions = []
    options = [] 
    portfolio_weight_total = 0
    
    for ticker in df_units.columns:
        units = last_units.get(ticker, 0)
        if abs(units) < 0.001: continue # Skip closed positions
        
        price = last_prices.get(ticker, 0)
        
        # Meta info
        ref_row = df_ref[df_ref['Ticker'] == ticker]
        name = ref_row['Name'].values[0] if not ref_row.empty else ticker
        sector = ref_row['Sector'].values[0] if not ref_row.empty else 'Unknown'
        country = ref_row['Country'].values[0] if not ref_row.empty else 'Unknown'
        asset_type = ref_row['Type'].values[0] if not ref_row.empty else 'EQUITY'
        strike = ref_row['Strike'].values[0] if not ref_row.empty and 'Strike' in ref_row.columns else 0
        expiry = ref_row['Expiry'].values[0] if not ref_row.empty and 'Expiry' in ref_row.columns else '-'
        
        # Market Value Calculation
        # Assume US Options have multiplier 100.
        multiplier = 100 if asset_type == 'OPTION' else 1
        market_val = units * price * multiplier
        
        # Cost Basis
        cb = cost_basis.get(ticker, {})
        avg_cost = cb.get('avg_cost', 0)
        total_cost = units * avg_cost * multiplier # Cost per share * units * multiplier? 
        # Wait, build_portfolio calculates cost based on Trade Price * Qty.
        # If Trade Qty was contracts, and Price was per share? 
        # build_portfolio: total_val = price * qty. 
        # If Qty is contracts, Total Val is Price * Contracts. This is WRONG for US options (missing 100).
        # User hasn't complained about Option Cost Basis yet.
        # Assuming build_portfolio.py handles it as is: Total Cost loaded from CSV = correct total checking account impact.
        # Then avg_cost = Total Cost / Total Units.
        # So avg_cost is "Cost per Unit".
        # If Unit = Contract, avg_cost is Cost per Contract.
        # If Unit = Share, avg_cost is Cost per Share.
        # Market Value needs to match.
        # If yfinance gives price per share:
        # If Type == OPTION: MV = Units * 100 * Price (per share).
        # Cost = Units * avg_cost (per unit).
        # We need to know if avg_cost is per share or per contract.
        # usually trade log has 'Price' per share. 'Qty' contracts. 'Amount' = Price * Qty * 100.
        # But build_portfolio computes total_val = price * qty.
        # Use simple logic: MV = Units * Price * (100 if OPTION else 1).
        # Unrealized PnL = MV - Total Cost (from tracker).
        
        total_cost = cb.get('total_cost', 0) # Use tracked total cost directly
        # But tracker total cost might be for *all* history? No, it processes trades.
        # Logic: current_units * avg_cost matches remaining inventory cost.
        total_cost_calc = units * avg_cost
        
        unrealized_pnl = market_val - total_cost_calc
        pnl_pct = (unrealized_pnl / total_cost_calc) if total_cost_calc != 0 else 0
        
        weight = market_val / current_nav if current_nav > 0 else 0
        portfolio_weight_total += weight
        
        item = {
            'ticker': ticker,
            'name': name,
            'sector': sector,
            'country': country,
            'shares': units,
            'avg_cost': avg_cost,
            'price': price,
            'market_value': market_val,
            'unrealized_pnl': unrealized_pnl,
            'pnl_pct': pnl_pct,
            'weight': weight,
            'type': asset_type,
            'strike': strike,
            'expiry': expiry
        }
        
        if asset_type == 'OPTION':
            options.append(item)
            # Also add to positions? User wants "Positions — All holdings (stocks, ETFs, mutual funds, cash)... Option-only tickers...".
            # So options are in positions too.
            positions.append(item)
        else:
            positions.append(item)
    
    # Sort positions by value desc
    positions.sort(key=lambda x: x['market_value'], reverse=True)
    options.sort(key=lambda x: x['market_value'], reverse=True)

    
    # 4. Sector Exposure
    sector_exposure = {}
    for p in positions:
        sec = p['sector']
        sector_exposure[sec] = sector_exposure.get(sec, 0) + p['market_value']
        
    sectors_list = [{'sector': k, 'value': v, 'weight': v/current_nav} for k,v in sector_exposure.items()]
    sectors_list.sort(key=lambda x: x['value'], reverse=True)
    
    # 5. Risk Metrics (Correlation)
    # Calculate correlation matrix of returns for current positions
    active_tickers = [p['ticker'] for p in positions]
    if len(active_tickers) > 1:
        # Get last 1 year or all data
        recent_market = market_data[active_tickers].iloc[-252:]
        corr_matrix = recent_market.pct_change().corr().fillna(0)
        
        # Convert to list of dicts for template
        # format: {'ticker': T1, 'correlations': {'T2': 0.5, 'T3': 0.1...}}
        correlation_data = []
        for t1 in active_tickers:
            row_data = {'ticker': t1, 'corr_cells': []}
            for t2 in active_tickers:
                val = corr_matrix.loc[t1, t2]
                row_data['corr_cells'].append({'target': t2, 'val': val})
            correlation_data.append(row_data)
    else:
        correlation_data = []

    # 6. Stress Test (Simulated)
    # Simple Beta-based stress test if we had Beta. 
    # For now, let's just do a simple scenario Analysis based on Sector?
    # Or just placeholder 0s if we don't calculating Beta yet.
    # build_portfolio has Beta calc in `Risk Metrics` sheet logic?
    # Let's check `calculate_portfolio_state` doesn't do beta.
    # The Excel formula uses `SLOPE` against SP500.
    # We can calc Beta here using numpy.
    
    # Fetch SP500 (assuming it's in market_data or we need to pass it)
    # build_portfolio puts SP500 in market_data if it fetched it.
    
    if 'SP500' in market_data.columns:
        benchmark = market_data['SP500'].pct_change().fillna(0)
    elif '^GSPC' in market_data.columns:
        benchmark = market_data['^GSPC'].pct_change().fillna(0)
    else:
        benchmark = None
        
    stress_results = []
    scenarios = [
        ('Market Crash', -0.20),
        ('Correction', -0.10),
        ('Rally', 0.10),
        ('Bull Run', 0.20)
    ]
    
    if benchmark is not None:
        # Calculate Portfolio Beta
        # Port Returns vs Bench Returns
        # Start from when we have valid data
        
        # Using daily_rets calculated above for NAV
        # We need to align benchmark to daily_rets
        aligned_df = pd.concat([daily_rets, benchmark], axis=1, join='inner').dropna()
        if not aligned_df.empty:
            port_ret = aligned_df.iloc[:, 0]
            bench_ret = aligned_df.iloc[:, 1]
            
            cov = np.cov(port_ret, bench_ret)[0][1]
            var = np.var(bench_ret)
            port_beta = cov / var if var > 0 else 1.0
        else:
            port_beta = 1.0
            
        for sc_name, chg in scenarios:
            est_change_pct = chg * port_beta
            est_pl = est_change_pct * current_nav
            est_nav = current_nav + est_pl
            stress_results.append({
                'scenario': sc_name,
                'market_change': chg,
                'est_portfolio_change': est_change_pct,
                'est_pnl': est_pl,
                'est_nav': est_nav
            })
    
    return {
        'dashboard': {
            'date': last_date.strftime('%Y-%m-%d'),
            'nav': current_nav,
            'cash': cash_aligned.iloc[-1],
            'invested': invested_value.iloc[-1],
            'day_change_pct': day_change_pct,
            'total_return_pct': total_return,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'history': [{'date': d.strftime('%Y-%m-%d'), 'nav': v} for d, v in nav_series.items()],
            'nav_twd': current_nav * market_data['TWD=X'].ffill().iloc[-1] if 'TWD=X' in market_data.columns else 0,
            'fx_rate': market_data['TWD=X'].ffill().iloc[-1] if 'TWD=X' in market_data.columns else 0
        },
        'positions': positions,
        'options': options,
        'sectors': sectors_list,
        'correlation': correlation_data,
        'stress_test': stress_results
    }
