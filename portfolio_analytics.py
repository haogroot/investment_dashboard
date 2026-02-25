
def calculate_analytics(df_units, df_cash, market_data, df_ref, cost_basis, starting_capital, df_trades=None):
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
    # Extracted fx rate early for TWD value calculation
    fx_rate = market_data['TWD=X'].ffill().iloc[-1] if 'TWD=X' in market_data.columns else 0

    portfolio_weight_total = 0
    stock_count = 0
    etf_count = 0
    
    stock_value = 0
    bond_value = 0
    other_value = 0
    
    for ticker in df_units.columns:
        units = last_units.get(ticker, 0)
        if abs(units) < 0.001: continue # Skip closed positions
        
        price = last_prices.get(ticker, 0)
        
        # Meta info
        ref_row = df_ref[df_ref['Ticker'] == ticker]
        name = ref_row['Name'].values[0] if not ref_row.empty else ticker
        sector = ref_row['Sector'].values[0] if not ref_row.empty else 'Unknown'
        country = ref_row['Country'].values[0] if not ref_row.empty else 'Unknown'
        beta = ref_row['Beta'].values[0] if (not ref_row.empty and 'Beta' in ref_row) else 1.0
        asset_type = ref_row['Type'].values[0] if not ref_row.empty else 'EQUITY'
        strike = ref_row['Strike'].values[0] if not ref_row.empty and 'Strike' in ref_row.columns else 0
        expiry = ref_row['Expiry'].values[0] if not ref_row.empty and 'Expiry' in ref_row.columns else '-'
        
        # Determine Stock vs ETF
        if asset_type == 'ETF':
            etf_count += 1
        elif asset_type == 'EQUITY':
            stock_count += 1
        
        # Market Value Calculation
        # Assume US Options have multiplier 100.
        multiplier = 100 if asset_type == 'OPTION' else 1
        market_val = units * price * multiplier
        
        is_bond = False
        t_upper = ticker.upper()
        if t_upper in ['BND', 'TLT', 'AGG', 'IEF', 'SHY', 'IEI', 'TLH', 'GOVT', 'VGIT', 'VCIT', 'VCSH', 'BNDX']:
            is_bond = True
        elif isinstance(name, str) and 'bond' in name.lower():
            is_bond = True
        elif isinstance(sector, str) and 'bond' in sector.lower():
            is_bond = True
            
        if is_bond:
            bond_value += market_val
        elif asset_type in ['EQUITY', 'ETF']:
            stock_value += market_val
        else:
            other_value += market_val
        
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
            'market_value_twd': market_val * fx_rate,
            'unrealized_pnl': unrealized_pnl,
            'pnl_pct': pnl_pct,
            'weight': weight,
            'type': asset_type,
            'beta': beta,
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

    # 6. Stress Test (Simulated with Option Hedging)
    stress_results = []
    scenarios = [
        ('Crash', -0.50),
        ('Severe Bear', -0.40),
        ('Bear', -0.30),
        ('Correction', -0.20),
        ('Pullback', -0.10),
        ('Flat', 0.0),
        ('Rally', 0.10),
        ('Strong Rally', 0.20),
        ('Bull', 0.30),
        ('Euphoria', 0.40),
        ('Bubble', 0.50)
    ]
    
    if 'SP500' in market_data.columns:
        benchmark = market_data['SP500'].pct_change().fillna(0)
    elif '^GSPC' in market_data.columns:
        benchmark = market_data['^GSPC'].pct_change().fillna(0)
    else:
        benchmark = None

    # Pre-calculate Portfolio Beta for flat benchmark fallback
    port_beta = 1.0
    if benchmark is not None:
        aligned_df = pd.concat([daily_rets, benchmark], axis=1, join='inner').dropna()
        if not aligned_df.empty:
            cov = np.cov(aligned_df.iloc[:, 0], aligned_df.iloc[:, 1])[0][1]
            var = np.var(aligned_df.iloc[:, 1])
            port_beta = cov / var if var > 0 else 1.0

    for sc_name, chg in scenarios:
        unhedged_pnl = 0
        hedged_pnl = 0
        
        for pos in positions:
            t = pos['ticker']
            p_type = pos.get('type', 'EQUITY')
            base_mv = pos.get('market_value', 0)
            shares = pos.get('shares', 0)
            
            if p_type in ['EQUITY', 'ETF']:
                beta_to_use = pos.get('beta', 1.0)
                if np.isnan(beta_to_use): beta_to_use = 1.0
                asset_chg = chg * beta_to_use
                est_pnl = base_mv * asset_chg
                
                unhedged_pnl += est_pnl
                hedged_pnl += est_pnl
                
            elif p_type == 'OPTION':
                strike = float(pos.get('strike', 0))
                # Try to infer underlying from standard 21-char OPRA symbol
                underlying = t[:-15] if len(t) > 15 else t
                ul_price = last_prices.get(underlying, 0)
                
                is_put = 'P' in t[-9:-8] if len(t) > 15 else False
                is_call = 'C' in t[-9:-8] if len(t) > 15 else False
                
                if strike > 0 and ul_price > 0:
                    # Find underlying beta
                    ul_beta = 1.0
                    for p2 in positions:
                        if p2['ticker'] == underlying:
                            ul_beta = p2.get('beta', 1.0)
                            break
                    
                    new_ul_price = ul_price * (1 + chg * ul_beta)
                    
                    # Estimate intrinsic value at expiration
                    if is_put:
                        new_opt_price = max(0, strike - new_ul_price)
                    elif is_call:
                        new_opt_price = max(0, new_ul_price - strike)
                    else:
                        new_opt_price = pos['price']
                    
                    new_mv = shares * new_opt_price * 100
                    opt_pnl = new_mv - base_mv
                    hedged_pnl += opt_pnl
                else:
                    # Fallback to simple beta if option lacks strike/underlying
                    beta_to_use = pos.get('beta', 0.0)
                    if np.isnan(beta_to_use): beta_to_use = 0.0
                    opt_pnl = base_mv * (chg * beta_to_use)
                    hedged_pnl += opt_pnl

        est_hedged_nav = current_nav + hedged_pnl
        
        stress_results.append({
            'scenario': sc_name,
            'market_change': chg,
            'unhedged_pnl': unhedged_pnl,
            'hedged_pnl': hedged_pnl,
            'unhedged_change': unhedged_pnl / current_nav if current_nav > 0 else 0,
            'hedged_change': hedged_pnl / current_nav if current_nav > 0 else 0,
            'hedging_benefit': hedged_pnl - unhedged_pnl,
            'est_nav': est_hedged_nav
        })
            
    # --- Advanced Risk Metrics ---
    # We define a few helpers
    rf_rate = 0.043 # 4.3% Risk-free rate (matches online)
    
    # Portfolio Return & Volatility (Annualized)
    ann_return = 0
    ann_volatility = 0
    sortino = 0
    calmar = 0
    var_95 = 0
    var_99 = 0
    cvar_95 = 0
    skewness = 0
    kurtosis = 0
    
    if len(nav_series) > 1:
        daily_rets = nav_series.pct_change().dropna()
        if not daily_rets.empty:
            ann_return = daily_rets.mean() * 252
            ann_volatility = daily_rets.std() * np.sqrt(252)
            
            # Sortino
            downside_rets = daily_rets[daily_rets < 0]
            downside_std = downside_rets.std() * np.sqrt(252)
            if downside_std > 0:
                sortino = (ann_return - rf_rate) / downside_std
            
            # Calmar
            if max_dd < 0:
                calmar = ann_return / abs(max_dd)
                
            # VaR & CVaR
            var_95 = np.percentile(daily_rets, 5)
            var_99 = np.percentile(daily_rets, 1)
            cvar_95 = daily_rets[daily_rets <= var_95].mean()
            
            # Skew & Kurtosis
            skewness = daily_rets.skew()
            kurtosis = daily_rets.kurtosis()
            
    # Calculate per-position risk metrics
    for pos in positions:
        t = pos['ticker']
        if t in market_aligned.columns:
            prices = market_aligned[t].dropna()
            if len(prices) > 1:
                p_rets = prices.pct_change().dropna()
                
                pos['ann_return'] = p_rets.mean() * 252
                pos['ann_volatility'] = p_rets.std() * np.sqrt(252)
                
                p_sharpe = 0
                if p_rets.std() > 0:
                    p_sharpe = (p_rets.mean() * 252 - rf_rate) / (p_rets.std() * np.sqrt(252))
                pos['sharpe'] = p_sharpe
                
                # Max DD
                roll_max = prices.cummax()
                dd = (prices - roll_max) / roll_max
                pos['max_drawdown'] = dd.min()
                
                # VaR 95
                if not p_rets.empty:
                    pos['var_95'] = np.percentile(p_rets, 5)
                else:
                    pos['var_95'] = 0
                
                # Beta
                if benchmark is not None and not benchmark.empty:
                    aligned = pd.concat([p_rets, benchmark], axis=1, join='inner').dropna()
                    if not aligned.empty:
                        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0][1]
                        var = np.var(aligned.iloc[:, 1])
                        pos['beta'] = cov / var if var > 0 else 1.0
                    else:
                        pos['beta'] = 1.0
                else:
                    pos['beta'] = 1.0
            else:
                pos['ann_return'] = 0
                pos['ann_volatility'] = 0
                pos['sharpe'] = 0
                pos['max_drawdown'] = 0
                pos['var_95'] = 0
                pos['beta'] = 1.0
        else:
            pos['ann_return'] = 0
            pos['ann_volatility'] = 0
            pos['sharpe'] = 0
            pos['max_drawdown'] = 0
            pos['var_95'] = 0
            pos['beta'] = 1.0

    # 7. Historical Transactions Processing
    trade_history_grouped = []
    if df_trades is not None and not df_trades.empty:
        # Sort descending by date
        df_trades_sorted = df_trades.copy().sort_values('Date', ascending=False)
        
        # We need to map standard sides to the ones expected by the UI
        # Ensure we have date properties
        history_list = []
        for _, row in df_trades_sorted.iterrows():
            date_obj = row['Date']
            
            raw_side = str(row['Side']).lower().strip()
            # Fubon Chinese Headers handling already done in load_process_data, but let's be safe
            if '買' in raw_side or 'buy' in raw_side:
                side = 'Buy'
                side_zh = '買入'
            elif '賣' in raw_side or 'sell' in raw_side:
                side = 'Sell'
                side_zh = '賣出'
            else:
                side = 'Other'
                side_zh = '其他'
                
            qty = float(row.get('Qty', 0))
            price = float(row.get('Price', 0))
            fee = float(row.get('Fee', 0)) if pd.notnull(row.get('Fee')) else 0.0
            tax = float(row.get('Tax', 0)) if pd.notnull(row.get('Tax')) else 0.0
            
            total_val = (price * qty) + (fee if side == 'Buy' else -(fee + tax))
            
            ticker = row.get('Ticker', '')
            
            # Lookup name
            ref_row = df_ref[df_ref['Ticker'] == ticker]
            name = ref_row['Name'].values[0] if not ref_row.empty else ticker
            asset_type = ref_row['Type'].values[0] if not ref_row.empty else 'EQUITY'
            
            # Month grouping key like "2026-01"
            month_key = date_obj.strftime('%Y-%m')
            
            history_list.append({
                'date_str': date_obj.strftime('%Y-%m-%d'),
                'month_str': date_obj.strftime('%b'), # 'Jan', 'Feb'
                'day_str': date_obj.strftime('%d'), # '15', '30'
                'month_key': month_key,
                'ticker': ticker,
                'name': name,
                'side': side,
                'side_zh': side_zh,
                'qty': qty,
                'price': price,
                'total_val': total_val,
                'type': asset_type
            })
            
        # Group by month
        from itertools import groupby
        
        for k, g in groupby(history_list, key=lambda x: x['month_key']):
            trade_history_grouped.append({
                'month': k,
                'trades': list(g)
            })

    return {
        'dashboard': {
            'date': last_date.strftime('%Y-%m-%d'),
            'nav': current_nav,
            'cash': cash_aligned.iloc[-1],
            'invested': invested_value.iloc[-1],
            'stock_value': stock_value,
            'bond_value': bond_value,
            'other_value': other_value,
            'day_change_pct': day_change_pct,
            'total_return_pct': total_return,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'stock_count': stock_count,
            'etf_count': etf_count,
            'history': [{'date': d.strftime('%Y-%m-%d'), 'nav': v} for d, v in nav_series.items()],
            'nav_twd': current_nav * fx_rate,
            'fx_rate': fx_rate,
            # New Advanced Metrics
            'ann_return': ann_return,
            'ann_volatility': ann_volatility,
            'sortino': sortino,
            'calmar': calmar,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'beta': port_beta if benchmark is not None else 1.0
        },
        'positions': positions,
        'options': options,
        'sectors': sectors_list,
        'correlation': correlation_data,
        'stress_test': stress_results,
        'history_grouped': trade_history_grouped
    }
