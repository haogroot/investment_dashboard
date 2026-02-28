import pandas as pd
import numpy as np
import math
from pathlib import Path
import os

def extract_asset_details(xl):
    """
    Extracts granular details for each liquid asset segment from '01資產負債詳情'.
    Returns a dict with a list of dictionaries for each category.
    """
    details = {
        'emergency_fund': [],
        'demand_deposit': [],
        'fx_cash': [],
        'fx_deposit': []
    }
    
    if '01資產負債詳情' not in xl.sheet_names:
        return details
        
    df = xl.parse('01資產負債詳情', header=None)
    
    category_map = {
        '備用金': ('emergency_fund', 1, 9),
        '活期存款': ('demand_deposit', 1, 9),
        '外幣現金': ('fx_cash', 1, 9),
        '外幣存款': ('fx_deposit', 1, 9)
    }
    
    current_category = None
    
    for idx, row in df.iterrows():
        cell_0 = str(row[0]).strip() if pd.notna(row[0]) else ""
        
        # Check if this row initiates a category
        if cell_0 in category_map:
            current_category = category_map[cell_0]
            continue
            
        if current_category:
            cat_key, name_col, amt_col = current_category
            
            # Check if we should stop parsing this category
            if pd.isna(row[0]) and pd.isna(row[name_col]):
                current_category = None
                continue
                
            if str(row[0]).strip() == '#':
                continue # header row
                
            name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
            try:
                amt_str = str(row[amt_col]).replace(',', '').strip()
                amt = float(amt_str)
                # Only keep valid positive records (sometimes there are empty records, or 0 balances)
                if name and not math.isnan(amt) and amt > 0:
                    details[cat_key].append({'name': name, 'amount': amt})
            except ValueError:
                pass

    return details

def load_property_data(file_path):
    """
    Reads the given property Excel file and extracts the latest asset/liability values.
    Returns a dictionary of extracted values or None if parsing fails.
    """
    print(f"Loading property data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Property file not found: {file_path}")
        return None

    try:
        xl = pd.ExcelFile(file_path)
        
        # Test all sheets to find the one with the correct keys
        target_keys = {
            '備用金': 0,
            '活期存款': 0,
            '外幣現金': 0,
            '外幣存款': 0,
            '不動產現值': 0,
            '負債(數值前請加-)': 0,
        }
        
        target_sheet = None
        row_indices = {}
        df = None
        
        # Hardcode '02資產負債彙總' as the primary target if it exists, as it contains the raw data
        primary_sheet = '02資產負債彙總'
        
        if primary_sheet in xl.sheet_names:
            target_sheet = primary_sheet
            df = xl.parse(primary_sheet, header=None)
            temp_row_indices = {k: [] for k in target_keys}
            for idx, row in df.iterrows():
                for col_idx in range(min(5, len(row))):
                    cell_val = str(row[col_idx]).strip() if pd.notna(row[col_idx]) else ""
                    if cell_val in target_keys:
                        temp_row_indices[cell_val].append(idx)
                        break
            row_indices_lists = temp_row_indices
        else:
            # Fallback detection
            for sheet in xl.sheet_names:
                temp_df = xl.parse(sheet, header=None)
                temp_row_indices = {k: [] for k in target_keys}
                for idx, row in temp_df.iterrows():
                    for col_idx in range(min(5, len(row))):
                        cell_val = str(row[col_idx]).strip() if pd.notna(row[col_idx]) else ""
                        if cell_val in target_keys:
                            temp_row_indices[cell_val].append(idx)
                            break
                
                # If we found at least 3 of our target keys, this is the right sheet
                # Count keys that have at least one index
                valid_keys = sum(1 for indices in temp_row_indices.values() if indices)
                if valid_keys >= 3:
                    target_sheet = sheet
                    row_indices_lists = temp_row_indices
                    df = temp_df
                    break

        if not target_sheet:
            print("Could not find a sheet containing the required property keys.")
            return None

        print(f"Found required keys in sheet: {target_sheet}")

        # Data rows are found dynamically based on the first cell in the row
        # Since the sheet has multiple years stacked vertically (e.g. 2026 at top, 2025 below),
        # there are multiple rows for '備用金'.
        # We want to collect all rows for each key, maintaining their vertical order.
        row_indices_lists = {k: [] for k in target_keys}
        for idx, row in df.iterrows():
            for col_idx in range(min(5, len(row))):
                cell_val = str(row[col_idx]).strip() if pd.notna(row[col_idx]) else ""
                if cell_val in target_keys:
                    row_indices_lists[cell_val].append(idx)
                    break
                    
        # Now, for each key, we want to find the latest valid numeric value.
        # The sheet is ordered with the most recent year at the top.
        # Within a year (a row), months go from left (Jan) to right (Dec).
        # So the "latest" value is the last valid value in the topmost row that contains valid data.
        results = {}
        for key, indices in row_indices_lists.items():
            found_latest = False
            for row_idx in indices:
                row_data = df.iloc[row_idx].values
                valid_values = []
                for val in row_data:
                    if pd.notna(val) and val != '':
                        try:
                            clean_val = str(val).replace(',', '').strip()
                            if clean_val in ['金額 (台幣)', '金額(台幣)', key] or '現值' in clean_val or '存款' in clean_val or '備用金' in clean_val or '負債' in clean_val:
                                continue
                            num_val = float(clean_val)
                            if not math.isnan(num_val):
                                 valid_values.append(num_val)
                        except ValueError:
                            pass
                
                if valid_values:
                    # Topmost row with data -> take the last value (most recent month in that year)
                    results[key] = valid_values[-1]
                    found_latest = True
                    break # Stop searching older years
                    
            if not found_latest:
                results[key] = 0.0
                
        # Clean up the liability key for easier access
        clean_results = {
            'emergency_fund': results.get('備用金', 0.0),
            'demand_deposit': results.get('活期存款', 0.0),
            'fx_cash': results.get('外幣現金', 0.0),
            'fx_deposit': results.get('外幣存款', 0.0),
            'real_estate': results.get('不動產現值', 0.0),
            'liabilities': results.get('負債(數值前請加-)', 0.0)
        }
        
        # Calculate liquid funds
        clean_results['liquid_funds'] = sum([
            clean_results['emergency_fund'],
            clean_results['demand_deposit'],
            clean_results['fx_cash'],
            clean_results['fx_deposit']
        ])
        
        # Extract individual bank/asset details
        clean_results['details'] = extract_asset_details(xl)
        
        print(f"Successfully extracted property data: {clean_results}")
        return clean_results

    except Exception as e:
        print(f"Error parsing property file: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        load_property_data(sys.argv[1])
    else:
        # Default test
        test_path = Path("property/Howard_202601.xlsx")
        load_property_data(test_path)
