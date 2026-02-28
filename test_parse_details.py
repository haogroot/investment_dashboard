import pandas as pd
import numpy as np

def extract_details(file_path):
    xl = pd.ExcelFile(file_path)
    if '01資產負債詳情' not in xl.sheet_names:
        return {}
        
    df = xl.parse('01資產負債詳情', header=None)
    
    details = {
        'emergency_fund': [],
        'demand_deposit': [],
        'fx_cash': [],
        'fx_deposit': []
    }
    
    # Mapping translated keys to their string in the sheet
    category_map = {
        '備用金': ('emergency_fund', 2, 8), # name is col 2, amount is col 8
        '活期存款': ('demand_deposit', 1, 8), # bank is col 1, amount is col 8
        '外幣現金': ('fx_cash', 1, 8), # bank/location is col 1, amount is col 8 (actually 台幣現值)
        '外幣存款': ('fx_deposit', 1, 9) # bank is col 1, 台幣現值 is col 9
    }
    
    current_category = None
    
    for idx, row in df.iterrows():
        cell_0 = str(row[0]).strip() if pd.notna(row[0]) else ""
        
        # Check if this row is a category header
        if cell_0 in category_map:
            current_category = category_map[cell_0]
            continue
            
        if current_category:
            cat_key, name_col, amt_col = current_category
            
            # Stop if we hit a total row or empty #
            if pd.isna(row[0]) or str(row[0]).strip() == '' or str(row[0]).strip() == '#':
                # wait, '#' is on the next line usually
                # Let's check cell_0
                if cell_0 == '#':
                    continue # header row
                
                # Check if it's mostly nan, or total row
                if pd.isna(row[0]):
                    # Maybe it's empty, maybe it's the total row. If it's empty in name_col, we stop
                    if pd.isna(row[name_col]):
                        current_category = None
                        continue
            
            name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
            try:
                amt = float(str(row[amt_col]).replace(',', '').strip())
                if name and not np.isnan(amt) and amt != 0:
                    details[cat_key].append({'name': name, 'amount': amt})
            except:
                pass

    return details

print(extract_details('property/Howard_202601.xlsx'))
