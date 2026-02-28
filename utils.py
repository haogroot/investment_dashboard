import os
import re
from pathlib import Path

def find_latest_files(directory):
    """
    Scans a directory for CSV/XLSX files matching the pattern {Owner}_{Market}_*_YYYYMMDD.ext
    and returns the latest file for each {Owner}_{Market} combination.

    Pattern:
      <Owner>_<Market>_<Broker>_<Date>.<ext>
      e.g., Howard_US_fubon-trade-record_20260224.csv

    Returns:
      A dictionary mapping '{Owner}_{Market}' to the path of the latest file.
      e.g., {'Howard_US': Path('...'), 'Debby_TW': Path('...')}
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        print(f"Warning: Directory {directory} not found.")
        return {}

    # This regex captures:
    # 1: Owner (e.g. Howard, Debby)
    # 2: Market (e.g. US, TW)
    # 3: Remaining part before the date
    # 4: Date (8 digits YYYYMMDD)
    # 5: Extension (.csv, .xlsx)
    pattern = re.compile(r'^([A-Za-z0-9]+)_(US|TW)_(.*?)_(\d{8})\.(csv|xlsx)$', re.IGNORECASE)

    latest_files = {} # Key: f"{Owner}_{Market}", Value: {'date': 'YYYYMMDD', 'path': Path}

    for f in directory_path.iterdir():
        if not f.is_file():
            continue
        
        match = pattern.match(f.name)
        if match:
            owner = match.group(1)
            market = match.group(2)
            date_str = match.group(4)
            
            key = f"{owner}_{market}"
            
            if key not in latest_files or date_str > latest_files[key]['date']:
                latest_files[key] = {
                    'date': date_str,
                    'path': f
                }
        else:
            # Fallback for old files like fubon-trade-record_20260216.csv
            # If we want to support them as Howard_US by default, we could do it here
            pass
            
    return {k: v['path'] for k, v in latest_files.items()}

def get_latest_us_file(directory):
    """
    Scans the directory and returns the path to the latest Howard_US file.
    Only Howard's US file is supported as requested.
    """
    latest = find_latest_files(directory)
    return latest.get('Howard_US')

def get_latest_tw_files(directory):
    """
    Scans the directory and returns a list of the latest TW files for all owners.
    """
    latest = find_latest_files(directory)
    return [path for key, path in latest.items() if key.endswith('_TW')]
