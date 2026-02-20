import os

files_to_delete = [
    "build_portfolio_fixed.py",
    "generate_html_fixed.py",
    "clean_files.py",
    "diagnose_env.py",
    "test_sync.py",
    "investment_portfolio_dashboard"
]

print("Cleaning up temporary files...")
for file in files_to_delete:
    if os.path.exists(file):
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    else:
        print(f"Skipped (not found): {file}")

print("Cleanup complete.")

# Self-destruct
try:
    os.remove(__file__)
    print(f"Deleted script: {os.path.basename(__file__)}")
except Exception as e:
    print(f"Could not delete self: {e}")
