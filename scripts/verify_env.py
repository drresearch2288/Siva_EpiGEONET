"""
verify_env.py
Script to verify the EpiGeoNet environment, including MPS availability and package imports.
"""
import argparse
import sys
import importlib

def main():
    parser = argparse.ArgumentParser(description="Verify EpiGeoNet Environment on Apple Silicon.")
    args = parser.parse_args()

    print("--- PyTorch & MPS Verification ---")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        mps_available = torch.backends.mps.is_available()
        print(f"MPS Available:   {mps_available}")
        
        if mps_available:
            try:
                print("Running tensor matmul on 'mps'...")
                device = torch.device('mps')
                a = torch.randn(100, 100, device=device)
                b = torch.randn(100, 100, device=device)
                c = torch.matmul(a, b)
                print("MPS matmul: OK")
            except Exception as e:
                print(f"MPS matmul: FAIL ({e})")
        else:
            print("MPS not available, skipping matmul.")
    except ImportError:
        print("PyTorch not installed or failed to import.")
        sys.exit(1)

    print("\n--- Package Import Verification ---")
    packages_to_check = [
        "torch_geometric",
        "geopandas",
        "libpysal",
        "shap",
        "captum",
        "xgboost",
        "statsmodels",
        "prophet"
    ]

    all_ok = True
    for pkg in packages_to_check:
        try:
            importlib.import_module(pkg)
            print(f"{pkg.ljust(18)}: OK")
        except ImportError as e:
            print(f"{pkg.ljust(18)}: FAIL ({e})")
            all_ok = False
            
    print("\nVerification Complete.")
    if not all_ok:
        print("Warning: Some packages failed to import.")
        sys.exit(1)

if __name__ == '__main__':
    main()
