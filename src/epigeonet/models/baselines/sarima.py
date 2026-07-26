"""Classical SARIMA Baseline."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_predict_sarima(series: pd.Series, horizons=[1, 2, 4]):
    """
    Fits SARIMA per district. Falls back to seasonal naive if fit fails.
    """
    order = (1, 1, 1)
    seasonal_order = (1, 0, 0, 52) # Weekly data, yearly seasonality
    
    predictions = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order, 
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            
        max_horizon = max(horizons)
        forecast = results.forecast(steps=max_horizon)
        
        for h in horizons:
            predictions[f'h_{h}'] = max(0, forecast.iloc[h-1]) # No negative cases
            
    except Exception as e:
        # Fallback to seasonal naive (last year's value)
        for h in horizons:
            if len(series) >= 52:
                predictions[f'h_{h}'] = max(0, series.iloc[-52])
            elif len(series) > 0:
                predictions[f'h_{h}'] = max(0, series.iloc[-1])
            else:
                predictions[f'h_{h}'] = 0.0
                
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/master_table.parquet')
    args = parser.parse_args()
    
    print("SARIMA baseline ready.")

if __name__ == '__main__':
    main()
