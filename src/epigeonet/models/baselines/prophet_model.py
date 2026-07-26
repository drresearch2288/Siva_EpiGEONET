"""Prophet Baseline with external covariates."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from prophet import Prophet
import logging

# Suppress prophet logs
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def fit_predict_prophet(df: pd.DataFrame, horizons=[1, 2, 4]):
    """
    Fits Prophet per district with weekly_precipitation and weekly_mean_temperature as add_regressor.
    df must have columns: ds (date), y (cases), weekly_precipitation, weekly_mean_temperature.
    """
    predictions = {}
    
    try:
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        
        has_precip = 'weekly_precipitation' in df.columns
        has_temp = 'weekly_mean_temperature' in df.columns
        
        if has_precip:
            m.add_regressor('weekly_precipitation')
        if has_temp:
            m.add_regressor('weekly_mean_temperature')
            
        m.fit(df)
        
        # We need future regressors to predict. For a true forecast, we'd use forecasted weather.
        # For simplicity in the baseline, we assume we know the weather or carry forward the last.
        future = m.make_future_dataframe(periods=max(horizons), freq='W')
        
        if has_precip:
            last_precip = df['weekly_precipitation'].iloc[-1]
            future['weekly_precipitation'] = last_precip
        if has_temp:
            last_temp = df['weekly_mean_temperature'].iloc[-1]
            future['weekly_mean_temperature'] = last_temp
            
        forecast = m.predict(future)
        
        for h in horizons:
            # The forecast dataframe includes the historical dates as well
            # We want the forecasted steps at the end
            val = forecast['yhat'].iloc[-max(horizons) + h - 1]
            predictions[f'h_{h}'] = max(0, val) # No negative cases
            
    except Exception as e:
        # Fallback to seasonal naive
        for h in horizons:
            if len(df) >= 52:
                predictions[f'h_{h}'] = max(0, df['y'].iloc[-52])
            elif len(df) > 0:
                predictions[f'h_{h}'] = max(0, df['y'].iloc[-1])
            else:
                predictions[f'h_{h}'] = 0.0
                
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/master_table.parquet')
    args = parser.parse_args()
    
    print("Prophet baseline ready.")

if __name__ == '__main__':
    main()
