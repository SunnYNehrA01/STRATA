import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
import pandas as pd
import numpy as np

# 1. Smarter Data Processing
def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    
    # Get the prices (low) for the environment
    # We use .iloc to avoid index name issues
    prices = env.df.loc[:, 'low'].to_numpy()[start:end]
    
    # --- AUTO-DETECT FEATURES ---
    # This finds whatever your MACD and RSI columns are actually named
    all_cols = env.df.columns.tolist()
    
    # We want: close, rsi, ema_20, ema_50, and the main MACD line
    # Usually the main MACD line is the one starting with 'macd' and ending in '9'
    rsi_col = [c for c in all_cols if 'rsi' in c.lower()][0]
    ema20_col = [c for c in all_cols if 'ema_20' in c.lower()][0]
    ema50_col = [c for c in all_cols if 'ema_50' in c.lower()][0]
    # Pick the first MACD column (the main line)
    macd_col = [c for c in all_cols if 'macd' in c.lower() and 'h' not in c.lower() and 's' not in c.lower()][0]
    
    features = ['close', rsi_col, ema20_col, ema50_col, macd_col]
    print(f"Using features: {features}")
    
    signal_features = env.df.loc[:, features].to_numpy()[start:end]
    return prices, signal_features

class CustomStocksEnv(StocksEnv):
    _process_data = my_process_data

def create_env(ticker, window_size=10):
    # Load and fix data
    df = pd.read_csv(f'data/{ticker}_processed.csv')
    
    # Ensure column names are clean strings
    df.columns = [str(col).lower() for col in df.columns]
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    env = CustomStocksEnv(df=df, window_size=window_size, frame_bound=(window_size, len(df)))
    return env

if __name__ == "__main__":
    try:
        test_ticker = "TSLA"
        env = create_env(test_ticker)
        print(f"Successfully created environment for {test_ticker}")
        state, info = env.reset()
        print("Observation shape:", state.shape)
    except Exception as e:
        print(f"Error: {e}")
        # If it fails, let's see what the columns actually are
        df_check = pd.read_csv(f'data/TSLA_processed.csv')
        print("Your CSV columns are actually:", df_check.columns.tolist())