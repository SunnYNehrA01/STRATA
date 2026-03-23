import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os

# Create folder for data
if not os.path.exists('data'):
    os.makedirs('data')

def get_and_clean_data(ticker):
    print(f"--- Processing {ticker} ---")
    
    # 1. Download Data
    df = yf.download(ticker, start="2016-01-01", end="2026-03-20", auto_adjust=True)
    
    if df.empty:
        print(f"Error: No data found for {ticker}")
        return

    # --- FIX: Flatten MultiIndex Columns ---
    # If columns are tuples like ('Close', 'TSLA'), keep only the first part ('Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Standardize column names to lowercase strings
    df.columns = [str(col).replace(" ", "_").lower() for col in df.columns]
    # ----------------------------------------

    # 2. Add Technical Indicators
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ema_20'] = ta.ema(df['close'], length=20)
    df['ema_50'] = ta.ema(df['close'], length=50)
    
    # ATR for volatility
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # MACD
    macd = ta.macd(df['close'])
    df = pd.concat([df, macd], axis=1)

    # 3. Final Cleaning
    df.dropna(inplace=True)
    
    # Save to CSV
    file_path = f"data/{ticker}_processed.csv"
    df.to_csv(file_path)
    print(f"Saved {ticker} data to {file_path}. Shape: {df.shape}")

# List of assets
stocks = ["TSLA", "AAPL", "NVDA", "AMD", "MSFT"]

for stock in stocks:
    get_and_clean_data(stock)

print("\nStep 1 FIXED and Complete.")