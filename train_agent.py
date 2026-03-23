import os
from stable_baselines3 import PPO
from trading_env import create_env  # Importing your Step 2 logic

# List of our target stocks
STOCKS = ["TSLA", "AAPL", "NVDA", "AMD", "MSFT"]

def train_agent(ticker):
    print(f"\n🚀 Starting Training for {ticker}...")
    
    # 1. Create the Environment
    env = create_env(ticker)
    
    # 2. Initialize the Model (The Brain)
    # MlpPolicy = Multi-layer Perceptron (Standard Neural Network)
    # verbose=1 shows us the progress in the console
    model = PPO("MlpPolicy", env, verbose=1, device="cpu") 
    
    # 3. Train the Model
    # 50,000 timesteps is a good 'prototype' length. 
    # For a production model, you'd use 1,000,000+.
    print(f"Learning {ticker} patterns...")
    model.learn(total_timesteps=50000)
    
    # 4. Save the Model
    model_path = f"models/ppo_{ticker}"
    model.save(model_path)
    print(f"✅ Model saved as {model_path}")

if __name__ == "__main__":
    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Train all agents one by one
    for stock in STOCKS:
        train_agent(stock)
        
    print("\n🎉 ALL AGENTS TRAINED AND SAVED!")