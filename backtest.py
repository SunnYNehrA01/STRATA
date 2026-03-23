import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from trading_env import create_env  # Reusing your environment logic

def run_backtest(ticker):
    print(f"--- Backtesting Agent for {ticker} ---")
    
    # 1. Re-initialize Environment with the CORRECT processed data
    env = create_env(ticker) 
    
    # 2. Load the trained model
    model_path = f"models/ppo_{ticker}.zip"
    model = PPO.load(model_path)
    
    # 3. Run the simulation
    obs, info = env.reset()
    done = False
    
    while not done:
        # Use deterministic=True for the final demo/backtest
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # 4. Plot the results with proper indicator rendering
    plt.figure(figsize=(15,6))
    env.unwrapped.render_all() # This shows the Buy/Sell markers
    plt.title(f"STRATA Global Backtest: {ticker}")
    
    # Calculate and print final stats to console
    print(f"Final Portfolio Value: {env.unwrapped.total_profit}")
    plt.show()

if __name__ == "__main__":
    # You can change this to any of your 5 stocks: TSLA, AAPL, NVDA, AMD, MSFT
    run_backtest("TSLA")