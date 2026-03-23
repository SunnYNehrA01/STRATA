# STRATA
### Strategic Trading via Reinforcement and Adaptive Temporal Agents

## 1. Abstract
Financial markets are characterized by high-dimensional complexity and non-stationary dynamics. Traditional linear models often fail to capture the underlying temporal patterns required for effective sequential decision-making. This project introduces STRATA, a framework that utilizes Proximal Policy Optimization (PPO) to train asset-specific reinforcement learning agents. By processing multi-modal inputs—including quantitative technical indicators and qualitative news sentiment via FinBERT—STRATA develops adaptive trading policies aimed at optimizing risk-adjusted returns.

## 2. Project Objectives
* **Environment Formalization:** Develop a custom Markov Decision Process (MDP) tailored for financial time-series data.
* **Agent Specialization:** Train independent RL models for high-volatility assets (TSLA, NVDA, AMD) and blue-chip equities (AAPL, MSFT).
* **Reward Engineering:** Implement objective functions that balance cumulative profit against maximum drawdown and volatility.
* **Sentiment Integration:** Utilize Natural Language Processing (NLP) to provide a fundamental overlay to technical signals.
* **Interactive Analytics:** Deploy a high-fidelity dashboard for real-time policy inference and historical stress testing.

## 3. Methodology

### 3.1 Data Acquisition and Preprocessing
The system retrieves historical OHLCV data via the `yfinance` API. To ensure indicator stability, a 100-day "warm-up" buffer is utilized prior to the start of any evaluation window. This prevents edge-case errors in the calculation of lag-sensitive indicators like EMA and MACD.

### 3.2 Environment Design
The trading environment is built on the Gymnasium interface. The state space is defined by a 10-day sliding window of technical features:
* **Price Momentum:** Relative Strength Index (RSI).
* **Trend Convergence:** MACD and Signal Line differentials.
* **Moving Averages:** 20-day and 50-day Exponential Moving Averages (EMA).

### 3.3 Reinforcement Learning Model
Agents are trained using the PPO algorithm, chosen for its stability in noisy environments. The model architecture consists of an Actor-Critic framework with a Multi-Layer Perceptron (MLP) policy. Training is conducted over 50,000 timesteps per asset to establish baseline strategic intent.

### 3.4 Sentiment Engine
STRATA incorporates a deep learning sentiment classifier based on the FinBERT architecture. It analyzes live financial headlines to categorize market mood as Bullish, Bearish, or Neutral, providing a confidence-weighted filter for the RL agent's technical recommendations.

## 4. Technical Stack
* **Deep Learning:** PyTorch, Transformers (HuggingFace)
* **Reinforcement Learning:** Stable Baselines3, Gymnasium
* **Quantitative Analysis:** Pandas-TA, NumPy, Pandas
* **Visualization:** Plotly, Streamlit
* **Data Source:** Yahoo Finance API

## 5. To-do List

| To do | Description | Done |
| :--- | :--- | :---: |
| Data pipeline | Implement automated yfinance downloading and MultiIndex flattening | ✅ |
| Feature Engineering | Calculate RSI, EMA_20, EMA_50, and MACD indicators | ✅ |
| Custom Environment | Define State, Action, and Reward spaces in Gymnasium | ✅ |
| PPO Training | Train 5 asset-specific agents for 50k+ timesteps | ✅ |
| Backtest Lab | Implement 2025 historical simulation with Market Baseline comparison | ✅ |
| Sentiment Engine | Integrate FinBERT model for live headline classification | ✅ |
| Interactive UI | Build bento-box style dashboard with Streamlit and Plotly | ✅ |
| Signal Confluence | Implement Master Advice logic combining RL and Sentiment | ✅ |
| Confidence Metrics | Extract and display action probability distributions | ✅ |
| Hyperparameter Tuning | Optimize PPO clipping range and entropy coefficient | ⬜ |
| Portfolio Ensemble | Develop a multi-agent system for cross-asset capital allocation | ⬜ |
| Live API Integration | Connect to brokerage API for paper trading execution | ⬜ |
| Risk Management | Integrate Kelly Criterion for dynamic position sizing | ⬜ |

## 6. Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yourusername/strata.git](https://github.com/yourusername/strata.git)
    cd strata
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Terminal:**
    ```bash
    streamlit run app.py
    ```

## 7. Conclusion
STRATA demonstrates the utility of reinforcement learning in modeling intelligent behavior within financial markets. By synthesizing technical policy optimization with real-time sentiment analysis, the framework provides a transparent and adaptive approach to sequential decision-making under uncertainty.

---
**Disclaimer:** STRATA is an experimental research project and does not constitute financial advice.