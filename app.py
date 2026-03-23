import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from stable_baselines3 import PPO
import torch
import numpy as np
from sentiment_engine import analyze_news_impact

# --- 1. Page Config & Professional CSS ---
st.set_page_config(page_title="STRATA", layout="wide", page_icon="")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .sticky-header { position: sticky; top: 0; z-index: 999; }
    .bento-card {
        background: #1C1F26;
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #FFFFFF; }
    .stMetric { background: #1C1F26; padding: 15px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class='bento-card sticky-header' style='margin-bottom: 20px;'>
        <h1 style='text-align:center; margin: 0; font-weight: 800;'>Strategic Trading via Reinforcement and Adaptive Temporal Agents</h1>
    </div>
""", unsafe_allow_html=True)

# --- 2. Logic Components ---
@st.cache_resource
def load_model(ticker):
    return PPO.load(f"models/ppo_{ticker}")

def get_features(df):
    macd_col = [c for c in df.columns if 'macd' in c.lower() and 'h' not in c.lower() and 's' not in c.lower()][0]
    return df[['close', 'rsi', 'ema_20', 'ema_50', macd_col]]

def get_confidence(model, obs):
    obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs[0]
    return probs

# --- 3. Header & Navigation ---
head_col1, head_col2 = st.columns([2, 1])
with head_col1:
    st.title("STRATA")
    nav = st.segmented_control(
        label="Navigation", 
        options=["Live Recommendation", "Deep Sentiment", "Backtest Lab"], 
        selection_mode="single", default="Live Recommendation", label_visibility="collapsed"
    )

with head_col2:
    ticker = st.sidebar.selectbox("Active Asset", ["TSLA", "AAPL", "NVDA", "AMD", "MSFT"])

st.divider()

# --- 4. Content Tabs ---

if nav == "Live Recommendation":
    st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
    st.subheader(f"Live Analysis: {ticker}")
    col_chart, col_sig = st.columns([2, 1])
    
    df_live = yf.download(ticker, period="120d", auto_adjust=True)
    if isinstance(df_live.columns, pd.MultiIndex): df_live.columns = df_live.columns.get_level_values(0)
    df_live.columns = [str(col).lower() for col in df_live.columns]
    df_live['rsi'] = ta.rsi(df_live['close'], length=14)
    df_live['ema_20'] = ta.ema(df_live['close'], length=20)
    df_live['ema_50'] = ta.ema(df_live['close'], length=50)
    df_live = pd.concat([df_live, ta.macd(df_live['close'])], axis=1).dropna()

    with col_chart:
        with st.container(border=True):
            fig = go.Figure(data=[go.Candlestick(x=df_live.index, open=df_live['open'], high=df_live['high'], low=df_live['low'], close=df_live['close'])])
            fig.add_trace(go.Scatter(x=df_live.index, y=df_live['ema_20'], line=dict(color='cyan', width=1), name="EMA 20"))
            fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    with col_sig:
        model = load_model(ticker)
        last_obs = get_features(df_live).tail(10).to_numpy()
        action, _ = model.predict(last_obs, deterministic=True)
        conf = get_confidence(model, last_obs)[action].item() * 100
        
        color = "#00ff00" if action == 1 else "#ff0000"
        st.markdown(f"""<div style='background:#1C1F26; border:2px solid {color}; padding:25px; border-radius:20px; text-align:center;'>
                    <h2 style='margin:0;'>{'BUY' if action == 1 else 'SELL'}</h2>
                    <p style='color:{color}; font-size:24px; font-weight:bold;'>{conf:.1f}% Confidence</p></div>""", unsafe_allow_html=True)
        st.write("")
        st.metric("Latest Close", f"${df_live['close'].iloc[-1]:.2f}", delta=f"{df_live['close'].iloc[-1] - df_live['close'].iloc[-2]:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

elif nav == "Deep Sentiment":
    st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
    st.subheader(f"News Sentiment: {ticker}")
    with st.spinner('Analyzing Global News Headlines...'):
        label, score, news_data = analyze_news_impact(ticker)
    
    s_col1, s_col2 = st.columns([1, 2])
    with s_col1:
        with st.container(border=True):
            s_color = "#00ff00" if label == "BULLISH" else "#ff0000" if label == "BEARISH" else "#cccccc"
            st.markdown(f"<h1 style='color:{s_color}; text-align:center;'>{label}</h1>", unsafe_allow_html=True)
            st.progress(score)
            st.write(f"Aggregate Score: {score:.2f}")

    with s_col2:
        with st.container(border=True):
            st.markdown("### High Impact Headlines")
            for item in news_data[:3]:
                st.markdown(f"**{item['headline']}** ([Read Article]({item['link']}))")
                st.caption(f"Sentiment: {item['sentiment'].upper()} | Confidence: {item['confidence']:.1%}")
                st.write("---")

    st.markdown("</div>", unsafe_allow_html=True)

elif nav == "Backtest Lab":
    st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
    st.markdown("""
    ### What is a Historical Comparison?
    This section is a **'Backtest'**. We take the AI agent and drop it into a specific 100-day window from the year **2025**.
    The goal is to see if the AI's logic could actually beat a simple **'Buy and Hold'** strategy during past market volatility.
    It helps us verify that the agent isn't just lucky, but has learned robust trading patterns.
    """)
    
    df_2025 = yf.download(ticker, start="2025-01-01", end="2025-12-31", auto_adjust=True)
    if isinstance(df_2025.columns, pd.MultiIndex): df_2025.columns = df_2025.columns.get_level_values(0)
    df_2025.columns = [str(col).lower() for col in df_2025.columns]
    df_2025['rsi'] = ta.rsi(df_2025['close'], length=14); df_2025['ema_20'] = ta.ema(df_2025['close'], length=20); df_2025['ema_50'] = ta.ema(df_2025['close'], length=50)
    df_2025 = pd.concat([df_2025, ta.macd(df_2025['close'])], axis=1).dropna()
    
    start_pt = st.slider("Select 2025 Window Start", 0, len(df_2025)-100, 0)
    w_df = df_2025.iloc[start_pt : start_pt + 100]
    
    # Restoring simulation logic
    balance = 10000.0; shares = 0; history = []
    model = load_model(ticker)
    for i in range(10, len(w_df)):
        obs = get_features(w_df.iloc[i-10:i]).to_numpy()
        act, _ = model.predict(obs, deterministic=True)
        p = w_df['close'].iloc[i]
        if act == 1 and balance > 0: shares = balance / p; balance = 0
        elif act == 0 and shares > 0: balance = shares * p; shares = 0
        history.append(balance + (shares * p))

    agent_ret = ((history[-1] - 10000) / 10000) * 100
    mkt_ret = ((w_df['close'].iloc[-1] - w_df['close'].iloc[0]) / w_df['close'].iloc[0]) * 100

    m1, m2, m3 = st.columns(3)
    with m1: st.container(border=True).metric("STRATA Return", f"{agent_ret:.2f}%")
    with m2: st.container(border=True).metric("Market Baseline", f"{mkt_ret:.2f}%")
    with m3: st.container(border=True).metric("Final Value", f"${history[-1]:,.0f}")

    with st.container(border=True):
        fig_h = go.Figure()
        # RESTORED: AI Line
        fig_h.add_trace(go.Scatter(x=w_df.index[10:], y=history, name="STRATA AI", line=dict(color="#00ff00", width=3)))
        # RESTORED: Market Baseline Line (Normalized to 10k)
        mkt_norm = (w_df['close'] / w_df['close'].iloc[0]) * 10000
        fig_h.add_trace(go.Scatter(x=w_df.index, y=mkt_norm, name="Market Baseline", line=dict(color="white", dash="dot")))
        fig_h.update_layout(template="plotly_dark", height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif nav == "Backtest Lab":
    st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
    st.markdown("""
    ### What is a Historical Comparison?
    This section is a **'Backtest'**. We take the AI agent and drop it into a specific 100-day window from the year **2025**.
    The goal is to see if the AI's logic could actually beat a simple **'Buy and Hold'** strategy during past market volatility.
    It helps us verify that the agent isn't just lucky, but has learned robust trading patterns.
    """)
    st.info("STRATA Logic: Evaluating AI-driven temporal decisions vs. Market Passive Baseline.")

    # 1. Fetch Warm-up Data
    df_2025_full = yf.download(ticker, start="2024-10-01", end="2025-12-31", auto_adjust=True)
    if isinstance(df_2025_full.columns, pd.MultiIndex): df_2025_full.columns = df_2025_full.columns.get_level_values(0)
    df_2025_full.columns = [str(col).lower() for col in df_2025_full.columns]
    
    df_2025_full['rsi'] = ta.rsi(df_2025_full['close'], length=14)
    df_2025_full['ema_20'] = ta.ema(df_2025_full['close'], length=20)
    df_2025_full['ema_50'] = ta.ema(df_2025_full['close'], length=50)
    df_2025_full = pd.concat([df_2025_full, ta.macd(df_2025_full['close'])], axis=1).dropna()
    
    df_2025 = df_2025_full.loc['2025-01-01':'2025-12-31']
    start_pt = st.slider("Slide Window across 2025", 0, len(df_2025)-100, 0)
    test_df = df_2025.iloc[start_pt : start_pt + 100].copy()
    
    # 2. Run Simulation using the ACTUAL Environment Logic
    # This is much more accurate than a manual loop
    from trading_env import CustomStocksEnv
    
    # Create a temporary environment for the backtest window
    env = CustomStocksEnv(df=test_df, window_size=10, frame_bound=(10, len(test_df)))
    model = load_model(ticker)
    
    obs, info = env.reset()
    done = False
    history = []
    actions = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Track the portfolio value calculated by the environment
        history.append(info['total_profit'] * 10000) # Start with 10k baseline
        actions.append(action)

    # 3. Stats Calculation
    agent_ret = ((history[-1] - 10000) / 10000) * 100
    mkt_start = test_df['close'].iloc[10]
    mkt_ret = ((test_df['close'].iloc[-1] - mkt_start) / mkt_start) * 100

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("STRATA Return", f"{agent_ret:.2f}%", delta=f"{agent_ret - mkt_ret:.2f}% vs Mkt")
    with m2: st.metric("Market Baseline", f"{mkt_ret:.2f}%")
    with m3: st.metric("Alpha Generated", f"{agent_ret - mkt_ret:.2f}%")

    # 4. Refined Comparison Chart
    with st.container(border=True):
        fig_h = go.Figure()
        
        # AI Equity Curve
        fig_h.add_trace(go.Scatter(x=test_df.index[10:], y=history, name="STRATA AI Portfolio", 
                                   line=dict(color="#00ff00", width=3)))
        
        # Market Baseline (Normalized to 10k)
        mkt_norm = (test_df['close'].iloc[10:] / mkt_start) * 10000
        fig_h.add_trace(go.Scatter(x=test_df.index[10:], y=mkt_norm, name="Market Buy & Hold", 
                                   line=dict(color="white", dash="dot")))
        
        fig_h.update_layout(template="plotly_dark", height=450, title="Equity Curve: AI vs Market Baseline",
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_h, use_container_width=True)
        
    # 5. Restore Explanation Feature
    st.markdown("""
    **Understanding the Results:**
    * **Identical Lines?** If the lines match, the AI has determined that 'Holding' is the optimal strategy for this market regime.
    * **Green Line Above White?** The AI is successfully 'Timing' the market (Buying low, Selling high).
    * **Green Line Below White?** The AI is 'Over-trading' or misinterpreting volatility as a trend change.
    """)
    st.markdown("</div>", unsafe_allow_html=True)