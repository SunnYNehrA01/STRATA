import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import streamlit as st

# --- 1. Load Model (Optimized Caching) ---
@st.cache_resource
def load_finbert():
    # FinBERT is the industry standard for financial sentiment
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# --- 2. Main Processing Function ---
def analyze_news_impact(ticker):
    nlp = load_finbert()
    stock = yf.Ticker(ticker)
    
    try:
        raw_news = stock.news
    except Exception:
        return None, 0.5, []

    if not raw_news:
        return None, 0.5, []

    processed_news = []
    total_score = 0
    
    # We only process the Top 5 to keep the speed fast
    for item in raw_news[:5]:
        # DEFENSIVE KEY CHECK: Handles both old and new yfinance news structures
        # Look for 'title' in the root, then look inside 'content'
        headline = item.get('title') or item.get('content', {}).get('title')
        link = item.get('link') or item.get('content', {}).get('canonicalUrl', {}).get('url') or "#"
        
        # Skip items that are just ads or have no text
        if not headline:
            continue
            
        result = nlp(headline)[0]
        
        # Map labels to weights
        val = 0.5
        if result['label'] == 'positive': val = 1.0
        elif result['label'] == 'negative': val = 0.0
        
        total_score += val
        processed_news.append({
            'headline': headline,
            'sentiment': result['label'],
            'confidence': result['score'],
            'link': link
        })

    if not processed_news:
        return "NEUTRAL", 0.5, []

    # Calculate Aggregate
    avg_score = total_score / len(processed_news)
    
    if avg_score > 0.6: label = "BULLISH"
    elif avg_score < 0.4: label = "BEARISH"
    else: label = "NEUTRAL"

    # Sort to show highest confidence news first
    high_impact = sorted(processed_news, key=lambda x: x['confidence'], reverse=True)
    
    return label, avg_score, high_impact