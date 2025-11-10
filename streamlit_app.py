import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np

st.set_page_config(page_title="NSE Stock Screener", page_icon="📈", layout="wide")

# App Title
st.title("📈 NSE Stock Screener")
st.markdown("Interactive tool for screening NSE stocks with technical and fundamental filters")

# NSE Stock List (Top 100 NSE stocks)
NSE_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
    "ITC.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "LT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NESTLEIND.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "M&M.NS",
    "BAJAJFINSV.NS", "TECHM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "ADANIPORTS.NS",
    "JSWSTEEL.NS", "HINDALCO.NS", "INDUSINDBK.NS", "DRREDDY.NS", "CIPLA.NS",
    "DIVISLAB.NS", "BRITANNIA.NS", "GRASIM.NS", "COALINDIA.NS", "SHREECEM.NS",
    "EICHERMOT.NS", "UPL.NS", "BAJAJ-AUTO.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "APOLLOHOSP.NS", "ADANIENT.NS", "TATACONSUM.NS", "HEROMOTOCO.NS", "VEDL.NS"
]

# Calculate ADX
def calculate_adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = abs(100 * (minus_dm.rolling(period).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    
    return adx.iloc[-1] if len(adx) > 0 else 0

# Calculate RSI
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        # Handle division by zero
    if loss.iloc[-1] == 0:
        rsi = 100.0  # When there's no loss, RSI is maximum
    else:
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) > 0 else 50

# Sidebar Filters
st.sidebar.header("🎯 Filter Settings")

# Strategy Presets
st.sidebar.subheader("Strategy Presets")
strategy = st.sidebar.selectbox(
    "Choose Strategy",
    ["Custom", "Aggressive Momentum", "Steady Growth", "Value + Growth"]
)

# Set default values based on strategy
if strategy == "Aggressive Momentum":
    default_adx = (25, 100)
    default_volume = 5000000
    default_rsi = (60, 80)
    default_pat = 20
    default_pe = (0, 50)
elif strategy == "Steady Growth":
    default_adx = (20, 50)
    default_volume = 1000000
    default_rsi = (40, 70)
    default_pat = 10
    default_pe = (0, 30)
elif strategy == "Value + Growth":
    default_adx = (15, 40)
    default_volume = 500000
    default_rsi = (30, 60)
    default_pat = 5
    default_pe = (0, 20)
else:
    default_adx = (20, 100)
    default_volume = 1000000
    default_rsi = (30, 70)
    default_pat = 0
    default_pe = (0, 100)

# Technical Filters
st.sidebar.subheader("Technical Filters")
adx_range = st.sidebar.slider("ADX Range", 0, 100, default_adx)
volume_min = st.sidebar.number_input("Minimum Volume", min_value=0, value=default_volume, step=100000)
rsi_range = st.sidebar.slider("RSI Range", 0, 100, default_rsi)

# Fundamental Filters
st.sidebar.subheader("Fundamental Filters")
pat_growth_min = st.sidebar.number_input("Min PAT Growth %", value=default_pat, step=5)
pe_range = st.sidebar.slider("P/E Ratio Range", 0, 100, default_pe)

# Sector Filter
st.sidebar.subheader("Sector Filter")
sectors = st.sidebar.multiselect(
    "Select Sectors",
    ["All", "IT", "Banking", "Pharma", "Auto", "FMCG", "Energy"],
    default=["All"]
)

# Screening Button
if st.sidebar.button("🔍 Run Screener", type="primary"):
    st.session_state.run_screening = True

# Main Content
if 'run_screening' in st.session_state and st.session_state.run_screening:
    with st.spinner("Screening stocks... Please wait"):
        results = []
        progress_bar = st.progress(0)
        
        for idx, stock in enumerate(NSE_STOCKS):
            try:
                # Download stock data
                ticker = yf.Ticker(stock)
                df = ticker.history(period="3mo")
                
                if len(df) < 20:
                    continue
                
                # Calculate indicators
                adx = calculate_adx(df)
                rsi = calculate_rsi(df)
                avg_volume = df['Volume'].tail(20).mean()
                current_price = df['Close'].iloc[-1]
                
                # Get info
                info = ticker.info
                pe_ratio = info.get('trailingPE', 0)
                market_cap = info.get('marketCap', 0)
                sector = info.get('sector', 'Unknown')
                
                # Mock PAT growth (in real scenario, fetch from financial statements)
                pat_growth = (df['Close'].iloc[-1] - df['Close'].iloc[-60]) / df['Close'].iloc[-60] * 100 if len(df) >= 60 else 0
                
                # Apply filters
                if (adx_range[0] <= adx <= adx_range[1] and
                    avg_volume >= volume_min and
                    rsi_range[0] <= rsi <= rsi_range[1] and
                    pat_growth >= pat_growth_min and
                    pe_range[0] <= pe_ratio <= pe_range[1]):
                    
                    results.append({
                        'Symbol': stock.replace('.NS', ''),
                        'Price': round(current_price, 2),
                        'ADX': round(adx, 2),
                        'RSI': round(rsi, 2),
                        'Volume': int(avg_volume),
                        'PAT Growth %': round(pat_growth, 2),
                        'P/E': round(pe_ratio, 2),
                        'Market Cap': market_cap,
                        'Sector': sector
                    })
                
                # Update progress
                progress_bar.progress((idx + 1) / len(NSE_STOCKS))
                
            except Exception as e:
                continue
        
        # Store results in session state
        st.session_state.results_df = pd.DataFrame(results)
        st.session_state.screening_complete = True

# Display Results
if 'screening_complete' in st.session_state and st.session_state.screening_complete:
    df_results = st.session_state.results_df
    
    if len(df_results) > 0:
        st.success(f"✅ Found {len(df_results)} stocks matching your criteria!")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Results Table", "📈 Charts", "💾 Export"])
        
        with tab1:
            st.dataframe(df_results, use_container_width=True, height=400)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Stocks", len(df_results))
            with col2:
                st.metric("Avg ADX", round(df_results['ADX'].mean(), 2))
            with col3:
                st.metric("Avg RSI", round(df_results['RSI'].mean(), 2))
            with col4:
                st.metric("Avg PAT Growth", f"{round(df_results['PAT Growth %'].mean(), 2)}%")
        
        with tab2:
            # ADX vs RSI Scatter Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_results['ADX'],
                y=df_results['RSI'],
                mode='markers+text',
                text=df_results['Symbol'],
                textposition='top center',
                marker=dict(size=10, color=df_results['PAT Growth %'], colorscale='Viridis', showscale=True),
                name='Stocks'
            ))
            fig.update_layout(
                title='ADX vs RSI Analysis',
                xaxis_title='ADX',
                yaxis_title='RSI',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 10 by PAT Growth
            top_10 = df_results.nlargest(10, 'PAT Growth %')
            fig2 = go.Figure(data=[
                go.Bar(x=top_10['Symbol'], y=top_10['PAT Growth %'], marker_color='lightgreen')
            ])
            fig2.update_layout(
                title='Top 10 Stocks by PAT Growth',
                xaxis_title='Stock',
                yaxis_title='PAT Growth %',
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.subheader("Export Results")
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"nse_screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.info("💡 Tip: Use this CSV file for further analysis in Excel or other tools!")
    else:
        st.warning("⚠️ No stocks found matching your criteria. Try adjusting the filters.")

# Footer
st.markdown("---")
st.markdown("📊 **NSE Stock Screener** | Built with Streamlit | Data from Yahoo Finance")
