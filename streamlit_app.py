import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

st.set_page_config(page_title="NSE Stock Screener", page_icon="📈", layout="wide")

# App Title
st.title("📈 NSE Stock Screener")
st.markdown("Interactive tool for screening NSE stocks with technical and fundamental filters")

# Comprehensive NSE Stock Database with Sector Mappings
STOCK_SECTORS = {
    # IT
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT", "TECHM.NS": "IT",
    "LTI.NS": "IT", "MPHASIS.NS": "IT", "COFORGE.NS": "IT",
    # Banking
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "KOTAKBANK.NS": "Banking", "SBIN.NS": "Banking",
    "AXISBANK.NS": "Banking", "INDUSINDBK.NS": "Banking", "BANKBARODA.NS": "Banking", "PNB.NS": "Banking",
    "IDFCFIRSTB.NS": "Banking", "FEDERALBNK.NS": "Banking",
    # Pharma
    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma", "DIVISLAB.NS": "Pharma",
    "BIOCON.NS": "Pharma", "AUROPHARMA.NS": "Pharma", "LUPIN.NS": "Pharma", "TORNTPHARM.NS": "Pharma",
    # Auto
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "M&M.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
    "HEROMOTOCO.NS": "Auto", "EICHERMOT.NS": "Auto", "ASHOKLEY.NS": "Auto", "TVSMOTOR.NS": "Auto",
    # FMCG
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG", "BRITANNIA.NS": "FMCG",
    "DABUR.NS": "FMCG", "MARICO.NS": "FMCG", "GODREJCP.NS": "FMCG", "TATACONSUM.NS": "FMCG",
    # Energy
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "NTPC.NS": "Energy", "POWERGRID.NS": "Energy",
    "COALINDIA.NS": "Energy", "IOC.NS": "Energy", "BPCL.NS": "Energy", "GAIL.NS": "Energy",
    # Metals
    "TATASTEEL.NS": "Metals", "HINDALCO.NS": "Metals", "JSWSTEEL.NS": "Metals", "VEDL.NS": "Metals",
    "JINDALSTEL.NS": "Metals", "SAIL.NS": "Metals",
    # Cement
    "ULTRACEMCO.NS": "Cement", "GRASIM.NS": "Cement", "SHREECEM.NS": "Cement",
    # Telecom
    "BHARTIARTL.NS": "Telecom",
    # Financial Services
    "BAJFINANCE.NS": "Financial Services", "BAJAJFINSV.NS": "Financial Services",
    "HDFCLIFE.NS": "Financial Services", "SBILIFE.NS": "Financial Services",
    # Infrastructure
    "LT.NS": "Infrastructure", "ADANIPORTS.NS": "Infrastructure",
    # Consumer Durables
    "TITAN.NS": "Consumer Durables", "HAVELLS.NS": "Consumer Durables",
    # Others
    "APOLLOHOSP.NS": "Healthcare", "ADANIENT.NS": "Diversified", "UPL.NS": "Chemicals",
}

NSE_STOCKS = list(STOCK_SECTORS.keys())

# --- Indicators ---
def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    high = df['High']
    low = df['Low']
    close = df['Close']

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period, min_periods=period).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.rolling(period, min_periods=period).mean()

    val = adx.iloc[-1]
    return float(val) if pd.notna(val) else 0.0

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
    delta = df['Close'].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(period, min_periods=period).mean()
    avg_loss = losses.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    last = rsi_series.iloc[-1]
    return float(last) if pd.notna(last) else 50.0

# Sidebar Filters
st.sidebar.header("🎯 Filter Settings")

# Strategy Presets
st.sidebar.subheader("Strategy Presets")
strategy = st.sidebar.selectbox(
    "Choose Strategy",
    ["Custom", "Aggressive Momentum", "Steady Growth", "Value + Growth"]
)

# Defaults by strategy
if strategy == "Aggressive Momentum":
    default_adx = (25, 100); default_volume = 5_000_000; default_rsi = (60, 80); default_pat = 20; default_pe = (0, 50)
elif strategy == "Steady Growth":
    default_adx = (20, 50); default_volume = 1_000_000; default_rsi = (40, 70); default_pat = 10; default_pe = (0, 30)
elif strategy == "Value + Growth":
    default_adx = (15, 40); default_volume = 500_000; default_rsi = (30, 60); default_pat = 5; default_pe = (0, 20)
else:
    default_adx = (20, 100); default_volume = 1_000_000; default_rsi = (30, 70); default_pat = 0; default_pe = (0, 100)

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
    ["All", "IT", "Banking", "Pharma", "Auto", "FMCG", "Energy", "Metals", "Cement", "Telecom",
     "Financial Services", "Infrastructure", "Consumer Durables", "Healthcare", "Diversified", "Chemicals"],
    default=["All"]
)

# Screening Button
if st.sidebar.button("🔍 Run Screener", type="primary"):
    st.session_state.run_screening = True

# Main Content
if st.session_state.get('run_screening'):
    with st.spinner("Screening stocks... Please wait"):
        results = []
        progress_bar = st.progress(0)

        for idx, stock in enumerate(NSE_STOCKS):
            try:
                ticker = yf.Ticker(stock)
                df = ticker.history(period="3mo")

                if len(df) < 20:
                    progress_bar.progress((idx + 1) / len(NSE_STOCKS))
                    continue

                # Indicators
                adx = calculate_adx(df)
                rsi = calculate_rsi(df)
                avg_volume = df['Volume'].tail(20).mean()
                current_price = df['Close'].iloc[-1]

                # Company info (safe defaults)
                pe_ratio = 0.0
                market_cap = 0
                try:
                    info = ticker.get_info()
                    if isinstance(info, dict):
                        pe_ratio = float(info.get('trailingPE') or 0.0)
                        market_cap = int(info.get('marketCap') or 0)
                except Exception:
                    pass

                sector = STOCK_SECTORS.get(stock, 'Unknown')

                # Mock PAT growth (placeholder)
                pat_growth = (
                    (df['Close'].iloc[-1] - df['Close'].iloc[-60]) / df['Close'].iloc[-60] * 100
                    if len(df) >= 60 else 0.0
                )

                # Apply filters
                if (
                    (adx_range[0] <= adx <= adx_range[1]) and
                    (avg_volume >= volume_min) and
                    (rsi_range[0] <= rsi <= rsi_range[1]) and
                    (pat_growth >= float(pat_growth_min)) and
                    (pe_range[0] <= pe_ratio <= pe_range[1]) and
                    (('All' in sectors) or (sector in sectors))
                ):
                    results.append({
                        'Symbol': stock.replace('.NS', ''),
                        'Price': round(float(current_price), 2),
                        'ADX': round(adx, 2),
                        'RSI': round(rsi, 2),
                        'Volume': int(avg_volume),
                        'PAT Growth %': round(float(pat_growth), 2),
                        'P/E': round(pe_ratio, 2),
                        'Market Cap': market_cap,
                        'Sector': sector
                    })

                progress_bar.progress((idx + 1) / len(NSE_STOCKS))

            except Exception:
                # Skip problematic tickers but keep the app running
                progress_bar.progress((idx + 1) / len(NSE_STOCKS))
                continue

        st.session_state.results_df = pd.DataFrame(results)
        st.session_state.screening_complete = True
        st.session_state.run_screening = False

# Display Results
if st.session_state.get('screening_complete'):
    df_results = st.session_state.results_df

    if len(df_results) > 0:
        st.success(f"✅ Found {len(df_results)} stocks matching your criteria!")
        tab1, tab2, tab3 = st.tabs(["📊 Results Table", "📈 Charts", "💾 Export"])

        with tab1:
            st.dataframe(df_results, use_container_width=True, height=400)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Stocks", len(df_results))
            col2.metric("Avg ADX", round(df_results['ADX'].mean(), 2))
            col3.metric("Avg RSI", round(df_results['RSI'].mean(), 2))
            col4.metric("Avg PAT Growth", f"{round(df_results['PAT Growth %'].mean(), 2)}%")

        with tab2:
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
            fig.update_layout(title='ADX vs RSI Analysis', xaxis_title='ADX', yaxis_title='RSI', height=500)
            st.plotly_chart(fig, use_container_width=True)

            top_10 = df_results.nlargest(10, 'PAT Growth %')
            fig2 = go.Figure(data=[go.Bar(x=top_10['Symbol'], y=top_10['PAT Growth %'])])
            fig2.update_layout(title='Top 10 Stocks by PAT Growth', xaxis_title='Stock', yaxis_title='PAT Growth %', height=400)
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

st.markdown("---")
st.markdown("📊 **NSE Stock Screener** | Built with Streamlit | Data from Yahoo Finance")
