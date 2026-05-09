import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import feedparser
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from streamlit_autorefresh import st_autorefresh
from nsetools import Nse
import requests
import base64

# ---------------- LOGIN ---------------- #

username = st.sidebar.text_input("Username")

password = st.sidebar.text_input("Password", type="password")

if username != "santosh" or password != "1234":
    st.warning("Enter Login Details")
    st.stop()
# ---------------- DARK THEME CSS ---------------- #

st.markdown("""
    <style>

    .stApp {
        background-color: #0E1117;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    h1, h2, h3, h4, h5, h6 {
        color: white;
    }

    </style>
""", unsafe_allow_html=True)

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="SANTOSH AI",
    page_icon="📈",
    layout="wide"
)
# ---------------- BACKGROUND IMAGE ---------------- #

def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64("background.jpg")

page_bg = f"""
<style>

.stApp {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0.7);
}}

.block-container {{
    background: rgba(0,0,0,0.65);
    padding: 2rem;
    border-radius: 20px;
}}

/* INPUT BOX */

.stTextInput input {{
    background-color: rgba(0,0,0,0.6) !important;
    color: white !important;
    border: 1px solid #00ff99 !important;
}}

/* DATAFRAME */

[data-testid="stDataFrame"] {{
    background-color: rgba(0,0,0,0.6) !important;
    color: white !important;
    border-radius: 15px;
}}

/* TABLE */

table {{
    background-color: rgba(0,0,0,0.6) !important;
    color: white !important;
}}

/* DOWNLOAD BUTTON */

.stDownloadButton button {{
    background-color: rgba(0,0,0,0.7) !important;
    color: white !important;
    border: 1px solid #00ff99 !important;
    border-radius: 10px;
}}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- AUTO REFRESH ---------------- #

st_autorefresh(interval=60000, key="refresh")

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    selected = option_menu(
        menu_title="SANTOSH AI",
        options=[
            "Dashboard",
            "Prediction",
            "Charts",
            "Signals"
        ],
        icons=[
            "speedometer2",
            "graph-up-arrow",
            "bar-chart",
            "activity"
        ],
        menu_icon="robot",
        default_index=0,
    )

st.sidebar.info("AI Powered Stock Prediction App 📈")

# ---------------- STOCK DROPDOWN ---------------- #

# ---------------- STOCK SEARCH ---------------- #

stock_input = st.text_input(
    "🔍 Enter NSE Stock Symbol",
    "RELIANCE"
)

stock = stock_input.upper() + ".NS"

# ---------------- DOWNLOAD DATA ---------------- #

ticker = yf.Ticker(stock)

data = ticker.history(period="1y")

if data.empty:
    st.error("No data found for selected stock")
    st.stop()
# ---------------- SHOW DATA ---------------- #

st.subheader("📊 Latest Stock Data")

st.dataframe(data.tail())

# ---------------- DOWNLOAD CSV ---------------- #

csv = data.to_csv().encode('utf-8')

st.download_button(
    label="📥 Download Stock Data",
    data=csv,
    file_name=f"{stock}_data.csv",
    mime="text/csv"
)

# ---------------- STOCK CHART ---------------- #

st.subheader("📈 Closing Price Chart")

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(data.index, data["Close"])

ax.set_xlabel("Date")
ax.set_ylabel("Price")

ax.grid()

st.pyplot(fig)

# ---------------- CANDLESTICK CHART ---------------- #

st.subheader("🕯️ Candlestick Chart")

fig_candle = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)])

fig_candle.update_layout(
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig_candle, use_container_width=True)

# ---------------- MOVING AVERAGE ---------------- #

data["MA50"] = data["Close"].rolling(50).mean()

st.subheader("📉 Moving Average")

fig2, ax2 = plt.subplots(figsize=(12,6))

ax2.plot(data["Close"], label="Closing Price")

ax2.plot(data["MA50"], label="50 Day MA")

ax2.legend()

ax2.grid()

st.pyplot(fig2)

# ---------------- ML PREDICTION ---------------- #

data["Prediction"] = data["Close"].shift(-1)

data.dropna(inplace=True)

# Features and Target

X = np.array(data[["Close"]])

y = np.array(data["Prediction"])

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Training

model = LinearRegression()

model.fit(X_train, y_train)

# Predictions

predictions = model.predict(X_test)

# ---------------- ACCURACY ---------------- #

score = model.score(X_test, y_test)

st.subheader("🎯 Model Accuracy")

st.success(f"{score:.2f}")

# ---------------- ERROR ---------------- #

error = mean_absolute_error(y_test, predictions)

st.subheader("📌 Mean Absolute Error")

st.write(error)

# ---------------- TOMORROW PREDICTION ---------------- #

latest_price = np.array([[data["Close"].iloc[-1]]])

tomorrow_price = model.predict(latest_price)

st.subheader("🚀 Tomorrow Predicted Price")

st.success(f"₹ {tomorrow_price[0]:.2f}")

# ---------------- BUY / SELL SIGNAL ---------------- #

st.subheader("📊 Trading Signal")

latest_close = data["Close"].iloc[-1]

latest_ma50 = data["MA50"].iloc[-1]

# BUY SIGNAL
if latest_close > latest_ma50:

    st.success("🟢 BUY SIGNAL")

    st.write("Current Price is above 50 Day Moving Average")

# SELL SIGNAL
else:

    st.error("🔴 SELL SIGNAL")

    st.write("Current Price is below 50 Day Moving Average")

# ---------------- ACTUAL VS PREDICTED ---------------- #

st.subheader("📊 Actual vs Predicted")

pred_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

fig3, ax3 = plt.subplots(figsize=(12,6))

ax3.plot(pred_df["Actual"][:50], label="Actual Price")

ax3.plot(pred_df["Predicted"][:50], label="Predicted Price")

ax3.legend()

ax3.grid()

st.pyplot(fig3)

# ---------------- STOCK NEWS ---------------- #

st.subheader("📰 Latest Stock News")

# Remove .NS from stock symbol
company_name = stock.replace(".NS", "")

# Dynamic News URL
news_url = f"https://news.google.com/rss/search?q={company_name}+stock"

feed = feedparser.parse(news_url)

if len(feed.entries) > 0:

    for entry in feed.entries[:5]:

        st.markdown(f"### 👉 {entry.title}")

        st.write(entry.link)

        st.write("---")

else:

    st.error("News not available")

# ---------------- PORTFOLIO TRACKER ---------------- #

st.subheader("💼 Portfolio Tracker")

shares = st.number_input("Enter Number of Shares", min_value=1)

investment = shares * data["Close"].iloc[-1]

st.success(f"Current Investment Value: ₹ {investment:.2f}")

# ---------------- LIVE NSE MARKET MOVERS ---------------- #

st.subheader("📊 Live NSE Market Movers")

headers = {
    "User-Agent": "Mozilla/5.0"
}

url = "https://www.nseindia.com/api/live-analysis-variations?index=gainers"

session = requests.Session()

session.get(
    "https://www.nseindia.com",
    headers=headers
)

response = session.get(
    url,
    headers=headers
)

data_json = response.json()

gainers_data = data_json["NIFTY"]["data"][:5]

# TOP GAINERS
st.markdown("🟢 Top Gainers")

for stock_data in gainers_data:
    st.write(stock_data["symbol"])

    stock_name = stock_data["symbol"]

    percent = stock_data.get("pChange")

    if percent is None:
        percent = stock_data.get("perChange")

    if percent is None:
        percent = stock_data.get("percentChange")

    if percent is None:
        percent = "N/A"

    st.success(f"{stock_name} : {percent}%")

# ---------------- TOP LOSERS ---------------- #

st.markdown("🔴 Top Losers")

losers_url = "https://www.nseindia.com/api/live-analysis-variations?index=loosers"

response_losers = session.get(
    losers_url,
    headers=headers
)

losers_json = response_losers.json()

losers_data = losers_json["NIFTY"]["data"][:5]

for stock_data in losers_data:

    stock_name = stock_data["symbol"]

    percent = stock_data.get("pChange")

    if percent is None:
        percent = stock_data.get("perChange")

    if percent is None:
        percent = stock_data.get("percentChange")

    if percent is None:
        percent = "N/A"

    st.error(
        f"📉 {stock_name} : {percent}%"
    )

# ---------------- FOOTER ---------------- #

st.write("Made with ❤️ Keep Smiling")
st.write("Ek smile aapke din ko behtar bana sakta hai")