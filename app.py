import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import base64
import os

import yfinance as yf
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, LSTM

st.set_page_config(page_title="Stock Trend Prediction", page_icon="📈", layout="wide")

# --- Palette (validated categorical + neutral ink, fixed slot order) ---
# Dark-mode uses the palette's own dark-surface steps, not a flip of the light ones.
IS_DARK = st.context.theme.type == "dark"

if IS_DARK:
    BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
    INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#ffffff", "#c3c2b7", "#898781"
    GRID, SURFACE = "#2c2c2a", "#1a1a19"
else:
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
    GRID, SURFACE = "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


@st.cache_data
def get_bg_b64():
    with open("back.jpeg", "rb") as f:
        return base64.b64encode(f.read()).decode()


def hero(title, subtitle=""):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(rgba(10,10,20,0.35), rgba(10,10,20,0.6)),
                        url(data:image/jpeg;base64,{get_bg_b64()});
            background-size: cover;
            background-position: center;
            border-radius: 16px;
            padding: 2.75rem 2.25rem;
            margin-bottom: 1.75rem;
        ">
            <h1 style="color:#ffffff; margin:0; font-size:2.1rem; font-weight:700;">{title}</h1>
            {f'<p style="color:#e9e9e6; margin-top:0.5rem; margin-bottom:0; font-size:1.05rem;">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def coming_soon(label):
    with st.container(border=True):
        st.markdown(f"#### 🚧 {label}")
        st.write("This section is still under construction — check back soon.")


def home_page():
    hero(
        "Stock Market Trend Prediction & Analysis",
        "Enter a ticker to explore historical trends and an LSTM-based price forecast.",
    )

    col1, _ = st.columns([2, 3])
    with col1:
        user_input = st.text_input("Enter Stock Ticker", "TATASTEEL.NS")

    try:
        df = yf.Ticker(user_input).history(period="5y")
    except Exception as e:
        st.error(f"Could not fetch data for '{user_input}' from Yahoo Finance: {e}")
        st.stop()

    if df.empty:
        st.error(f"No data found for ticker '{user_input}'. NSE-listed stocks need a '.NS' suffix, e.g. 'TATASTEEL.NS'.")
        st.stop()

    latest_close = df['Close'].iloc[-1]
    latest_date = df.index[-1].strftime('%b %d, %Y')
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
    change = latest_close - prev_close
    pct_change = (change / prev_close * 100) if prev_close else 0

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Latest Close · {latest_date}", f"{latest_close:,.2f}", f"{change:+,.2f} ({pct_change:+.2f}%)")
        c2.metric("Day High", f"{df['High'].iloc[-1]:,.2f}")
        c3.metric("Day Low", f"{df['Low'].iloc[-1]:,.2f}")

    with st.container(border=True):
        st.markdown("##### Summary statistics")
        st.dataframe(df.describe(), use_container_width=True)

    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()

    tab1, tab2, tab3 = st.tabs(["Closing Price", "MA100", "MA100 & MA200"])

    with tab1:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.Close, color=BLUE, linewidth=1.6, label="Close")
        plt.legend()
        st.pyplot(fig)

    with tab2:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.Close, color=BLUE, linewidth=1.4, label="Close")
        plt.plot(ma100, color=ORANGE, linewidth=1.8, label="MA100")
        plt.legend()
        st.pyplot(fig)

    with tab3:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.Close, color=BLUE, linewidth=1.4, label="Close")
        plt.plot(ma100, color=ORANGE, linewidth=1.8, label="MA100")
        plt.plot(ma200, color=AQUA, linewidth=1.8, label="MA200")
        plt.legend()
        st.pyplot(fig)

    #Splitting data into Training and Testing
    data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.50)])                           #50%
    data_testing=pd.DataFrame(df['Close'][int(len(df)*0.50):int(len(df))])                 #50%

    LOOK_BACK = 100
    if len(data_training) <= LOOK_BACK or len(data_testing) <= LOOK_BACK:
        st.warning(f"Not enough price history for '{user_input}' to build a {LOOK_BACK}-day lookback window. Showing charts only, skipping prediction.")
        st.stop()

    scaler=MinMaxScaler(feature_range=(0,1))
    data_training_array=scaler.fit_transform(data_training)

    #ML

    @st.cache_resource(show_spinner=f"Training LSTM model for {user_input} (first run only, cached after)...")
    def train_lstm_model(ticker, training_array, look_back):
        x_train, y_train = [], []
        for i in range(look_back, training_array.shape[0]):
            x_train.append(training_array[i - look_back:i])
            y_train.append(training_array[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        model = Sequential()
        model.add(Input(shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=64, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=0)
        return model

    model = train_lstm_model(user_input, data_training_array, LOOK_BACK)

    past_100_days = data_training.tail(LOOK_BACK)
    final_test_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_test_df)

    x_test, y_test = [], []
    for i in range(LOOK_BACK, input_data.shape[0]):
        x_test.append(input_data[i - LOOK_BACK:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test, verbose=0)

    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted.flatten() * scale_factor
    y_test = y_test * scale_factor

    with st.container(border=True):
        st.markdown("##### Predictions vs original")
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, color=BLUE, linewidth=1.6, label="Original Price")
        plt.plot(y_predicted, color=ORANGE, linewidth=1.6, label="Predicted Price")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(fig2)


def news_page():
    hero("BuzzNation News Hub", "Latest business headlines by country.")

    import pycountry
    import requests as req

    col1, col2 = st.columns([3, 1])
    with col1:
        user = st.text_input('Enter country name', 'India')
    with col2:
        st.write("")
        st.write("")
        btn = st.button('Search', use_container_width=True)

    try:
        secrets_key = st.secrets.get("NEWS_API_KEY", "")
    except Exception:
        secrets_key = ""
    apiKEY = os.environ.get("NEWS_API_KEY") or secrets_key

    if btn:
        if not apiKEY:
            st.error("No NewsAPI key configured. Add NEWS_API_KEY to .streamlit/secrets.toml or set it as an environment variable.")
            st.stop()

        country_obj=pycountry.countries.get(name=user)
        if country_obj is None:
            st.error(f"Could not recognize country '{user}'. Try the official English name, e.g. 'India' or 'United States'.")
            st.stop()
        country=country_obj.alpha_2

        url=f"https://newsapi.org/v2/top-headlines?country={country}&category=business&apiKey={apiKEY}"

        r=req.get(url)
        payload=r.json()
        if payload.get('status') != 'ok':
            st.error(f"NewsAPI error: {payload.get('message', 'unknown error')}")
            st.stop()

        articles=payload.get('articles', [])
        if not articles:
            st.info("No articles found.")
        for article in articles:
            with st.container(border=True):
                st.markdown(f"##### {article['title']}")
                meta = article['source']['name']
                if article['author']:
                    meta += f" · {article['author']}"
                meta += f" · {article['publishedAt']}"
                st.caption(meta)
                if article['description']:
                    st.write(article['description'])
                st.markdown(f"[Read more →]({article['url']})")


def insights_page():
    hero("Insights")
    coming_soon("Insights")


def community_page():
    hero("Community")
    coming_soon("Community")


def about_page():
    hero("About")
    with st.container(border=True):
        st.write("This is a minor project created by **Harshit**, **Harsh Goyal**, and **Satwik Shukla**.")


pg = st.navigation([
    st.Page(home_page, title="Home", icon="🏠", default=True),
    st.Page(news_page, title="News", icon="📰"),
    st.Page(insights_page, title="Insights", icon="💡"),
    st.Page(community_page, title="Community", icon="👥"),
    st.Page(about_page, title="About", icon="ℹ️"),
])
st.sidebar.markdown("### 📈 StockTrend")
pg.run()
