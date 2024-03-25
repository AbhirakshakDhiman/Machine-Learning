import streamlit as st
import datetime 
import yfinance
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs

START="2018-01-01"
TODAY=datetime.date.today().strftime("%Y-%m-%d")
st.title("Stock predicition app")

stocks=("GOOG","AAPL","MSFT","GME")
selected_stock=st.selectbox("Selected stock",stocks)

@st.cache_data
def load_data(stock):
    data=yfinance.download(stock,START,TODAY)
    data.reset_index(inplace=True)
    return data

data=load_data(selected_stock)
st.subheader("Stock data")
st.write(data.tail())

fig1=graph_objs.Figure()
fig1.add_trace(graph_objs.Scatter(x=data["Date"],y=data["Open"],name="stock_open"))
fig1.add_trace(graph_objs.Scatter(x=data["Date"],y=data["Close"],name="stock_close"))
fig1.layout.update(title_text="Time Series data",xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)


df_train=data[["Date","Close"]]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=30)
forecast=m.predict(future)

st.subheader("Forecast data")
st.write(forecast.tail())
fig2=plot_plotly(m,forecast)
st.plotly_chart(fig2)