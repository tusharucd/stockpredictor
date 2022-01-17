from pandas.core.base import DataError
from datetime  import date
from datetime import timedelta
from datetime import datetime
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
import time
import base64

#Set the page content to cover entire screen
st.set_page_config(layout="wide")

#Function to load data according to selected stock, start date, end date
@st.cache(allow_output_mutation=True)#Caching the data to improve performance
def load_data(ticker):
    data = yf.download(ticker, Start, Today)
    data.reset_index(inplace = True)
    return data

#Function to load the title page
def title():
    st.markdown("<h1 style = 'color:Black; background-color:White; text-align:center; opacity:0.6'>  Stock Predictor Application </h1>",unsafe_allow_html=True)
    main_bg = "./stock-market-image-getty.jpeg"
    main_bg_ext = "jpeg"
    st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
        opacity: 0.9
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#Function to display company information
def companyinfo(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    with st.spinner('Loading...'):
        st.subheader(info['longName']) 
        st.markdown('** Sector **: ' + info['sector'])
        st.markdown('** Industry **: ' + info['industry'])
        st.markdown('** Phone **: ' + info['phone'])
        st.markdown('** Address **: ' + info['address1'] + ', ' + info['city'] + ', ' + info['zip'] + ', '  +  info['country'])
        st.markdown('** Business Summary **')
        st.info(info['longBusinessSummary'])

        fundamental = {
                'Enterprise Value (USD)': info['enterpriseValue'],
                'Enterprise To Revenue Ratio': info['enterpriseToRevenue'],
                'Enterprise To Ebitda Ratio': info['enterpriseToEbitda'],
                'Net Income (USD)': info['netIncomeToCommon'],
                'Profit Margin Ratio': info['profitMargins'],
                'Forward PE Ratio': info['forwardPE'],
                'PEG Ratio': info['pegRatio'],
                'Price to Book Ratio': info['priceToBook'],
                'Forward EPS (USD)': info['forwardEps'],
                'Beta ': info['beta'],
                'Book Value (USD)': info['bookValue'],
                'Dividend Rate (%)': info['dividendRate'], 
                'Dividend Yield (%)': info['dividendYield'],
                'Five year Avg Dividend Yield (%)': info['fiveYearAvgDividendYield'],
                'Payout Ratio': info['payoutRatio']
            }
        
        fund_info = pd.DataFrame.from_dict(fundamental, orient='index')
        fund_info = fund_info.rename(columns={0: 'Value'})
        st.subheader('Fundamental Info') 
        st.table(fund_info)

#Function to display data for last 5 days
def tabular_data():
    st.subheader("Stock data for last 5 days")
    st.write(data.tail())

#Function to display decriptive statistics on the stock data
def descriptive_stats():
    st.subheader("Statistical Analysis of Stock")
    desc=data.describe()
    desc.rename(index={
        'count': 'Count', 
        'mean': 'Mean', 
        'std': 'Standard Deviation', 
        'min': 'Minimum', 
        '25%' : 'First Quartile', 
        '50%' : 'Second Quartile', 
        '75%' : 'Third Quartile', 
        'max': 'Maximum'},
        inplace =True)
    st.write(desc)

#Function to display Daily Returns wrt Mean and Standard Deviation
def descriptive():
    close = data['Close']
    returns = close.pct_change(1)
    data['Returns'] = returns
    st.subheader("Daily Return w.r.t mean and standard deviation")
    mean = data['Returns'].mean()
    std = data['Returns'].std()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=returns)
    )
    fig.add_vline(x=mean, line_width=3, line_dash="dash", line_color="red")
    fig.add_vline(x=std, line_width=3, line_dash="dash", line_color="green")
    fig.add_vline(x=-std, line_width=3, line_dash="dash", line_color="green")
    fig.update_layout(xaxis_title='Daily Returns')
    st.plotly_chart(fig,use_container_width=True)

#Function to display rolling standard deviation data
def rolling_std():
    close = data['Close']
    returns = close.pct_change(1)
    data['Returns'] = returns
    st.subheader("20 days rolling standard deviation data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=returns.rolling(20).std(), line = dict(color='blue')))
    fig.update_layout(xaxis_title='20 days rolling standard deviation data')
    st.plotly_chart(fig,use_container_width=True)

#Function to display time-series graph
def plot_data():
    st.subheader('Graphical Analysis of Stock')
    fig = go.Figure()
    fig.add_trace(
            go.Scatter(
                    x=data['Date'], 
                    y=data['Open'], 
                    name='Stock Open Price'
                    )
                )
    fig.add_trace(
            go.Scatter(
                    x=data['Date'], 
                    y=data['Close'], 
                    name='Stock Close Price'
                    )
                )
    fig.update_layout(title_text='Time Series Data',xaxis_title='Years', yaxis_title='Stock Price', xaxis_rangeslider_visible = True)
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(fig,use_container_width=True)

#Function to display candlestick graph
def candlestick():
    fig = go.Figure(
        data=[go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'] ,
            close=data['Close'])])
    fig.layout.update(
        title_text='Candlestick data',
        xaxis_rangeslider_visible = True,
        xaxis_title = 'Years',
        yaxis_title = 'Stock Price')
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(fig,use_container_width=True)

#Function to display Relative Strength Index on the stock
def RSI():
    st.subheader("Relative Strength Index (RSI)")
    delta = data['Adj Close'].diff(1)
    delta.dropna(inplace = True)

    positive = delta.copy()
    negative = delta.copy()
    
    positive[positive < 0] = 0  #filtering out positive returns from whole data
    negative[negative > 0] = 0  #filtering out negative returns from whole data

    period = st.number_input('Insert period (Days): ', min_value=1, max_value=100, value=14)
    
    avg_gain = positive.rolling(window = period).mean()
    avg_loss = abs(negative.rolling(window = period).mean())

    relative_strength = avg_gain / avg_loss
    rsi = 100.0 - (100 / (1 + relative_strength))

    rsi_data = pd.DataFrame()
    rsi_data['Adj Close'] = data['Adj Close']
    rsi_data['RSI'] = rsi
    fig = go.Figure()
    fig.add_trace(
            go.Scatter(
                    x=data['Date'], 
                    y=rsi_data['RSI']
                    )
                )
    fig.add_hline(y=70, line_dash='dash', line_color='green')
    fig.add_hline(y=30, line_dash='dash', line_color='green')
    fig.layout.update(
        title_text='Relative Strength Index (RSI)',
        xaxis_title='Years', 
        yaxis_title='Percentage', 
        xaxis_rangeslider_visible = True)
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig,use_container_width=True) 

#Function to graph Daily Returns on the stock
def daily_returns():
    close = data['Close']
    returns = close.pct_change(1)
    data['Returns'] = returns
    st.subheader("Daily Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Returns'], line = dict(color='firebrick')))
    fig.update_layout(xaxis_title='Years', yaxis_title='Returns %', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig,use_container_width=True)

#Function to calculate Moving Average on the stock
def MovingAverage(data, size):
    df = data.copy()
    df['sma'] = df['Adj Close'].rolling(size).mean() #Simple Moving Average Array
    df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean() #Exponential Moving Average Array
    df.dropna(inplace=True)
    return df

#Function to display Moving Average on the stock
def PlotMA(ticker):
    st.subheader("Moving Average Analysis")
    numYear = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)    
    windowSize = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1) 
    start = date.today()-timedelta(numYear * 365)
    end = date.today()
    dataMA = yf.download(ticker,start,end)
    df_ma = MovingAverage(dataMA, windowSize)
    df_ma = df_ma.reset_index()
    fig = go.Figure()
    
    fig.add_trace(
            go.Scatter(
                    x = df_ma['Date'],
                    y = df_ma['Adj Close'],
                    name = "Prices Over Last " + str(numYear) + " Year(s)"
                )
        )
    
    fig.add_trace(
                go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['sma'],
                        name = "SMA" + str(windowSize) + " Over Last " + str(numYear) + " Year(s)"
                    )
            )
    
    fig.add_trace(
                go.Scatter(
                        x = df_ma['Date'],
                        y = df_ma['ema'],
                        name = "EMA" + str(windowSize) + " Over Last " + str(numYear) + " Year(s)"
                    )
            )

    fig.update_layout(legend_title_text='Trend', xaxis_rangeslider_visible = True)
    fig.update_yaxes(tickprefix="$")
    
    st.plotly_chart(fig, use_container_width=True) 

#Function to model and forecast stock value
def forecasting(ticker):

    #Modelling past data
    st.subheader('Select the modeling period')
    start_train = st.date_input("From date", date.today() - timedelta(days=365), min_value=date(1980, 1, 1), max_value=date.today() - timedelta(days=5)  , key='sd')
    end_train = st.date_input("To date", date.today(), min_value=start_train + timedelta(days=5), max_value=date.today(), key ='ed')
    if(end_train<=start_train):
        st.error('Error: End date must fall after start date.')
    else:
        data_train = yf.download(ticker, start_train, end_train)
        data_train.reset_index(inplace = True)
        df_train = data_train[['Date','Close']]
        df_train = df_train.rename(
            columns={
                'Date': 'ds',
                'Close': 'y'
                }
        )

    #Forecast model to predict future prices
    st.subheader('Select the prediction period')
    n_years = st.slider("Years of prediction", 1, 10)
    period = n_years * 365
    with st.spinner('Loading...'):
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods = period)
        forecast = m.predict(future)

        st.subheader("Forecast Data")
        forecast_actual = forecast.copy()
        forecast_actual = forecast_actual.rename(
            columns={
                'ds':'Date',
                'yhat':'Predicted Stock Price',
                'yhat_upper':'Predicted Upper Limit',
                'yhat_lower':'Predicted Lower Limit'}
                )
        st.write(forecast_actual[['Date','Predicted Stock Price','Predicted Lower Limit','Predicted Upper Limit']].tail())

        fig1 = plot_plotly(m,forecast)
        fig1.update_layout(xaxis_title = 'Time', yaxis_title = 'Closing Price')
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = m.plot_components(forecast)
        st.write(fig2)

#main function
if __name__ == "__main__":

    companies = pd.read_csv("./nasdaq_screener_1636450584038.csv") #dataframe to store nasdaq list of companies
    stocks = companies[['Symbol']] #array to extract symbols/tickers of companies

    #sidebar for menu
    with st.sidebar:
        st.subheader("Enter Details")
        selected_stock = st.selectbox("Select stock for prediction", stocks)
        Start = st.date_input("Start date", date(2000, 1, 1), min_value=date(1980, 1, 1), max_value=datetime.today())
        Today = st.date_input("End date", date.today(), min_value=date(1980, 1, 1), max_value=datetime.today())
        if(Today<=Start):
            st.error('Error: End date must fall after start date.')
        menu = st.selectbox("Menu Options",('Home','Company Profile','Descriptive Data','Technical Indicators','Prediction')) 
        data = load_data(selected_stock)

    #calling all defined functions based on user input of menu
    if(menu == "Home"):
        title()
    elif(menu == "Company Profile"):
        companyinfo(selected_stock)
    elif(menu == "Descriptive Data"):
        tabular_data()
        descriptive_stats()
        descriptive()
        rolling_std()
    elif(menu == "Technical Indicators"):
        plot_data() 
        candlestick()
        daily_returns()
        PlotMA(selected_stock)
        RSI()
    elif(menu == "Prediction"):
        forecasting(selected_stock)
        

        