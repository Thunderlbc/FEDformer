import requests
import datetime
import time
import csv

# Functions for working with time
def get_milliseconds(date_str):
    epoch = datetime.datetime.utcfromtimestamp(0)
    return int((datetime.datetime.strptime(date_str, "%Y-%m-%d") - epoch).total_seconds() * 1000.0)

# Function for getting data from Binance
def get_binance_data(symbol, start_str, end_str, interval):
    url = "https://api.binance.com/api/v3/klines"
    start_time = get_milliseconds(start_str)
    end_time = get_milliseconds(end_str)
    limit = 1000

    all_data = []

    while start_time < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
            'startTime': start_time,
            'endTime': end_time
        }
        print(url, params)
        response = requests.get(url, params=params, verify=False)
        if response.status_code == 200:
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            start_time = data[-1][0] + 1
        else:
            break

        time.sleep(0.5) # Delay to prevent exceeding request limits

    return all_data

# Binance timeframes
timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

# Retrieving and saving data for each timeframe
#for timeframe in timeframes:
#    data = get_binance_data('BTCUSDT', '2025-01-01', '2025-01-21', timeframe)
#    filename = f'btcusdt_{timeframe}_2025.csv'
#
#    with open(filename, 'w', newline='') as file:
#        writer = csv.writer(file)
#        # Add headers
#        headers = [
#            'Open_time',
#            'Open_price',
#            'High_price',
#            'Low_price',
#            'Close_price',
#            'Volume',
#            'Close_time',
#            'Quote_asset_volume',
#            'Number_of_trades',
#            'Taker_buy_base_volume',
#            'Taker_buy_quote_volume',
#            'Ignore'
#        ]
#        writer.writerow(headers)
#        for row in data:
#            writer.writerow(row)
#    
#    print(f"Data saved in {filename}")
#
#print("All data has been successfully collected and saved.")

"""
BTCUSDT_1m_2025.csv文件包含了2025年1月1日到2025年1月21日的BTCUSDT的1分钟K线数据
     他的每一列分别是： Open_time, Open_price, High_price, Low_price, Close_price, Volume, Close_time, Quote_asset_volume, Number_of_trades, Taker_buy_base_volume, Taker_buy_quote_volume, Ignork
    基于这些数据，我们生成：
    1. delta_open_price_ratio: 开盘价与前一个开盘价的差值比
    2. delta_high_price_ratio: 最高价与前一个最高价的差值比
    3. delta_low_price_ratio: 最低价与前一个最低价的差值比
    4. delta_close_price_ratio: 收盘价与前一个收盘价的差值比
    5. delta_volume_ratio: 成交量与前一个成交量的差值比
    6. delta_quote_asset_volume_ratio: 成交额与前一个成交额的差值比
    7. delta_number_of_trades_ratio: 成交笔数与前一个成交笔数的差值比
    8. delta_taker_buy_base_volume_ratio: 主动买入成交量与前一个主动买入成交量的差值比
    9. delta_taker_buy_quote_volume_ratio: 主动买入成交额与前一个主动买入成交额的差值比
"""
import pandas as pd
for timeframe in timeframes:
    filename = f'btcusdt_{timeframe}_2025.csv'
    df = pd.read_csv(filename)
    df['delta_open_price_ratio'] = df['Open_price'].diff() / df['Open_price'].shift(1)
    df['delta_high_price_ratio'] = df['High_price'].diff() / df['High_price'].shift(1)
    df['delta_low_price_ratio'] = df['Low_price'].diff() / df['Low_price'].shift(1)
    df['delta_close_price_ratio'] = df['Close_price'].diff() / df['Close_price'].shift(1)
    df['delta_volume_ratio'] = df['Volume'].diff() / df['Volume'].shift(1)
    df['delta_quote_asset_volume_ratio'] = df['Quote_asset_volume'].diff() / df['Quote_asset_volume'].shift(1)
    df['delta_number_of_trades_ratio'] = df['Number_of_trades'].diff() / df['Number_of_trades'].shift(1)
    df['delta_taker_buy_base_volume_ratio'] = df['Taker_buy_base_volume'].diff() / df['Taker_buy_base_volume'].shift(1)
    df['delta_taker_buy_quote_volume_ratio'] = df['Taker_buy_quote_volume'].diff() / df['Taker_buy_quote_volume'].shift(1)
    # 1: 是扔掉第一行，因为第一行没有前一行
    df[1:].to_csv(f'btcusdt_{timeframe}_2025_delta.csv', index=False)