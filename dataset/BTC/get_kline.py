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
for timeframe in timeframes:
    data = get_binance_data('BTCUSDT', '2025-01-01', '2025-01-21', timeframe)
    filename = f'btcusdt_{timeframe}_2025.csv'

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Add headers
        headers = [
            'Open_time',
            'Open_price',
            'High_price',
            'Low_price',
            'Close_price',
            'Volume',
            'Close_time',
            'Quote_asset_volume',
            'Number_of_trades',
            'Taker_buy_base_volume',
            'Taker_buy_quote_volume',
            'Ignore'
        ]
        writer.writerow(headers)
        for row in data:
            writer.writerow(row)
    
    print(f"Data saved in {filename}")

print("All data has been successfully collected and saved.")
