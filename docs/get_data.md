# Get data

## Get all pairs
python3 binance/done_get_all_pairs.py

## Get pairs with BTC
python3 binance/get_all_pairs.py btc

## Get pairs with BTC and USD
python3 binance/get_all_pairs.py btc usd

## Save historical data for DOGE/USDT from 1. Dec 2019 by 1minute interval
python3 binance/save_historical_data.py --symbol DOGEUSDT --start "1. Dec 2019" --interval 1m

## Filter 1. and 5. columns and show data
python3 csv_columns.py --action show --columns "0 4"

## Filter 1. 2. 3. 4. 5. and 6. columns (time + [ohlcv](https://www.kaiko.com/products/binance-ohlcv-trade-data)) and show data
python3 csv_columns.py --action show --columns "0 1 2 3 4 5"

## Filter 1. and 5. columns (time and close) and save data (file_new.csv)
python3 csv_columns.py --action save --columns "0 4"

## Filter 1. and 5. columns and show graph
python3 csv_columns.py --action graph --columns "0 4"