# CryptoCoin_stock_trading
Automatic stock trading with cryptocoins based on AI

## Why
Blog in Slovak language https://linuxos.sk/blog/zumpa/detail/umela-inteligencia-obchodovanie-na-burze-s-kr/

## Installation
git clone https://github.com/bedna-KU/CryptoCoin_stock_trading.git

cd CryptoCoin_stock_trading

sudo apt install python3-dev

python3 -m pip install -r requirements.txt

## Files
    binanve/get_all_pairs.py           - Get all pairs (symbols)
    binance/save_historical_data.py    - Save historical data
    csv_columns.py                     - Extract specific columns from file.csv

## Get all pairs
python3 binance/get_all_pairs.py
## Get pairs with BTC
python3 binance/get_all_pairs.py btc
## Get pairs with BTC and USD
python3 binance/get_all_pairs.py btc usd
## Save historical data for DOGE/USDT from 1. Dec 2019 by 1minute interval
save_historical_data.py --symbol DOGEUSDT --start "1. Dec 2019" --interval 1m
## Filter 1. and 5. columns and show data
python3 csv_columns.py --action show --columns "0 4"
## Filter 1. 2. 3. 4. 5. and 6. columns (time + [ohlcv](https://www.kaiko.com/products/binance-ohlcv-trade-data)) and show data
python3 csv_columns.py --action show --columns "0 1 2 3 4 5"
## Filter 1. and 5. columns (time and close) and save data (file_new.csv)
python3 csv_columns.py --action save --columns "0 4"
## Filter 1. and 5. columns and show graph
python3 csv_columns.py --action graph --columns "0 4"

## Work in progress

## TODO
Download dataset               - DONE

Make Artificial Intelligence   - will be

Learn Artificial Intelligence  - will be

Connect on CryptoCoin stock    - will be

Automated trading with AI      - will be
