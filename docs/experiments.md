## Download/Train/Predict DOGE on DOGE closing price (Indian fairy tales)
```
python3 binance/done_save_historical_data.py --symbol DOGEUSDT --start "1. Dec 2019" --interval 1m
python3 done_csv_columns.py --action save --columns "0 1 2 3 4 5"
python3 done_train.py
python3 done_predict.py
```
You have to wait 10 minutes for the result

result ~75%

(See Big data error below)

## Download/Train/Predict DOGE on DOGE and BTC closing price
```
python3 binance/done_download_doge_btc.py
python3 done_train_doge_btc.py
python3 done_predict_doge_btc.py

```
The result is always bad, but interesting

## Download/Train/Predict DOGE on DOGE closing price and volume
```
python3 binance/done_save_historical_data.py --symbol DOGEUSDT --start "7. Jan 2020" --interval 1m --file file_doge_year_raw.csv
python3 done_csv_columns.py --action save --columns "0 1 2 3 4 5" --input file_doge_year_raw.csv --output file_doge_year.csv
python3 done_train_cv.py
python3 done_predict_cv.py
```
The result is always bad