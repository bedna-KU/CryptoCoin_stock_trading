# NEWS
## 6. Feb 2021

## Change filenames nomenclature:

Finished solution prefix:
> done_

Prefix for work in progress:
> test_

Other (or without prefix) prefix is for test

## What works now:

## List all pairs from binance
> python3 binance/done_get_all_pairs.py

## Save historical data
> python3 binance/done_save_historical_data.py

## Data processing
> python3 done_csv_columns.py

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

## Work in progess

## :red_circle: Big data error :red_circle:
I tested the completeness of the data and found a few gaps there.
Is this bug or not?

For chceck data I write skript:
> python3 done_check_data.py --action show --file file_new.csv --interval 1m

Script check data for gaps every minute.
A very bad result is the missing data:
```
>>> 1581213540000
>>> 1581217200000
>>> 2020-02-09 01:59:00
>>> 2020-02-09 03:00:00
>>> 3660000
>>> Time delta (1m) 61

>>> 1582112100000
>>> 1582133400000
>>> 2020-02-19 11:35:00
>>> 2020-02-19 17:30:00
>>> 21300000
>>> Time delta (1m) 355

>>> 1583313660000
>>> 1583321400000
>>> 2020-03-04 09:21:00
>>> 2020-03-04 11:30:00
>>> 7740000
>>> Time delta (1m) 129

>>> 1587779940000
>>> 1587789000000
>>> 2020-04-25 01:59:00
>>> 2020-04-25 04:30:00
>>> 9060000
>>> Time delta (1m) 151

>>> 1593309540000
>>> 1593322200000
>>> 2020-06-28 01:59:00
>>> 2020-06-28 05:30:00
>>> 12660000
>>> Time delta (1m) 211

>>> 1606715940000
>>> 1606719600000
>>> 2020-11-30 05:59:00
>>> 2020-11-30 07:00:00
>>> 3660000
>>> Time delta (1m) 61

>>> 1608559680000
>>> 1608573600000
>>> 2020-12-21 14:08:00
>>> 2020-12-21 18:00:00
>>> 13920000
>>> Time delta (1m) 232

>>> 1608861540000
>>> 1608865200000
>>> 2020-12-25 01:59:00
>>> 2020-12-25 03:00:00
>>> 3660000
>>> Time delta (1m) 61
```
