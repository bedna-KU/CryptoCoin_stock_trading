#!/usr/bin/env python3
from binance.client import Client
import sys

binance_client = Client()
prices = binance_client.get_all_tickers ()

if len(sys.argv) == 2:
    filter = sys.argv[1].upper()
    for pair in prices:
        if filter in pair["symbol"]:
            print (pair["symbol"])
elif len(sys.argv) == 3:
    filter = sys.argv[1].upper()
    filter2 = sys.argv[2].upper()
    for pair in prices:
        if filter in pair["symbol"]: 
            string = pair["symbol"]
            if filter2 in string: 
                print(string)
else:
    for pair in prices:
        print (pair["symbol"])