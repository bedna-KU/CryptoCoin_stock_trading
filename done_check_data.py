#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime

# Set input args
list_of_actions = ["show"]
list_of_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"]
ap = argparse.ArgumentParser ()
ap.add_argument("-a", "--action", type = str, required = True, choices = list_of_actions,
	help = "select action")
ap.add_argument("-f", "--file", type = str, required = True,
	help = "filename for read")
ap.add_argument ("-i", "--interval", type = str, required = True, choices=list_of_intervals,
	help = "interval")
args = vars(ap.parse_args())

# Interval to seconds
interval_seconds = {"1m" : 60, "3m" : 180, "5m" : 300, "15m" : 900, "30m" : 1800,
					"1h" : 3600, "2h" : 7200, "4h" : 14400, "6h" : 21600,
					"8h" : 28800, "12h" : 43200, "1d" : 86400, "3d" : 259200, "1w" : 604800}

# Args parse
action = args["action"]
filename = args["file"]
interval = args["interval"]

data = []
# Read CSV file into array
with open (filename, newline = "") as csvfile:
	reader = csv.reader (csvfile, delimiter = ',')
	for row in reader:
		data.append (row)

prev_item = int(data[0][0])
for item in data:
	if int(item[0]) - prev_item > interval_seconds[interval] * 1000:
		print(">>>", int(prev_item))
		print(">>>", int(item[0]))
		prev_unixtime = prev_item / 1000
		unixtime = int(item[0]) / 1000
		print(">>>", datetime.utcfromtimestamp(prev_unixtime).strftime('%Y-%m-%d %H:%M:%S'))
		print(">>>", datetime.utcfromtimestamp(unixtime).strftime('%Y-%m-%d %H:%M:%S'))
		delta = int(item[0]) - prev_item
		print(">>>", delta)
		value = int((delta / 1000) / interval_seconds[interval])
		print(">>> Time delta ({})".format(interval), value)
		print()
	prev_item = int(item[0])
