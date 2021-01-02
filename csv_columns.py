#!/usr/bin/env python3
import csv
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as md
from math import log
from datetime import datetime
import numpy as np

def prRed(skk): print("\033[91m{}\033[00m" .format(skk)) 

# Set input args
list_of_actions = ["show", "save", "graph"]
ap = argparse.ArgumentParser ()
ap.add_argument("-a", "--action", type = str, required = True, choices = list_of_actions,
	help = "select action")
ap.add_argument("-c", "--columns", type = str, required = True,
	help = "select columns")
args = vars(ap.parse_args())

# Args parse
action = args["action"]
columns = args["columns"]

# Convert columns to integers list
columns_list = columns.split()
columns_list_int = list(map(int, columns_list))

# Read CSV file
with open('file.csv', newline = '') as f:
    reader = csv.reader(f)
    data_list = list(reader)

# Count of columns
columns_count = len(data_list[0])

# Check if column is out of range
for item in columns_list_int:
    if item > columns_count - 1 or item < -columns_count:
        prRed(f"Column {str(item)} is out of range")
        exit("End with error")

# Extract only specific columns
data_list_columns = [[each_list[i] for i in columns_list_int] for each_list in data_list]

# Show output on std
if action == "show":
    for item in data_list_columns:
        for subitem in item:
            print(subitem, end = "  ")
        print()

# Save file to new CSV file_new.csv
if action == "save":
    with open('file_new.csv', 'w') as f:
        write = csv.writer(f) 
        write.writerows(data_list_columns)
        print("Rows:", len(data_list_columns))
        print("CSV file saved!")
# Show garph
if action == "graph":
    x_list = []
    y_list = []
    for s in data_list_columns:
        x_list.append (float(s[0]) / 1000)
        y_list.append (float(s[1]))
    
    dates = [datetime.fromtimestamp(ts) for ts in x_list]
    plt.subplots_adjust(bottom = 0.2)
    plt.xticks(rotation = 25)
    ax = plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)

    plt.plot(dates, y_list)
    
    plt.show()