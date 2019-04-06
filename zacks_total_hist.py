import pandas as pd
import datetime as dt
import numpy as np
import os

filepath = os.getcwd()
csv_files_path = '\\csv_files\\'
csv_filepath = filepath + csv_files_path
history = pd.read_csv(os.path.join(csv_filepath, 'Top Stocks History.csv'), header=None)
rank_date = '2019-04-03'
current_rank = pd.read_csv(os.path.join(csv_filepath, 'zacks_custom_screen_' + rank_date + '.csv'), header=None)

for i in range(0, len(history)):
    stock = str(history[0][i])

    if history[1][i] == 'open':

        # if history shows stock is open, check if it is in newest rank file
        result = stock in current_rank[0].tolist()
        if result == True:
            # if the stock is in the newest rank file, keep open and update current date
            history[3][i] = rank_date

        else:
            # if the stock in not in the newest rank file, close it and update the closed date
            history[1][i] = 'closed'
            history[3][i] = rank_date

for j in range(0, len(current_rank)):

    stock = str(current_rank[0][j])
    check_hist = history.index[history[0]==stock].tolist()

    # if the stock is not on the history list, add it if it is in the newest rank list
    if len(check_hist) == 0:
        history = history.append(pd.Series([stock, 'open', rank_date, rank_date]), ignore_index=True)

    # if the stock is on the history list, check if it is closed
    else:
        # if the last instance is closed, add an instance of the stock
        if history[1][check_hist[len(check_hist)-1]] == 'closed':
            history = history.append(pd.Series([stock, 'open', rank_date, rank_date]), ignore_index=True)


np.savetxt(os.path.join(csv_filepath, "Top Stocks History.csv"), history, fmt=['%s', '%s', '%s', '%s'], delimiter=",")