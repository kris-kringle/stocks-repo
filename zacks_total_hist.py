import pandas as pd
import datetime as dt
import numpy as np
import os

filepath = os.getcwd()
csv_files_path = '\\csv_files\\'
csv_filepath = filepath + csv_files_path
rank1_history = pd.read_csv(os.path.join(csv_filepath, 'rank1_history.csv'), header=None)
rank2_history = pd.read_csv(os.path.join(csv_filepath, 'rank2_history.csv'), header=None)
rank3_history = pd.read_csv(os.path.join(csv_filepath, 'rank3_history.csv'), header=None)
rank4_history = pd.read_csv(os.path.join(csv_filepath, 'rank4_history.csv'), header=None)
rank5_history = pd.read_csv(os.path.join(csv_filepath, 'rank5_history.csv'), header=None)
rank_date = '2019-04-05'
current_rank = pd.read_csv(os.path.join(csv_filepath, 'zacks_custom_screen_' + rank_date + '.csv'), header=0)
current_rank = current_rank.replace('', '0')
current_rank = current_rank.replace(np.NaN, '0')
current_rank = current_rank[['Ticker', 'Zacks Rank']]
current_rank1 = current_rank[(current_rank['Zacks Rank'] == 1)]
current_rank1 = current_rank1.reset_index(drop=True)
current_rank2 = current_rank[(current_rank['Zacks Rank'] == 2)]
current_rank2 = current_rank2.reset_index(drop=True)
current_rank3 = current_rank[(current_rank['Zacks Rank'] == 3)]
current_rank3 = current_rank3.reset_index(drop=True)
current_rank4 = current_rank[(current_rank['Zacks Rank'] == 4)]
current_rank4 = current_rank4.reset_index(drop=True)
current_rank5 = current_rank[(current_rank['Zacks Rank'] == 5)]
current_rank5 = current_rank5.reset_index(drop=True)




# Rank 1
for i in range(0, len(rank1_history)):
    stock = str(rank1_history[0][i])
    if rank1_history[1][i] == 'open':
        # if history shows stock is open, check if it is in newest rank file
        result = stock in current_rank1['Ticker'].tolist()
        if result == True:
            # if the stock is in the newest rank file, keep open and update current date
            rank1_history[3][i] = rank_date
        else:
            # if the stock in not in the newest rank file, close it and update the closed date
            rank1_history[1][i] = 'closed'
            rank1_history[3][i] = rank_date
for j in range(0, len(current_rank1)):
    stock = str(current_rank1['Ticker'][j])
    check_hist = rank1_history.index[rank1_history[0]==stock].tolist()
    # if the stock is not on the history list, add it if it is in the newest rank list
    if len(check_hist) == 0:
        rank1_history = rank1_history.append(pd.Series([stock, 'open', rank_date, rank_date]), ignore_index=True)
    # if the stock is on the history list, check if it is closed
    else:
        # if the last instance is closed, add an instance of the stock
        if rank1_history[1][check_hist[len(check_hist)-1]] == 'closed':
            rank1_history = rank1_history.append(pd.Series([stock, 'open', rank_date, rank_date]), ignore_index=True)
np.savetxt(os.path.join(csv_filepath, "rank1_history.csv"), rank1_history, fmt=['%s', '%s', '%s', '%s'], delimiter=",")






# Rank 2
for i in range(0, len(rank2_history)):
    stock = str(rank2_history[0][i])
    if rank2_history[1][i] == 'open':
        # if history shows stock is open, check if it is in newest rank file
        result = stock in current_rank2['Ticker'].tolist()
        if result == True:
            # if the stock is in the newest rank file, keep open and update current date
            rank2_history[3][i] = rank_date
        else:
            # if the stock in not in the newest rank file, close it and update the closed date
            rank2_history[1][i] = 'closed'
            rank2_history[3][i] = rank_date
for j in range(0, len(current_rank2)):
    stock = str(current_rank2['Ticker'][j])
    check_hist = rank2_history.index[rank2_history[0]==stock].tolist()
    # if the stock is not on the history list, add it if it is in the newest rank list
    if len(check_hist) == 0:
        rank2_history = rank2_history.append(pd.Series([stock, 'open', rank_date, rank_date]), ignore_index=True)
    # if the stock is on the history list, check if it is closed
    else:
        # if the last instance is closed, add an instance of the stock
        if rank2_history[1][check_hist[len(check_hist)-1]] == 'closed':
            rank2_history = rank2_history.append(pd.Series([stock, 'open', rank_date, rank_date]), ignore_index=True)
np.savetxt(os.path.join(csv_filepath, "rank2_history.csv"), rank2_history, fmt=['%s', '%s', '%s', '%s'], delimiter=",")






# Rank 3
for i in range(0, len(rank3_history)):
    stock = str(rank3_history[0][i])
    if rank3_history[1][i] == 'open':
        # if history shows stock is open, check if it is in newest rank file
        result = stock in current_rank3['Ticker'].tolist()
        if result == True:
            # if the stock is in the newest rank file, keep open and update current date
            rank3_history[3][i] = rank_date
        else:
            # if the stock in not in the newest rank file, close it and update the closed date
            rank3_history[1][i] = 'closed'
            rank3_history[3][i] = rank_date
for j in range(0, len(current_rank3)):
    stock = str(current_rank3['Ticker'][j])
    check_hist = rank3_history.index[rank3_history[0]==stock].tolist()
    # if the stock is not on the history list, add it if it is in the newest rank list
    if len(check_hist) == 0:
        rank3_history = rank3_history.append(pd.Series([stock, 'open', rank_date, rank_date]), ignore_index=True)
    # if the stock is on the history list, check if it is closed
    else:
        # if the last instance is closed, add an instance of the stock
        if rank3_history[1][check_hist[len(check_hist)-1]] == 'closed':
            rank3_history = rank3_history.append(pd.Series([stock, 'open', rank_date, rank_date]), ignore_index=True)
np.savetxt(os.path.join(csv_filepath, "rank3_history.csv"), rank3_history, fmt=['%s', '%s', '%s', '%s'], delimiter=",")






# Rank 4
for i in range(0, len(rank4_history)):
    stock = str(rank4_history[0][i])
    if rank4_history[1][i] == 'open':
        # if history shows stock is open, check if it is in newest rank file
        result = stock in current_rank4['Ticker'].tolist()
        if result == True:
            # if the stock is in the newest rank file, keep open and update current date
            rank4_history[3][i] = rank_date
        else:
            # if the stock in not in the newest rank file, close it and update the closed date
            rank4_history[1][i] = 'closed'
            rank4_history[3][i] = rank_date
for j in range(0, len(current_rank4)):
    stock = str(current_rank4['Ticker'][j])
    check_hist = rank4_history.index[rank4_history[0]==stock].tolist()
    # if the stock is not on the history list, add it if it is in the newest rank list
    if len(check_hist) == 0:
        rank4_history = rank4_history.append(pd.Series([stock, 'open', rank_date, rank_date]), ignore_index=True)
    # if the stock is on the history list, check if it is closed
    else:
        # if the last instance is closed, add an instance of the stock
        if rank4_history[1][check_hist[len(check_hist)-1]] == 'closed':
            rank4_history = rank4_history.append(pd.Series([stock, 'open', rank_date, rank_date]), ignore_index=True)
np.savetxt(os.path.join(csv_filepath, "rank4_history.csv"), rank4_history, fmt=['%s', '%s', '%s', '%s'], delimiter=",")






# Rank 5
for i in range(0, len(rank5_history)):
    stock = str(rank5_history[0][i])
    if rank5_history[1][i] == 'open':
        # if history shows stock is open, check if it is in newest rank file
        result = stock in current_rank5['Ticker'].tolist()
        if result == True:
            # if the stock is in the newest rank file, keep open and update current date
            rank5_history[3][i] = rank_date
        else:
            # if the stock in not in the newest rank file, close it and update the closed date
            rank5_history[1][i] = 'closed'
            rank5_history[3][i] = rank_date
for j in range(0, len(current_rank5)):
    stock = str(current_rank5['Ticker'][j])
    check_hist = rank5_history.index[rank5_history[0]==stock].tolist()
    # if the stock is not on the history list, add it if it is in the newest rank list
    if len(check_hist) == 0:
        rank5_history = rank5_history.append(pd.Series([stock, 'open', rank_date, rank_date]), ignore_index=True)
    # if the stock is on the history list, check if it is closed
    else:
        # if the last instance is closed, add an instance of the stock
        if rank5_history[1][check_hist[len(check_hist)-1]] == 'closed':
            rank5_history = rank5_history.append(pd.Series([stock, 'open', rank_date, rank_date]), ignore_index=True)
np.savetxt(os.path.join(csv_filepath, "rank5_history.csv"), rank5_history, fmt=['%s', '%s', '%s', '%s'], delimiter=",")