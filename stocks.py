import tkinter as tk
import pandas as pd
import datetime as dt
import matplotlib
matplotlib.use('TkAgg')
import os
import processStockData as psd
import stockGui
import shutil
import numpy as np

filepath = os.getcwd()
csv_files_path = '\\csv_files\\'
csv_filepath = filepath + csv_files_path

pics_filepath = filepath + '\\pics\\'

if os.path.isdir(pics_filepath) == True:
    shutil.rmtree(pics_filepath)

os.mkdir('pics')

root = tk.Tk()
my_gui = stockGui.gui(root)
my_gui.pics_filepath = pics_filepath

stock_params = psd.stock_data()
stock_params.pics_filepath = pics_filepath

stock_params.rank1_hist = pd.read_csv(os.path.join(csv_filepath, 'rank1_history.csv'), header=None)
stock_params.rank2_hist = pd.read_csv(os.path.join(csv_filepath, 'rank2_history.csv'), header=None)
stock_params.rank3_hist = pd.read_csv(os.path.join(csv_filepath, 'rank3_history.csv'), header=None)
stock_params.rank4_hist = pd.read_csv(os.path.join(csv_filepath, 'rank4_history.csv'), header=None)
stock_params.rank5_hist = pd.read_csv(os.path.join(csv_filepath, 'rank5_history.csv'), header=None)

stock_params.zacks_total_params = pd.read_csv(os.path.join(csv_filepath, 'Zacks Total Params.csv'), header=0)
stock_params.zacks_total_params = stock_params.zacks_total_params.replace('', '0')
stock_params.zacks_total_params = stock_params.zacks_total_params.replace(np.NaN, '0')

err = stock_params.hist_prices(my_gui.stock, my_gui.pull_date)
test_row, test_col = stock_params.short_norm_stock_df.shape
my_gui.market_performance = round(((stock_params.short_norm_stock_df['close'][test_row - 1] - stock_params.short_norm_stock_df['close'][test_row - 21]) / stock_params.short_norm_stock_df['close'][test_row - 21]) * 100,2)
if err == False:
    stock_gain = stock_params.plot_chart()
    my_gui.plot_fig_tab1()
    my_gui.plot_stock_gain_tab1(stock_gain)

old_stock = ""

while True:

    # current_time = dt.datetime.now()
    # if current_time.hour == 16 and current_time.minute == 1:
    #     my_gui.pull_list_date_entry.delete(0, tk.END)
    #     my_gui.pull_list_date_entry.insert(tk.END, str(dt.datetime(current_time.year, current_time.month, current_time.day))[0:10])
    #     my_gui.update_pull_list_date("yes")

    if my_gui.state == "pull":

        if my_gui.stock != old_stock:
            err = stock_params.hist_prices(my_gui.stock, my_gui.pull_date)
            if err == False:
                # stock_params.zacks_hist = pd.read_csv(os.path.join(csv_filepath, 'Top Stocks History.csv'), header=None)
                my_gui.stock_performance = round(((stock_params.short_norm_stock_df['close'][stock_params.row - 1] -stock_params.short_norm_stock_df['close'][stock_params.row - 21]) /stock_params.short_norm_stock_df['close'][stock_params.row - 21]) * 100, 2)
                stock_gain = stock_params.plot_chart()
                my_gui.plot_stock_gain_tab1(stock_gain)
                my_gui.temp_stock_params = stock_params.get_zacks_params(my_gui.stock)
                my_gui.plot_fig_tab1()
                result = [stock_params.weekly_red_above_zero(stock_params.row - 1),
                          stock_params.daily_red_above_zero(stock_params.row - 1),
                          stock_params.price_above_EMA200(stock_params.row - 1),
                          stock_params.obv_volume_slope_up(stock_params.row - 1),
                          stock_params.EMA200_slope_up(stock_params.row - 1)]
            old_stock = my_gui.stock

    elif my_gui.state == "pull list":

        criteria = "weekly slope crossover up"
        stock_list = pd.read_csv(os.path.join(csv_filepath, my_gui.csv_variable + '.csv'), header=None)
        # stock_params.zacks_hist = pd.read_csv(os.path.join(csv_filepath, 'Top Stocks History.csv'), header=None)
        end = my_gui.pull_list_date

        tested_stocks = pd.DataFrame()
        sell_stocks = pd.DataFrame()
        stock_params.hist_prices('aapl', my_gui.pull_list_date)
        reg_data = pd.DataFrame(index=range(0), columns=range(30))
        reg_data.columns = stock_params.stock_df.columns
        x = 0
        for x in range(0, len(stock_list)):
            stock = stock_list[0][x]
            err = stock_params.hist_prices(stock, my_gui.pull_list_date)
            if err == False:
                stock_params.slope_crossover_history()
                my_gui.pull_list_stock = stock
                if stock_params.row > 200:
                    result = [stock_params.weekly_red_above_zero(stock_params.row - 1), stock_params.daily_red_above_zero(stock_params.row - 1), stock_params.price_above_EMA200(stock_params.row - 1), stock_params.obv_volume_slope_up(stock_params.row - 1), stock_params.EMA200_slope_up(stock_params.row - 1)]
                    if my_gui.csv_variable == "total_stock_list" or my_gui.csv_variable == "zacks":
                        if (result == [True, True, True, True, True] or result == [True, True, True, False, True] or result == [False, True, True, True, True]):
                            yest = 1
                            firstResult = result
                            while (result == [True, True, True, True, True] or result == [True, True, True, False, True] or result == [False, True, True, True, True]):
                                result = [stock_params.weekly_red_above_zero(stock_params.row - 1 - yest), stock_params.daily_red_above_zero(stock_params.row - 1 - yest) ,stock_params.price_above_EMA200(stock_params.row - 1 - yest), stock_params.obv_volume_slope_up(stock_params.row - 1 - yest), stock_params.EMA200_slope_up(stock_params.row - 1 - yest)]
                                yest += 1
                            if yest >= 1:
                                if stock_params.short_norm_stock_df['ema26'][stock_params.row - 1]*(1 - stock_params.env_percent*.6) < stock_params.short_norm_stock_df['close'][stock_params.row - 1] <= stock_params.short_norm_stock_df['ema26'][stock_params.row - 1]*(1 + stock_params.env_percent*.6):
                                    if stock_params.avg_volume >= 500000 and stock_params.env_percent > 0.04 and str(stock_params.short_norm_stock_df.index[stock_params.row - 1])[0:10] == my_gui.pull_list_date:
                                        tested_stocks = tested_stocks.append(pd.Series(stock), ignore_index=True)
                                        zacks_array = stock_params.get_zacks_params(my_gui.pull_list_stock)
                                        zacks_above_zero = ((zacks_array[['% Change Q1 Est. (4 weeks)', '% Change Q2 Est. (4 weeks)', 'Last EPS Surprise (%)', 'Earnings ESP']]) >= 0).sum(axis=1)
                                        zacks_analysts = zacks_array['# of Analysts in Q1 Consensus'][0]
                                        if zacks_above_zero[0] >= 3 and zacks_analysts >= 3:
                                            four_week_perf = round(((stock_params.short_norm_stock_df['close'][stock_params.row - 1] -stock_params.short_norm_stock_df['close'][stock_params.row - 21]) /stock_params.short_norm_stock_df['close'][stock_params.row - 21]) * 100, 2)
                                            print(four_week_perf)
                                            my_gui.four_week_performance_dict.update({stock: four_week_perf})
                                            my_gui.pull_list_stock_zacks_params_dict.update({stock: zacks_array})
                                            print(my_gui.four_week_performance_dict)
                                            my_gui.update_buy_list()
                                            stock_gain = stock_params.plot_chart()
                                            my_gui.pull_list_stock_gain_dict.update({stock: stock_gain})
                    else:
                        tested_stocks = tested_stocks.append(pd.Series(stock), ignore_index=True)
                        my_gui.update_buy_list()
                        four_week_perf = round(((stock_params.short_norm_stock_df['close'][stock_params.row - 1] -stock_params.short_norm_stock_df['close'][stock_params.row - 21]) /stock_params.short_norm_stock_df['close'][stock_params.row - 21]) * 100, 2)
                        print(four_week_perf)
                        my_gui.four_week_performance_dict.update({stock: four_week_perf})
                        print(my_gui.four_week_performance_dict)
                        stock_gain = stock_params.plot_chart()
                        my_gui.pull_list_stock_zacks_params_dict.update({stock: stock_params.get_zacks_params(stock)})
                        my_gui.pull_list_stock_gain_dict.update({stock: stock_gain})

            my_gui.update_pull_list_status()
            root.update_idletasks()
            root.update()

        my_gui.state = ""
        my_gui.pull_list_stock = "D-O-N-E"
        my_gui.update_pull_list_status()

    elif my_gui.state == "back test":

        print("backtest")

        backtest_list = pd.read_csv(os.path.join(csv_filepath, 'backtest.csv'), header=None)
        stock_params.hist_prices('aapl', my_gui.pull_date)

        x = 0
        for x in range(0, len(backtest_list)):
            stock = backtest_list[0][x]
            err = stock_params.hist_prices(stock, my_gui.pull_date)
            if err == False:
                stock_params.backtest()

        print(stock_params.backtest_df)
        np.savetxt(os.path.join(csv_filepath, "save_backtest.csv"), stock_params.backtest_df, fmt=['%s', '%s', '%s', '%s'], delimiter=",")


        my_gui.state = ""


        # -------------------

    my_gui.update_every_time()
    root.update_idletasks()
    root.update()

shutil.rmtree(pics_filepath)