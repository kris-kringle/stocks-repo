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

my_gui.csv_path = csv_filepath

err = stock_params.hist_prices(my_gui.stock, my_gui.pull_date)
test_row, test_col = stock_params.short_norm_stock_df.shape
my_gui.market_performance = round(((stock_params.short_norm_stock_df['close'][test_row - 1] - stock_params.short_norm_stock_df['close'][test_row - 21]) / stock_params.short_norm_stock_df['close'][test_row - 21]) * 100,2)
if err == False:
    stock_gain = stock_params.plot_chart()
    my_gui.plot_fig_tab1()
    my_gui.plot_stock_gain_tab1(stock_gain)
    stock_params.spy_baseline = True
    last_date_index = len(stock_params.short_norm_stock_df)
    my_gui.pull_date = str(stock_params.short_norm_stock_df.index[last_date_index - 1])[0:10]
    my_gui.pull_list_date = str(stock_params.short_norm_stock_df.index[last_date_index - 1])[0:10]
    my_gui.update_pull_date_only()
    my_gui.update_pull_list_date_only()

old_stock = ""

while True:

    if my_gui.state == "pull":

        if my_gui.stock != old_stock:
            err = stock_params.hist_prices(my_gui.stock, my_gui.pull_date)
            if err == False:
                my_gui.stock_performance = round(((stock_params.short_norm_stock_df['close'][stock_params.row - 1] -stock_params.short_norm_stock_df['close'][stock_params.row - 21]) /stock_params.short_norm_stock_df['close'][stock_params.row - 21]) * 100, 2)
                stock_gain = stock_params.plot_chart()
                my_gui.plot_stock_gain_tab1(stock_gain)
                stock_params.zacks_total_params = my_gui.get_zacks_params
                my_gui.temp_stock_params = stock_params.get_zacks_params(my_gui.stock)
                my_gui.plot_fig_tab1()
                result = [stock_params.weekly_red_above_zero(stock_params.row - 1),
                          stock_params.daily_red_above_zero(stock_params.row - 1),
                          stock_params.price_above_EMA200(stock_params.row - 1),
                          stock_params.obv_volume_slope_up(stock_params.row - 1),
                          stock_params.EMA200_slope_up(stock_params.row - 1)]
                yest = 1
                while (result == [True, True,True]):  # or result == [False, True, True, True, True]):# or result == [True, True, True, False, True]):
                    result = [stock_params.weekly_red_above_zero(stock_params.row - 1 - yest),
                              stock_params.daily_red_above_zero(stock_params.row - 1 - yest),
                              stock_params.price_above_EMA200(stock_params.row - 1 - yest),
                              stock_params.obv_volume_slope_up(stock_params.row - 1 - yest),
                              stock_params.EMA200_slope_up(stock_params.row - 1 - yest)]
                    yest += 1
                print(yest)
            old_stock = my_gui.stock

    elif my_gui.state == "pull list":


        stock_params.zacks_total_params = my_gui.get_zacks_params
        if my_gui.csv_variable == "total_stock_list":
            stock_list = my_gui.get_zacks_params
        else:
            stock_list = pd.read_csv(os.path.join(csv_filepath, my_gui.csv_variable + '.csv'), header=0)
        end = my_gui.pull_list_date

        tested_stocks = pd.DataFrame()
        sell_stocks = pd.DataFrame()
        stock_params.hist_prices('aapl', my_gui.pull_list_date)
        reg_data = pd.DataFrame(index=range(0), columns=range(30))
        reg_data.columns = stock_params.stock_df.columns
        x = 0
        for x in range(0, len(stock_list)):
            stock = stock_list['Ticker'][x]
            print(stock)
            my_gui.pull_list_stock = stock
            zacks_array = stock_params.get_zacks_params(my_gui.pull_list_stock)
            zacks_above_zero = ((zacks_array[['% Change Q1 Est. (4 weeks)', '% Change Q2 Est. (4 weeks)', '% Change F1 Est. (4 weeks)','% Change F2 Est. (4 weeks)']]) >= 0).sum(axis=1)
            if zacks_above_zero[0] >= 2 and int(zacks_array['# of Analysts in Q1 Consensus'][0]) >= 3 and int(zacks_array['Zacks Rank'][0]) <= 3 and int(zacks_array['Zacks Industry Rank'][0]) <= 150 \
                    and (zacks_array['Value Score'][0] == 'A' or zacks_array['Value Score'][0] == 'B' or zacks_array['Value Score'][0] == 'C' or zacks_array['Value Score'][0] == 'D') \
                    and (zacks_array['Growth Score'][0] == 'A' or zacks_array['Growth Score'][0] == 'B' or zacks_array['Growth Score'][0] == 'C' or zacks_array['Growth Score'][0] == 'D') \
                    and (zacks_array['Momentum Score'][0] == 'A' or zacks_array['Momentum Score'][0] == 'B' or zacks_array['Momentum Score'][0] == 'C') \
                    and (zacks_array['VGM Score'][0] == 'A' or zacks_array['VGM Score'][0] == 'B'):
                # print("enter")
                err = stock_params.hist_prices(stock, my_gui.pull_list_date)
                if err == False:
                    stock_params.slope_crossover_history()
                    if stock_params.row > 200:
                        result = [stock_params.weekly_red_above_zero(stock_params.row - 1), stock_params.daily_red_above_zero(stock_params.row - 1), stock_params.obv_volume_slope_up(stock_params.row - 1)]#, stock_params.price_above_EMA200(stock_params.row - 1), stock_params.EMA200_slope_up(stock_params.row - 1)]
                        if my_gui.csv_variable == "total_stock_list" or my_gui.csv_variable == "zacks":
                            if (result == [True, True, True] or result == [False, True, True]):# or result == [True, True, True, False, True]):
                                yest = 1
                                firstResult = result
                                while (result == [True, True, True]):# or result == [False, True, True, True, True]):# or result == [True, True, True, False, True]):
                                    result = [stock_params.weekly_red_above_zero(stock_params.row - 1 - yest), stock_params.daily_red_above_zero(stock_params.row - 1 - yest) ,stock_params.price_above_EMA200(stock_params.row - 1 - yest), stock_params.obv_volume_slope_up(stock_params.row - 1 - yest), stock_params.EMA200_slope_up(stock_params.row - 1 - yest)]
                                    yest += 1
                                # print(yest)
                                if yest <= 20:
                                    if my_gui.pull_list_param_variable == "Inside envelope":
                                        envelope_num = 1.0
                                    elif my_gui.pull_list_param_variable == "Inside 50%":
                                        envelope_num = 0.50
                                    if stock_params.short_norm_stock_df['ema26'][stock_params.row - 1]*(1 - stock_params.env_percent*envelope_num) < stock_params.short_norm_stock_df['close'][stock_params.row - 1] <= stock_params.short_norm_stock_df['ema26'][stock_params.row - 1]*(1 + stock_params.env_percent*envelope_num):
                                        if stock_params.avg_volume >= 300000 and stock_params.env_percent > 0.04 and str(stock_params.short_norm_stock_df.index[stock_params.row - 1])[0:10] == my_gui.pull_list_date:
                                            tested_stocks = tested_stocks.append(pd.Series(stock), ignore_index=True)
                                            # zacks_array = stock_params.get_zacks_params(my_gui.pull_list_stock)
                                            # zacks_above_zero = ((zacks_array[['% Change Q1 Est. (4 weeks)', '% Change Q2 Est. (4 weeks)', '% Change F1 Est. (4 weeks)', '% Change F2 Est. (4 weeks)']]) >= 0).sum(axis=1)
                                            if zacks_above_zero[0] >= 2 and int(zacks_array['# of Analysts in Q1 Consensus'][0]) >= 3 and int(zacks_array['Zacks Rank'][0]) <= 3 \
                                                    and (   zacks_array['Value Score'][0] == 'A' or    zacks_array['Value Score'][0] == 'B' or    zacks_array['Value Score'][0] == 'C' or  zacks_array['Value Score'][0] == 'D') \
                                                    and (  zacks_array['Growth Score'][0] == 'A' or   zacks_array['Growth Score'][0] == 'B' or   zacks_array['Growth Score'][0] == 'C' or zacks_array['Growth Score'][0] == 'D') \
                                                    and (zacks_array['Momentum Score'][0] == 'A' or zacks_array['Momentum Score'][0] == 'B' or zacks_array['Momentum Score'][0] == 'C')\
                                                    and (     zacks_array['VGM Score'][0] == 'A' or      zacks_array['VGM Score'][0] == 'B' or zacks_array['VGM Score'][0] == 'C'):
                                                four_week_perf = round(((stock_params.short_norm_stock_df['close'][stock_params.row - 1] -stock_params.short_norm_stock_df['close'][stock_params.row - 21]) /stock_params.short_norm_stock_df['close'][stock_params.row - 21]) * 100, 2)
                                                my_gui.four_week_performance_dict.update({stock: four_week_perf})
                                                my_gui.pull_list_stock_zacks_params_dict.update({stock: zacks_array})
                                                my_gui.update_buy_list()
                                                stock_gain = stock_params.plot_chart()
                                                my_gui.pull_list_stock_gain_dict.update({stock: stock_gain})
                        else:
                            tested_stocks = tested_stocks.append(pd.Series(stock), ignore_index=True)
                            my_gui.update_buy_list()
                            four_week_perf = round(((stock_params.short_norm_stock_df['close'][stock_params.row - 1] -stock_params.short_norm_stock_df['close'][stock_params.row - 21]) /stock_params.short_norm_stock_df['close'][stock_params.row - 21]) * 100, 2)
                            my_gui.four_week_performance_dict.update({stock: four_week_perf})
                            stock_gain = stock_params.plot_chart()
                            zacks_array = stock_params.get_zacks_params(my_gui.pull_list_stock)
                            zacks_above_zero = ((zacks_array[['% Change Q1 Est. (4 weeks)', '% Change Q2 Est. (4 weeks)', 'Last EPS Surprise (%)','Earnings ESP']]) >= 0).sum(axis=1)
                            zacks_analysts = zacks_array['# of Analysts in Q1 Consensus'][0]
                            my_gui.pull_list_stock_zacks_params_dict.update({stock: zacks_array})
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