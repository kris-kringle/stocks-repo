import tkinter as tk
import pandas as pd
import datetime as dt
import matplotlib
matplotlib.use('TkAgg')
import os
import processStockData as psd
import stockGui
import shutil

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

err = stock_params.hist_prices(my_gui.stock, my_gui.pull_date)
if err == False:
    stock_gain = stock_params.plot_chart()
    my_gui.plot_fig_tab1()
    my_gui.plot_stock_gain_tab1(stock_gain)
    my_gui.trend_status = stock_params.check_trends(stock_params.row)

old_stock = ""

while True:

    current_time = dt.datetime.now()
    if current_time.hour == 16 and current_time.minute == 1:
        my_gui.pull_list_date_entry.delete(0, tk.END)
        my_gui.pull_list_date_entry.insert(tk.END, str(dt.datetime(current_time.year, current_time.month, current_time.day))[0:10])
        my_gui.update_pull_list_date("yes")

    if my_gui.state == "pull":

        if my_gui.stock != old_stock:
            # print("stock")
            err = stock_params.hist_prices(my_gui.stock, my_gui.pull_date)
            # print(err)
            if err == False:
                # print("stock")
                stock_gain = stock_params.plot_chart()
                my_gui.plot_stock_gain_tab1(stock_gain)
                my_gui.plot_fig_tab1()
                # print("stock")
                my_gui.trend_status = stock_params.check_trends(stock_params.row)
                print(my_gui.trend_status)
                result = [stock_params.weekly_red_above_zero(stock_params.row - 1),
                          stock_params.daily_red_above_zero(stock_params.row - 1),
                          stock_params.price_above_EMA200(stock_params.row - 1),
                          stock_params.obv_volume_slope_up(stock_params.row - 1),
                          stock_params.EMA200_slope_up(stock_params.row - 1)]
                print("result", result)
            old_stock = my_gui.stock

    elif my_gui.state == "pull list":

        criteria = "weekly slope crossover up"
        stock_list = pd.read_csv(os.path.join(csv_filepath, my_gui.csv_variable + '.csv'), header=None)
        end = my_gui.pull_list_date

        tested_stocks = pd.DataFrame()
        sell_stocks = pd.DataFrame()
        stock_params.hist_prices('aapl', my_gui.pull_list_date)
        reg_data = pd.DataFrame(index=range(0), columns=range(30))
        reg_data.columns = stock_params.stock_df.columns
        x = 0
        for x in range(0, len(stock_list)):
            stock = stock_list[0][x]
            # print("1", stock)
            err = stock_params.hist_prices(stock, my_gui.pull_list_date)
            # print("2", stock)
            if err == False:
                stock_params.slope_crossover_history()
                # print("3", stock)
                my_gui.pull_list_stock = stock
                if stock_params.row > 200:
                    # result = stock_params.weekly_red_slope_crossover_zero(stock_params.row - 1)
                    result = [stock_params.weekly_red_above_zero(stock_params.row - 1), stock_params.daily_red_above_zero(stock_params.row - 1), stock_params.price_above_EMA200(stock_params.row - 1), stock_params.obv_volume_slope_up(stock_params.row - 1), stock_params.EMA200_slope_up(stock_params.row - 1)]
                    print("result", result)
                    if my_gui.csv_variable == "total_stock_list" or my_gui.csv_variable == "zacks":
                        print(stock, str(stock_params.short_norm_stock_df.index[stock_params.row - 1])[0:10])
                        # if (result == [True, True, True, True, True] or result == [True, True, True, False, True] or result == [False, True, True, True, True]):
                        if (result == [True, True, True, True, True]):
                            yest = 1
                            firstResult = result
                            # while (result == [True, True, True, True, True] or result == [True, True, True, False, True] or result == [False, True, True, True, True]):
                            while (result == [True, True, True, True, True]):
                                result = [stock_params.weekly_red_above_zero(stock_params.row - 1 - yest), stock_params.daily_red_above_zero(stock_params.row - 1 - yest) ,stock_params.price_above_EMA200(stock_params.row - 1 - yest), stock_params.obv_volume_slope_up(stock_params.row - 1 - yest), stock_params.EMA200_slope_up(stock_params.row - 1 - yest)]
                                yest += 1
                            print(yest, stock_params.short_norm_stock_df.index[stock_params.row - 1], firstResult)
                            if yest <= 20 and yest >= 2:
                                if stock_params.short_norm_stock_df['ema26_lowenv'][stock_params.row - 1] < stock_params.short_norm_stock_df['close'][stock_params.row - 1] <= stock_params.short_norm_stock_df['ema26_highenv'][stock_params.row - 1]:
                                    if stock_params.avg_volume >= 500000 and stock_params.env_percent > 0.04 and str(stock_params.short_norm_stock_df.index[stock_params.row - 1])[0:10] == my_gui.pull_list_date:
                                        tested_stocks = tested_stocks.append(pd.Series(stock), ignore_index=True)
                                        my_gui.update_buy_list()
                                        stock_gain = stock_params.plot_chart()
                                        my_gui.pull_list_trend_dict.update({stock: stock_params.check_trends(stock_params.row)})
                                        # my_gui.pull_list_dict.update({stock:fig})
                                        my_gui.pull_list_stock_gain_dict.update({stock: stock_gain})
                                        print("possible buy", len(tested_stocks))
                    else:
                        tested_stocks = tested_stocks.append(pd.Series(stock), ignore_index=True)
                        my_gui.update_buy_list()
                        stock_gain = stock_params.plot_chart()
                        my_gui.pull_list_trend_dict.update({stock: stock_params.check_trends(stock_params.row)})
                        # my_gui.pull_list_dict.update({stock:fig})
                        my_gui.pull_list_stock_gain_dict.update({stock: stock_gain})
                        print("possible buy", len(tested_stocks))

            # print("here")
            my_gui.update_pull_list_status()
            root.update_idletasks()
            root.update()

        my_gui.state = ""
        my_gui.pull_list_stock = "D-O-N-E"
        my_gui.update_pull_list_status()

        # -------------------

    my_gui.update_every_time()
    root.update_idletasks()
    root.update()

shutil.rmtree(pics_filepath)