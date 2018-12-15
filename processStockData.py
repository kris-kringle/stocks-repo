import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
import numpy as np
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import csv
from decimal import *
from datetime import datetime
from pandas import *
from mpl_finance import *
import matplotlib.dates as mdates
from matplotlib.dates import MONDAY
from matplotlib.dates import *
import datetime as dt
from random import *
import iexfinance as iex
from iexfinance import Stock
from dateutil import parser

class stock_data:

    def __init__(self):
        self.now = dt.datetime.now()
        self.start = datetime(self.now.year - 4, 1, 1)
        # self.end = datetime(self.now.year, self.now.month, self.now.day)
        self.pull_years = 2
        self.row = 0
        self.stock = ""
        self.data_years = 0
        self.stock_df = pd.DataFrame()
        self.short_stock_df = pd.DataFrame()
        self.short_norm_stock_df = pd.DataFrame()

    def hist_prices(self, stock, end):

        self.stock = stock.upper()
        try:
            self.stock_df = data.DataReader(self.stock, 'iex', self.start, end)
        except:
            print("ignore this stock 1---------------" + str(self.stock))
            return True

        self.row, col = self.stock_df.shape
        if self.row < 100:
            print("not enough data" + str(self.stock))
            return True
        if self.stock_df.index[self.row - 1] != str(end)[0:10]:
            stock_quote = iex.Stock(self.stock)
            try:
                stock_pd = stock_quote.get_quote()
            except:
                print("ignore todays prices 2----------" + str(self.stock))
                return True
            # raise
            # print(stock_pd)
            stock_date = stock_pd['latestTime']
            stock_date = str(parser.parse(stock_date))[0:10]
            #stock_date = pd.to_datetime(stock_date)[0:10]
            #print(stock_date, end)
            if stock_date == end:
                #print("here234")
                stock_open = stock_pd['open']
                stock_high = stock_pd['high']
                stock_low = stock_pd['low']
                stock_close = stock_pd['close']
                stock_volume = stock_pd['latestVolume']
                if stock_volume == None:
                    stock_volume = self.stock_df['volume'].mean()
                    print("here")
                f_day = {'open': [stock_open], 'high': [stock_high], 'low': [stock_low], 'close': [stock_close],
                         'volume': [stock_volume]}
                date_index = pd.date_range(stock_date, periods=1, freq='D')
                f_day = pd.DataFrame(data=f_day, index=date_index)
                f_day.index.name = 'date'
                self.stock_df = self.stock_df.append(f_day.ix[0])

        self.row, col = self.stock_df.shape

        self.stock_df = self.stock_df.set_index(pd.to_datetime(self.stock_df.index))
        self.data_years = self.row - int(round(253 * self.pull_years, 0))
        self.stock_df['date'] = self.stock_df.index.map(mdates.date2num)

        # calculate MACD parameters
        self.stock_df['ema26'] = self.stock_df['close'].ewm(span=26, adjust=False).mean()
        self.stock_df['ema12'] = self.stock_df['close'].ewm(span=12, adjust=False).mean()
        self.stock_df['daily_black'] = self.stock_df['ema12'] - self.stock_df['ema26']
        self.stock_df['daily_red'] = self.stock_df['daily_black'].ewm(span=18, adjust=False).mean()
        self.stock_df['ema60'] = self.stock_df['close'].ewm(span=60, adjust=False).mean()
        self.stock_df['ema130'] = self.stock_df['close'].ewm(span=130, adjust=False).mean()
        self.stock_df['weekly_black'] = self.stock_df['ema60'] - self.stock_df['ema130']
        self.stock_df['weekly_red'] = self.stock_df['weekly_black'].ewm(span=45, adjust=False).mean()

        env_percent = .04
        self.stock_df['fit_env'] = 0
        while self.stock_df['fit_env'].mean() <= .9:
            self.stock_df['ema26_highenv'] = self.stock_df['ema26'] * (1 + env_percent)
            self.stock_df['ema26_lowenv'] = self.stock_df['ema26'] * (1 - env_percent)
            self.stock_df['fit_env'] = np.where((self.stock_df['ema26_highenv'] >= self.stock_df['high']) &
                                                (self.stock_df['ema26_lowenv'] <= self.stock_df['low']), 1, 0)
            env_percent += .005

        pd.options.mode.chained_assignment = None
        self.stock_df['obv_volume'] = 0
        self.stock_df['obv_volume'][0] = self.stock_df['volume'][0]
        for i in range(1, self.row):
            if self.stock_df['close'][i] > self.stock_df['close'][i - 1]:
                self.stock_df['obv_volume'][i] = self.stock_df['obv_volume'][i - 1] + self.stock_df['volume'][i]
            elif self.stock_df['close'][i] < self.stock_df['close'][i - 1]:
                self.stock_df['obv_volume'][i] = self.stock_df['obv_volume'][i - 1] - self.stock_df['volume'][i]
            else:
                self.stock_df['obv_volume'][i] = self.stock_df['obv_volume'][i - 1]

        self.stock_df['volume'] = self.stock_df['volume'] / self.stock_df['volume'].max()
        self.stock_df['obv_volume'] = self.stock_df['obv_volume'] / self.stock_df['obv_volume'].max()

        avg_volume = self.stock_df['volume'][self.row - 10:].mean()
        self.stock_df['close_from_top_env'] = self.stock_df['ema26_highenv'] - self.stock_df['close']
        if self.row > 5:
            self.stock_df['weekly_black_deriv'] = np.gradient(self.stock_df['weekly_black'])
            self.stock_df['weekly_black_deriv'] = self.stock_df['weekly_black_deriv'].ewm(span=8, adjust=False).mean()
            self.stock_df['weekly_black_deriv'] = self.stock_df['weekly_black_deriv'] / self.stock_df['weekly_black_deriv'].max()
            self.stock_df['weekly_red_deriv'] = np.gradient(self.stock_df['weekly_red'])
            self.stock_df['weekly_red_deriv'] = self.stock_df['weekly_red_deriv'].ewm(span=8, adjust=False).mean()
            self.stock_df['weekly_red_deriv'] = self.stock_df['weekly_red_deriv'] / self.stock_df['weekly_red_deriv'].max()

        self.stock_df['decision'] = 0
        self.stock_df['stock-name'] = stock
        self.stock_df['performance'] = 0


        self.short_stock_df = self.stock_df[self.data_years:]
        self.short_norm_stock_df = self.stock_df[self.data_years:]
        self.row, col = self.short_norm_stock_df.shape
        self.short_norm_stock_df['open'] = self.short_norm_stock_df['open'] / self.short_norm_stock_df['open'].max()
        self.short_norm_stock_df['high'] = self.short_norm_stock_df['high'] / self.short_norm_stock_df['high'].max()
        self.short_norm_stock_df['low'] = self.short_norm_stock_df['low'] / self.short_norm_stock_df['low'].max()
        self.short_norm_stock_df['close'] = self.short_norm_stock_df['close'] / self.short_norm_stock_df['close'].max()

        self.short_norm_stock_df['ema26'] = self.short_norm_stock_df['close'].ewm(span=26, adjust=False).mean()
        self.short_norm_stock_df['ema12'] = self.short_norm_stock_df['close'].ewm(span=12, adjust=False).mean()

        self.short_norm_stock_df['daily_black'] = self.short_norm_stock_df['daily_black'] / self.short_norm_stock_df['daily_black'].abs().max()
        self.short_norm_stock_df['daily_red'] = self.short_norm_stock_df['daily_red'] / self.short_norm_stock_df['daily_red'].abs().max()

        self.short_norm_stock_df['ema60'] = self.short_norm_stock_df['close'].ewm(span=60, adjust=False).mean()
        self.short_norm_stock_df['ema130'] = self.short_norm_stock_df['close'].ewm(span=130, adjust=False).mean()

        self.short_norm_stock_df['weekly_black'] = self.short_norm_stock_df['weekly_black'] / self.short_norm_stock_df['weekly_black'].abs().max()
        self.short_norm_stock_df['weekly_red'] = self.short_norm_stock_df['weekly_red'] / self.short_norm_stock_df['weekly_red'].abs().max()
        env_percent = .04
        self.short_norm_stock_df['fit_env'] = 0
        while self.short_norm_stock_df['fit_env'].mean() <= .9:
            self.short_norm_stock_df['ema26_highenv'] = self.short_norm_stock_df['ema26'] * (1 + env_percent)
            self.short_norm_stock_df['ema26_lowenv'] = self.short_norm_stock_df['ema26'] * (1 - env_percent)
            self.short_norm_stock_df['fit_env'] = np.where((self.short_norm_stock_df['ema26_highenv'] >= self.short_norm_stock_df['high']) &
                                                           (self.short_norm_stock_df['ema26_lowenv'] <= self.short_norm_stock_df['low']),1, 0)
            env_percent += .0005

        pd.options.mode.chained_assignment = None
        self.short_norm_stock_df['obv_volume'] = 0
        self.short_norm_stock_df['obv_volume'][0] = self.short_norm_stock_df['volume'][0]
        for i in range(1, self.row):
            if self.short_norm_stock_df['close'][i] > self.short_norm_stock_df['close'][i - 1]:
                self.short_norm_stock_df['obv_volume'][i] = self.short_norm_stock_df['obv_volume'][i - 1] + self.short_norm_stock_df['volume'][i]
            elif self.short_norm_stock_df['close'][i] < self.short_norm_stock_df['close'][i - 1]:
                self.short_norm_stock_df['obv_volume'][i] = self.short_norm_stock_df['obv_volume'][i - 1] - self.short_norm_stock_df['volume'][i]
            else:
                self.short_norm_stock_df['obv_volume'][i] = self.short_norm_stock_df['obv_volume'][i - 1]
                self.short_norm_stock_df['volume'] = self.short_norm_stock_df['volume'] / self.short_norm_stock_df['volume'].max()
                self.short_norm_stock_df['obv_volume'] = self.short_norm_stock_df['obv_volume'] / self.short_norm_stock_df['obv_volume'].max()
        avg_volume = self.short_norm_stock_df['volume'][self.row - 10:].mean()
        self.short_norm_stock_df['close_from_top_env'] = self.short_norm_stock_df['ema26_highenv'] - self.short_norm_stock_df['close']
        if self.row > 5:
            self.short_norm_stock_df['weekly_black_deriv'] = np.gradient(self.short_norm_stock_df['weekly_black'])
            self.short_norm_stock_df['weekly_black_deriv'] = self.short_norm_stock_df['weekly_black_deriv'].ewm(span=8, adjust=False).mean()
            self.short_norm_stock_df['weekly_black_deriv'] = self.short_norm_stock_df['weekly_black_deriv'] / self.short_norm_stock_df['weekly_black_deriv'].abs().max()
            self.short_norm_stock_df['weekly_red_deriv'] = np.gradient(self.short_norm_stock_df['weekly_red'])
            self.short_norm_stock_df['weekly_red_deriv'] = self.short_norm_stock_df['weekly_red_deriv'].ewm(span=8, adjust=False).mean()
            self.short_norm_stock_df['weekly_red_deriv'] = self.short_norm_stock_df['weekly_red_deriv'] / self.short_norm_stock_df['weekly_red_deriv'].abs().max()

        return False

    def plot_chart(self):

        f1 = plt.figure(figsize=(10, 6))

        ax = plt.axes([0.05, 0.52, 0.9, 0.4])
        ax.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['ema26'], color='purple', label='ema26', linewidth=1.0)
        ax.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['ema26_highenv'], color='purple', label='ema26_highenv', linewidth=1.0)
        ax.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['ema26_lowenv'], color='purple', label='ema26_lowenv', linewidth=1.0)
        ax.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['close'])

        ax.grid(True, which='both')
        ax.set_title(self.stock)
        ax.minorticks_on()
        ax.tick_params(labelright=True)
        ax.get_xaxis().set_visible(False)

        ax2 = plt.axes([0.05, 0.40, 0.9, 0.1])
        ax5 = ax2.twinx()
        ax5 = plt.axes([0.05, 0.40, 0.9, 0.1])

        for j in range(1, self.row):
            if self.short_norm_stock_df['close'][j] >= self.short_norm_stock_df['close'][j - 1]:
                ax2.bar(self.short_norm_stock_df.index[j], self.short_norm_stock_df['volume'][j].astype('float'), color='black', width=1)
            elif self.short_norm_stock_df['close'][j] < self.short_norm_stock_df['close'][j - 1]:
                ax2.bar(self.short_norm_stock_df.index[j], self.short_norm_stock_df['volume'][j].astype('float'), color='red', width=1)

        ax5.plot(self.stock_df.index[self.data_years:], self.stock_df['obv_volume'][self.data_years:])
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax2.grid(True, which='both')
        ax2.tick_params(labelright=True)
        ax2.get_xaxis().set_visible(False)
        ax5.get_xaxis().set_visible(False)
        ax5.get_yaxis().set_visible(False)

        ax3 = plt.axes([0.05, 0.28, 0.9, 0.1])
        ax3.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['daily_black'], color='black', label='black')
        ax3.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['daily_red'], color='red', label='red')
        ax3.plot(self.short_norm_stock_df.index, np.zeros(self.row), color='black', label='black')
        ax3.grid(True, which='both')
        ymin, ymax = ax3.get_ylim()
        if abs(ymax) >= abs(ymin):
            ax3lim = abs(ymax)
        else:
            ax3lim = abs(ymin)
        ax3.set_ylim([-ax3lim, ax3lim])
        ax3.get_xaxis().set_visible(False)
        ax3.grid(True, which='both')
        ax3.tick_params(labelright=True)

        ax4 = plt.axes([0.05, 0.16, 0.9, 0.1])
        ax4.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['weekly_black'], color='black', label='black')
        ax4.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['weekly_red'], color='red', label='red')
        ax4.plot(self.short_norm_stock_df.index, np.zeros(self.row), color='black', label='black')
        ax4.grid(True, which='both')
        ymin, ymax = ax4.get_ylim()
        if abs(ymax) >= abs(ymin):
            ax4lim = abs(ymax)
        else:
            ax4lim = abs(ymin)
        ax4.set_ylim([-ax4lim, ax4lim])
        ax4.get_xaxis().set_visible(False)
        ax4.grid(True, which='both')
        ax4.tick_params(labelright=True)

        mondays = WeekdayLocator(MONDAY)
        months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
        monthsFmt = DateFormatter("%b")
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)

        ax6 = plt.axes([0.05, 0.04, 0.9, 0.1])
        ax6.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['weekly_black_deriv'])
        ax6.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['weekly_red_deriv'])
        ax6.plot(self.short_norm_stock_df.index, np.zeros(self.row), color='black', label='black')
        ax6.grid(True, which='both')
        ymin, ymax = ax6.get_ylim()
        if abs(ymax) >= abs(ymin):
            ax6lim = abs(ymax)
        else:
            ax6lim = abs(ymin)
        ax6.set_ylim([-ax6lim, ax6lim])
        ax6.grid(True, which='both')
        ax6.tick_params(labelright=True)

        stock_gain = self.weekly_slope_crossover_history()
        if len(stock_gain) == 0:
            print("length zero")
            stock_gain = pd.DataFrame(data=[0])
        print(round(stock_gain, 1))
        print("Mean gain: ", round(stock_gain.mean(), 1), "%")

        criteria = "weekly slope crossover up"
        i = 0
        for i in range(1, self.row):
            if criteria == "weekly slope crossover up":
                result = self.weekly_slope_crossover_zero(i)
            else:
                result = False
            if result == True:
                ax.axvline(x=str(self.short_norm_stock_df.index[i]), color='green', linewidth=1)
                ax3.axvline(x=str(self.short_norm_stock_df.index[i]), color='green', linewidth=1)
                ax4.axvline(x=str(self.short_norm_stock_df.index[i]), color='green', linewidth=1)
                #ax5.axvline(x=str(self.short_norm_stock_df.index[i]), color='green', linewidth=1)
                ax6.axvline(x=str(self.short_norm_stock_df.index[i]), color='green', linewidth=1)

        criteria = "weekly slope crossover down"
        i = 0
        for i in range(1, self.row):
            if criteria == "weekly slope crossover down":
                result = self.weekly_slope_crossover_down(i)
            else:
                result = False
            if result == True:
                ax.axvline(x=str(self.short_norm_stock_df.index[i]), color='red', linewidth=1)
                ax3.axvline(x=str(self.short_norm_stock_df.index[i]), color='red', linewidth=1)
                ax4.axvline(x=str(self.short_norm_stock_df.index[i]), color='red', linewidth=1)
                #ax5.axvline(x=str(self.short_norm_stock_df.index[i]), color='red', linewidth=1)
                ax6.axvline(x=str(self.short_norm_stock_df.index[i]), color='red', linewidth=1)

        return f1, stock_gain

    def weekly_slope_crossover_up(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] > self.short_norm_stock_df['weekly_red_deriv'][effective_row] and \
                self.short_norm_stock_df['weekly_black_deriv'][effective_row-1] < self.short_norm_stock_df['weekly_red_deriv'][effective_row-1]:
            passCriteria = True
        return passCriteria

    def weekly_slope_crossover_zero(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] > 0 and self.short_norm_stock_df['weekly_black_deriv'][effective_row-1] < 0:
            passCriteria = True
        return passCriteria

    def weekly_slope_crossover_history(self):
        #global stock_gain
        # i = 0
        crossed_up = False
        stock_gain = pd.DataFrame()
        for i in range(1, self.row):
            # criteria = "weekly slope crossover up"
            if self.weekly_slope_crossover_zero(i) == True and crossed_up == False:
                price_up = self.short_norm_stock_df['close'][i - 1]
                crossed_up = True

            # criteria = "weekly slope crossover down"
            if self.weekly_slope_crossover_down(i) == True and crossed_up == True:
                price_down = self.short_norm_stock_df['close'][i]
                stock_gain = stock_gain.append(pd.Series((price_down - price_up) / price_up), ignore_index=True)
                crossed_up = False
                price_up = 0
                # price_down = 0

        stock_gain = stock_gain * 100

        return stock_gain

    def weekly_slope_crossover_down(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] < self.short_norm_stock_df['weekly_red_deriv'][effective_row] and \
                self.short_norm_stock_df['weekly_black_deriv'][effective_row-1] > self.short_norm_stock_df['weekly_red_deriv'][effective_row-1]:
            passCriteria = True
        return passCriteria