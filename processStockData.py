import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from pandas import *
from mpl_finance import *
import matplotlib.dates as mdates
from matplotlib.dates import *
import datetime as dt
import iexfinance as iex
from dateutil import parser
from matplotlib.ticker import Formatter
from matplotlib.dates import bytespdate2num, num2date
import os
from textwrap import wrap

class stock_data:

    def __init__(self):
        self.now = dt.datetime.now()
        self.start = datetime(self.now.year - 4, 1, 1)
        self.pull_years = 1.5
        self.row = 0
        self.stock = ""
        self.data_years = 0
        self.stock_df = pd.DataFrame()
        self.short_stock_df = pd.DataFrame()
        self.short_norm_stock_df = pd.DataFrame()
        self.pics_filepath = ""
        self.env_percent = 0.04
        self.last_gain = ""
        self.stock_name = ""
        self.avg_volume = 0

        self.trend_status = [None] * 6

    def hist_prices(self, stock, end):

        self.stock = stock.upper()
        # print("here1")
        try:
            self.stock_df = data.DataReader(self.stock, 'iex', self.start, end)
        except:
            print("ignore this stock 1---------------" + str(self.stock))
            return True
        # print("here2")
        self.row, col = self.stock_df.shape
        if self.row < 100:
            print("not enough data" + str(self.stock))
            return True
        # print(self.stock_df.index[self.row - 1], str(end)[0:10])
        if self.stock_df.index[self.row - 1] == str(end)[0:10]:
            # print("drop")
            self.stock_df = self.stock_df.drop(self.stock_df.index[self.row - 1])
            self.row, col = self.stock_df.shape
        # print("here3")
        if self.stock_df.index[self.row - 1] != str(end)[0:10]:
            stock_quote = iex.Stock(self.stock)
            try:
                stock_pd = stock_quote.get_quote()
            except:
                print("ignore todays prices 2----------" + str(self.stock))
                return True
            stock_date = stock_pd['latestTime']
            stock_date = str(parser.parse(stock_date))[0:10]
            if stock_date == end:
                stock_open = stock_pd['open']
                stock_high = stock_pd['high']
                stock_low = stock_pd['low']
                stock_close = stock_pd['close']
                stock_volume = stock_pd['latestVolume']
                self.stock_name = stock_pd['companyName']
                if stock_volume == None:
                    stock_volume = self.stock_df['volume'].mean()
                f_day = {'open': [stock_open], 'high': [stock_high], 'low': [stock_low], 'close': [stock_close],
                         'volume': [stock_volume]}
                date_index = pd.date_range(stock_date, periods=1, freq='D')
                f_day = pd.DataFrame(data=f_day, index=date_index)
                f_day.index.name = 'date'
                self.stock_df = self.stock_df.append(f_day.ix[0])

        # print("here4")
        self.row, col = self.stock_df.shape
        # print(self.stock_df)
        if self.stock_df.isnull().values.any() == True:
            print("nans in stockdf", stock)
            return True
        # print("here41")
        self.stock_df = self.stock_df.set_index(pd.to_datetime(self.stock_df.index))
        self.data_years = self.row - int(round(253 * self.pull_years, 0))
        self.stock_df['date'] = self.stock_df.index.map(mdates.date2num)
        self.stock_df['averaged_close'] = self.stock_df['close'].ewm(span=4, adjust=False).mean()
        # print("here42")
        # calculate MACD parameters
        self.stock_df['ema12'] = self.stock_df['close'].ewm(span=30, adjust=False).mean()
        self.stock_df['ema26'] = self.stock_df['close'].ewm(span=65, adjust=False).mean()
        self.stock_df['daily_black'] = self.stock_df['ema12'] - self.stock_df['ema26']
        self.stock_df['daily_red'] = self.stock_df['daily_black'].ewm(span=35, adjust=False).mean()
        self.stock_df['ema60'] = self.stock_df['close'].ewm(span=60, adjust=False).mean()
        self.stock_df['ema130'] = self.stock_df['close'].ewm(span=130, adjust=False).mean()
        self.stock_df['weekly_black'] = self.stock_df['ema60'] - self.stock_df['ema130']
        self.stock_df['weekly_red'] = self.stock_df['weekly_black'].ewm(span=45, adjust=False).mean()
        self.stock_df['sma_26'] = self.stock_df['close'].rolling(window=26).mean()
        self.stock_df['ema_200'] = self.stock_df['close'].ewm(span=200, adjust=False).mean()
        # print("here43")
        env_percent = .04
        self.stock_df['fit_env'] = 0
        # pd.set_option('display.max_rows', 5000)


        # print(self.stock_df['high'].isnull().values.any())
        while self.stock_df['fit_env'].mean() <= .9:
            self.stock_df['ema26_highenv'] = self.stock_df['ema26'] * (1 + env_percent)
            self.stock_df['ema26_lowenv'] = self.stock_df['ema26'] * (1 - env_percent)
            self.stock_df['fit_env'] = np.where((self.stock_df['ema26_highenv'] >= self.stock_df['high']) &
                                                (self.stock_df['ema26_lowenv'] <= self.stock_df['low']), 1, 0)
            # print(self.stock_df['fit_env'].mean())
            env_percent += .005

        # print("here5")
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

        self.avg_volume = self.stock_df['volume'][self.row - 3:self.row - 1].mean()

        self.stock_df['volume'] = self.stock_df['volume'] / self.stock_df['volume'].max()
        self.stock_df['obv_volume'] = self.stock_df['obv_volume'] / self.stock_df['obv_volume'].max()
        self.stock_df['obv_volume'] = self.stock_df['obv_volume'].ewm(span=50, adjust=False).mean()


        self.stock_df['close_from_top_env'] = self.stock_df['ema26_highenv'] - self.stock_df['close']
        print(self.row)
        if self.row > 5:
            self.stock_df['daily_black_deriv'] = np.gradient(self.stock_df['daily_black'])
            self.stock_df['daily_red_deriv'] = np.gradient(self.stock_df['daily_red'])
            self.stock_df['weekly_black_deriv'] = np.gradient(self.stock_df['weekly_black'])
            self.stock_df['weekly_red_deriv'] = np.gradient(self.stock_df['weekly_red'])
            self.stock_df['ema_200_slope'] = np.gradient(self.stock_df['ema_200'])
        elif self.row < 5:
            print("gradient row error", str(self.stock))
            return True
        # print("here6")
        self.stock_df['decision'] = 0
        self.stock_df['stock-name'] = stock
        self.stock_df['performance'] = 0

        self.short_stock_df = self.stock_df[self.data_years:]
        self.short_norm_stock_df = self.stock_df[self.data_years:]
        self.row, col = self.short_norm_stock_df.shape

        if self.row < 5:
            print("gradient row error", str(self.stock))
            return True

        self.short_norm_stock_df['open'] = self.short_norm_stock_df['open'] / self.short_norm_stock_df['open'].max()
        self.short_norm_stock_df['high'] = self.short_norm_stock_df['high'] / self.short_norm_stock_df['high'].max()
        self.short_norm_stock_df['low'] = self.short_norm_stock_df['low'] / self.short_norm_stock_df['low'].max()
        self.short_norm_stock_df['close'] = self.short_norm_stock_df['close'] / self.short_norm_stock_df['close'].max()
        self.short_norm_stock_df['volume'] = self.short_norm_stock_df['volume'] / self.short_norm_stock_df['volume'].max()
        self.short_norm_stock_df['obv_volume'] = self.short_norm_stock_df['obv_volume'] / self.short_norm_stock_df['obv_volume'].abs().max()
        self.short_norm_stock_df['obv_volume_slope'] = np.gradient(self.short_norm_stock_df['obv_volume'])
        self.short_norm_stock_df['averaged_close'] = self.short_norm_stock_df['averaged_close'] / self.short_norm_stock_df['averaged_close'].max()

        self.short_norm_stock_df['sma_26'] = self.short_norm_stock_df['close'].rolling(window=26).mean()
        self.short_norm_stock_df['ema_200'] = self.short_norm_stock_df['close'].ewm(span=200, adjust=False).mean()
        self.short_norm_stock_df['ema26'] = self.short_norm_stock_df['close'].ewm(span=26, adjust=False).mean()

        daily_max = self.short_norm_stock_df['daily_black'].abs().max()
        if daily_max < self.short_norm_stock_df['daily_red'].abs().max():
            daily_max = self.short_norm_stock_df['daily_red'].abs().max()

        self.short_norm_stock_df['daily_black'] = self.short_norm_stock_df['daily_black'] / daily_max
        self.short_norm_stock_df['daily_red'] = self.short_norm_stock_df['daily_red'] / daily_max
        # print("here7")
        weekly_max = self.short_norm_stock_df['weekly_black'].abs().max()
        if weekly_max < self.short_norm_stock_df['weekly_red'].abs().max():
            weekly_max = self.short_norm_stock_df['weekly_red'].abs().max()

        self.short_norm_stock_df['weekly_black'] = self.short_norm_stock_df['weekly_black'] / weekly_max
        self.short_norm_stock_df['weekly_red'] = self.short_norm_stock_df['weekly_red'] / weekly_max
        self.short_norm_stock_df['fit_env'] = 0
        self.env_percent = 0.04
        while self.short_norm_stock_df['fit_env'].mean() <= .95:
            self.short_norm_stock_df['ema26_highenv'] = self.short_norm_stock_df['ema26'] * (1 + self.env_percent)
            self.short_norm_stock_df['ema26_lowenv'] = self.short_norm_stock_df['ema26'] * (1 - self.env_percent)
            self.short_norm_stock_df['fit_env'] = np.where((self.short_norm_stock_df['ema26_highenv'] >= self.short_norm_stock_df['high']) &
                                                           (self.short_norm_stock_df['ema26_lowenv'] <= self.short_norm_stock_df['low']),1, 0)
            self.env_percent += .0005

        # print(self.short_norm_stock_df['close'][self.row - 1], self.short_norm_stock_df['close'][self.row - 2])
        self.last_gain = "Last Gain: " + str(round((((self.short_norm_stock_df['close'][self.row - 1] - self.short_norm_stock_df['close'][self.row - 2])/self.short_norm_stock_df['close'][self.row - 2]) * 100), 2)) + " %"
        # print(self.last_gain)

        pd.options.mode.chained_assignment = None

        avg_volume = self.short_norm_stock_df['volume'][self.row - 10:].mean()
        self.short_norm_stock_df['close_from_top_env'] = self.short_norm_stock_df['ema26_highenv'] - self.short_norm_stock_df['close']
        if self.row > 5:
            self.short_norm_stock_df['weekly_black_deriv'] = np.gradient(self.short_norm_stock_df['weekly_black'])
            self.short_norm_stock_df['weekly_red_deriv'] = np.gradient(self.short_norm_stock_df['weekly_red'])

            self.short_norm_stock_df['weekly_black_deriv'] = self.short_norm_stock_df['weekly_black_deriv'].ewm(span=6, adjust=False).mean()
            self.short_norm_stock_df['daily_black_deriv'] = self.short_norm_stock_df['daily_black_deriv'].ewm(span=6,adjust=False).mean()

            self.short_norm_stock_df['daily_black_deriv'] = np.gradient(self.short_norm_stock_df['daily_black'])
            self.short_norm_stock_df['daily_red_deriv'] = np.gradient(self.short_norm_stock_df['daily_red'])
            self.short_norm_stock_df['ema_200_slope'] = np.gradient(self.short_norm_stock_df['ema_200'])

            weekly_slope_max = self.short_norm_stock_df['weekly_black_deriv'].abs().max()
            if weekly_slope_max < self.short_norm_stock_df['weekly_red_deriv'].abs().max():
                weekly_slope_max = self.short_norm_stock_df['weekly_red_deriv'].abs().max()

            daily_slope_max = self.short_norm_stock_df['daily_black_deriv'].abs().max()
            if daily_slope_max < self.short_norm_stock_df['daily_red_deriv'].abs().max():
                daily_slope_max = self.short_norm_stock_df['daily_red_deriv'].abs().max()

            self.short_norm_stock_df['daily_black_deriv'] = self.short_norm_stock_df['daily_black_deriv'] / daily_slope_max
            self.short_norm_stock_df['daily_red_deriv'] = self.short_norm_stock_df['daily_red_deriv'] / daily_slope_max
            self.short_norm_stock_df['daily_slope_histogram'] = self.short_norm_stock_df['daily_black_deriv'] - self.short_norm_stock_df['daily_red_deriv']

            self.short_norm_stock_df['weekly_black_deriv'] = self.short_norm_stock_df['weekly_black_deriv'] / weekly_slope_max
            self.short_norm_stock_df['weekly_red_deriv'] = self.short_norm_stock_df['weekly_red_deriv'] / weekly_slope_max
            self.short_norm_stock_df['weekly_slope_histogram'] = self.short_norm_stock_df['weekly_black_deriv'] - self.short_norm_stock_df['weekly_red_deriv']
        # print("here8")
        return False

    def plot_chart(self):

        class MyFormatter(Formatter):
            def __init__(self, dates, fmt='%Y-%m-%d'):
                self.dates = dates
                self.fmt = fmt

            def __call__(self, x, pos=0):
                'Return the label for time x at position pos'
                ind = int(np.round(x))
                if ind >= len(self.dates) or ind < 0:
                    return ''

                return num2date(self.dates[ind]).strftime(self.fmt)

        formatter = MyFormatter(self.short_norm_stock_df.values[:, 5])

        f1 = plt.figure(figsize=(12, 9))

        ax = plt.axes([0.05, 0.57, 0.9, 0.38])
        ax.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['ema26'], color='purple', label='ema26', linewidth=1.0)
        ax.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['ema26_highenv'], color='purple', label='ema26_highenv', linewidth=1.0)
        ax.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['ema26_lowenv'], color='purple', label='ema26_lowenv', linewidth=1.0)
        ax.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['close'])
        ax.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['sma_26'])
        ax.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['ema_200'])
        ax.fill_between(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['ema26']*(1 + self.env_percent/2), self.short_norm_stock_df['ema26']*(1 - self.env_percent/2), facecolor='purple', alpha=0.25)

        ax.set_title(self.stock + " - " + self.stock_name + "\nEnvelope: " + str(round(self.env_percent*100,1)) + " %" + "           " + self.last_gain + "           Avg. Volume: " + str("{:,}".format(int(self.avg_volume))))
        ax.grid(True, which='both')
        ax.tick_params(labelright=True)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_ticklabels([])

        ax2 = plt.axes([0.05, 0.46, 0.9, 0.1])
        ax5 = ax2.twinx()
        ax5 = plt.axes([0.05, 0.46, 0.9, 0.1])

        for j in range(1, self.row):
            if self.short_norm_stock_df['close'][j] >= self.short_norm_stock_df['close'][j - 1]:
                ax2.bar(j, self.short_norm_stock_df['volume'][j].astype('float'), color='black', width=1)
            elif self.short_norm_stock_df['close'][j] < self.short_norm_stock_df['close'][j - 1]:
                ax2.bar(j, self.short_norm_stock_df['volume'][j].astype('float'), color='red', width=1)

        ax5.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['obv_volume'])
        ax5.get_yaxis().set_visible(False)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax2.grid(True, which='both')
        ax2.tick_params(labelright=True)
        ax2.xaxis.set_major_formatter(formatter)
        ax2.xaxis.set_ticklabels([])

        ax3 = plt.axes([0.05, 0.35, 0.9, 0.1])
        ax3.plot(np.arange(len(self.short_norm_stock_df.index)), np.zeros(self.row), color='gray', label='gray')
        ax3.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['daily_black'], color='black', label='black')
        ax3.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['daily_red'], color='red', label='red')
        ymin, ymax = ax3.get_ylim()
        if abs(ymax) >= abs(ymin):
            ax3lim = abs(ymax)
        else:
            ax3lim = abs(ymin)
        ax3.set_ylim([-ax3lim, ax3lim])
        ax3.grid(True, which='both')
        ax3.tick_params(labelright=True)
        ax3.xaxis.set_major_formatter(formatter)
        ax3.xaxis.set_ticklabels([])

        ax4 = plt.axes([0.05, 0.19, 0.9, 0.15])
        ax4.plot(np.arange(len(self.short_norm_stock_df.index)), np.zeros(self.row), color='gray', label='gray')
        ax4.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['weekly_black'], color='black', label='black')
        ax4.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['weekly_red'], color='red', label='red')
        ymin, ymax = ax4.get_ylim()
        if abs(ymax) >= abs(ymin):
            ax4lim = abs(ymax)
        else:
            ax4lim = abs(ymin)
        ax4.set_ylim([-ax4lim, ax4lim])
        ax4.grid(True, which='both')
        ax4.tick_params(labelright=True)
        ax4.xaxis.set_major_formatter(formatter)
        ax4.xaxis.set_ticklabels([])

        ax6 = plt.axes([0.05, 0.03, 0.9, 0.15])
        ax6.xaxis.set_major_formatter(formatter)

        ax6.plot(np.arange(len(self.short_norm_stock_df.index)), np.zeros(self.row), color='gray', label='gray')
        ax6.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['weekly_black_deriv'])
        ax6.plot(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['weekly_red_deriv'])
        ax6.bar(np.arange(len(self.short_norm_stock_df.index)), self.short_norm_stock_df['weekly_slope_histogram'].astype('float'), edgecolor='gray', color='gray', width=1)
        ymin, ymax = ax6.get_ylim()
        if abs(ymax) >= abs(ymin):
            ax6lim = abs(ymax)
        else:
            ax6lim = abs(ymin)
        ax6.set_ylim([-ax6lim, ax6lim])
        ax6.grid(True, which='both')
        ax6.tick_params(labelright=True)

        stock_gain = self.slope_crossover_history()
        if len(stock_gain) == 0:
            print("length zero")
            stock_gain = pd.DataFrame(data=[0])
        # print(round(stock_gain, 1))
        # print("Mean gain: ", round(stock_gain.mean(), 1), "%")

        i = 0
        price_up = 0
        bought = False
        crossed_up = False
        ax.fill
        for i in range(1, self.row):
            if self.weekly_red_above_zero(i) == True and self.daily_red_above_zero(i) == True and self.price_above_EMA200(i) == True and self.obv_volume_slope_up(i) == True and self.EMA200_slope_up(i) == True:# and crossed_up == False:
                # if i + 1 != self.row:
                    # price_up = self.short_norm_stock_df['open'][i + 1]
                # print(i, self.short_norm_stock_df.index[i])
                ax.axvline(x=i, color='green', linewidth=2, alpha=0.5)
                ax3.axvline(x=i, color='green', linewidth=2, alpha=0.5)
                ax4.axvline(x=i, color='green', linewidth=2, alpha=0.5)
                ax6.axvline(x=i, color='green', linewidth=2, alpha=0.5)
                    # bought = True

            # elif self.weekly_red_above_zero(i) == False and bought == True and crossed_up == True:
            #     ax.axvline(x=i, color='red', linewidth=2)
            #     ax3.axvline(x=i, color='red', linewidth=2)
            #     ax4.axvline(x=i, color='red', linewidth=2)
            #     ax6.axvline(x=i, color='red', linewidth=2)
            #     crossed_up = False
            #     bought = False

            elif self.weekly_red_above_zero(i) == True and self.daily_red_above_zero(i) == True and self.price_above_EMA200(i) == True and self.obv_volume_slope_up(i) == False and self.EMA200_slope_up(i) == True:
                ax.axvline(x=i, color='lightgreen', linewidth=2, alpha=0.5)
                ax3.axvline(x=i, color='lightgreen', linewidth=2, alpha=0.5)
                ax4.axvline(x=i, color='lightgreen', linewidth=2, alpha=0.5)
                ax6.axvline(x=i, color='lightgreen', linewidth=2, alpha=0.5)

            elif self.weekly_red_above_zero(i) == True and self.daily_red_above_zero(i) == False and self.price_above_EMA200(i) == True and self.EMA200_slope_up(i) == True:
                ax.axvline(x=i, color='yellow', linewidth=2, alpha=0.5)
                ax3.axvline(x=i, color='yellow', linewidth=2, alpha=0.5)
                ax4.axvline(x=i, color='yellow', linewidth=2, alpha=0.5)
                ax6.axvline(x=i, color='yellow', linewidth=2, alpha=0.5)

            elif self.weekly_red_above_zero(i) == False and self.daily_red_above_zero(i) == True and self.price_above_EMA200(i) == True  and self.obv_volume_slope_up(i) == True and self.EMA200_slope_up(i) == True:
                ax.axvline(x=i, color='lightgreen', linewidth=2, alpha=0.5)
                ax3.axvline(x=i, color='lightgreen', linewidth=2, alpha=0.5)
                ax4.axvline(x=i, color='lightgreen', linewidth=2, alpha=0.5)
                ax6.axvline(x=i, color='lightgreen', linewidth=2, alpha=0.5)

            # elif self.weekly_red_above_zero(i) == False and self.daily_red_above_zero(i) == False and self.price_above_EMA200(i) == True  and self.EMA200_slope_up(i) == True:
            #     ax.axvline(x=i, color='red', linewidth=2, alpha=0.2)
            #     ax3.axvline(x=i, color='red', linewidth=2, alpha=0.2)
            #     ax4.axvline(x=i, color='red', linewidth=2, alpha=0.2)
            #     ax6.axvline(x=i, color='red', linewidth=2, alpha=0.2)
            #
            # elif self.weekly_red_above_zero(i) == False and self.daily_red_above_zero(i) == True and self.price_above_EMA200(i) == True  and self.EMA200_slope_up(i) == True:
            #     ax.axvline(x=i, color='red', linewidth=2, alpha=0.2)
            #     ax3.axvline(x=i, color='red', linewidth=2, alpha=0.2)
            #     ax4.axvline(x=i, color='red', linewidth=2, alpha=0.2)
            #     ax6.axvline(x=i, color='red', linewidth=2, alpha=0.2)
            #
            # # elif self.weekly_red_above_zero(i) == True and self.daily_red_above_zero(i) == False and self.price_above_EMA200(i) == True and self.EMA200_slope_up(i) == True:
            # #     ax.axvline(x=i, color='lightgreen', linewidth=2, alpha=0.5)
            # #     ax3.axvline(x=i, color='lightgreen', linewidth=2, alpha=0.5)
            # #     ax4.axvline(x=i, color='lightgreen', linewidth=2, alpha=0.5)
            # #     ax6.axvline(x=i, color='lightgreen', linewidth=2, alpha=0.5)
            # #
            # # elif self.weekly_red_above_zero(i) == True and self.daily_red_above_zero(i) == True and self.price_above_EMA200(i) == False and self.EMA200_slope_up(i) == True:
            # #     ax.axvline(x=i, color='red', linewidth=2, alpha=0.1)
            # #     ax3.axvline(x=i, color='red', linewidth=2, alpha=0.1)
            # #     ax4.axvline(x=i, color='red', linewidth=2, alpha=0.1)
            # #     ax6.axvline(x=i, color='red', linewidth=2, alpha=0.1)
            # #
            # # elif self.weekly_red_above_zero(i) == True and self.daily_red_above_zero(i) == False and (self.price_above_EMA200(i) == False or self.EMA200_slope_up(i) == False):
            # #     ax.axvline(x=i, color='red', linewidth=2, alpha=0.1)
            # #     ax3.axvline(x=i, color='red', linewidth=2, alpha=0.1)
            # #     ax4.axvline(x=i, color='red', linewidth=2, alpha=0.1)
            # #     ax6.axvline(x=i, color='red', linewidth=2, alpha=0.1)
            #
            # elif self.price_above_EMA200(i) == False or self.EMA200_slope_up(i) == False:
            #     ax.axvline(x=i, color='red', linewidth=2, alpha=0.4)
            #     ax3.axvline(x=i, color='red', linewidth=2, alpha=0.4)
            #     ax4.axvline(x=i, color='red', linewidth=2, alpha=0.4)
            #     ax6.axvline(x=i, color='red', linewidth=2, alpha=0.4)

            # else:
            #     print(i, self.weekly_red_above_zero(i), self.daily_red_above_zero(i), self.price_above_EMA200(i), self.obv_volume_slope_up(i), self.EMA200_slope_up(i))

        plt.close(f1)
        f1.savefig(os.path.join(self.pics_filepath, str(self.stock) + '.png'))

        return stock_gain

    def slope_crossover_history(self):
        crossed_up = False
        bought = False
        stock_gain = pd.DataFrame()
        price_up = 0
        percent = 3
        for i in range(1, self.row):
            if self.weekly_red_above_zero(i) == True and self.obv_volume_slope_up(i) == True and self.EMA200_slope_up(i) == True and crossed_up == False:
                if i + 1 != self.row:
                    price_up = self.short_norm_stock_df['open'][i+1]
                    bought = True
                crossed_up = True
            elif self.weekly_red_above_zero(i) == False and bought == True  and crossed_up == True: # (self.weekly_black_slope_cross_below_zero(i) == True or self.weekly_black_slope_cross_below_red(i)) and crossed_up == True:
                # print(self.daily_red_above_zero(i), self.EMA200_slope_up(i))
                price_down = self.short_norm_stock_df['close'][i]
                if bought == True:
                    stock_gain = stock_gain.append(pd.Series((price_down - price_up) / price_up), ignore_index=True)
                crossed_up = False
                bought = False
                price_up = 0

        stock_gain = stock_gain * 100

        return stock_gain

    def weekly_black_slope_cross_below_red(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] < self.short_norm_stock_df['weekly_red_deriv'][effective_row] - 0.05 and \
                self.short_norm_stock_df['weekly_black_deriv'][effective_row-1] > self.short_norm_stock_df['weekly_red_deriv'][effective_row-1] - 0.05:
            passCriteria = True
        return passCriteria

    def weekly_black_slope_crossover_zero(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] > 0 and self.short_norm_stock_df['weekly_black_deriv'][effective_row-1] < 0:
            passCriteria = True
        return passCriteria

    def weekly_red_slope_crossover_zero(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_red_deriv'][effective_row] > 0 and self.short_norm_stock_df['weekly_red_deriv'][effective_row-1] < 0:
            passCriteria = True
        return passCriteria

    def weekly_black_slope_cross_below_zero(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] < 0 and \
                self.short_norm_stock_df['weekly_black_deriv'][effective_row - 1] > 0:
            passCriteria = True
        return passCriteria

    def weekly_red_slope_cross_below_zero(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_red_deriv'][effective_row] < 0 and \
                self.short_norm_stock_df['weekly_red_deriv'][effective_row - 1] > 0:
            passCriteria = True
        return passCriteria

    def weekly_black_slope_crossover_red_slope(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] > self.short_norm_stock_df['weekly_red_deriv'][effective_row] and \
                self.short_norm_stock_df['weekly_black_deriv'][effective_row-1] < self.short_norm_stock_df['weekly_red_deriv'][effective_row-1]:
            passCriteria = True
        return passCriteria

    def weekly_black_slope_cross_below_red_slope(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] < self.short_norm_stock_df['weekly_red_deriv'][effective_row] and \
                self.short_norm_stock_df['weekly_black_deriv'][effective_row-1] > self.short_norm_stock_df['weekly_red_deriv'][effective_row-1]:
            passCriteria = True
        return passCriteria

    def check_trends(self, row1):

        trend_status = [None] * 8

        if self.short_norm_stock_df['close'][row1 - 6:row1 - 1].mean() > self.short_norm_stock_df['close'][row1 - 20 - 6:row1 - 20 - 1].mean():
            trend_status[0] = ' Up '
        else:
            trend_status[0] = 'Down'

        if self.short_norm_stock_df['close'][row1 - 6:row1 - 1].mean() > self.short_norm_stock_df['close'][row1 - 60 - 6:row1 - 60 - 1].mean():
            trend_status[1] = ' Up '
        else:
            trend_status[1] = 'Down'

        if self.short_norm_stock_df['obv_volume'][row1 - 1] > self.short_norm_stock_df['obv_volume'][row1 - 20 - 1]:
            trend_status[2] = ' Up '
        else:
            trend_status[2] = 'Down'

        if self.short_norm_stock_df['obv_volume'][row1 - 1] > self.short_norm_stock_df['obv_volume'][row1 - 60 - 1]:
            trend_status[3] = ' Up '
        else:
            trend_status[3] = 'Down'

        return trend_status

    def weekly_black_and_red_above_zero(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] > 0 and self.short_norm_stock_df['weekly_red_deriv'][effective_row] > 0:
            passCriteria = True
        return passCriteria

    def weekly_black_or_red_below_zero(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] < 0 or self.short_norm_stock_df['weekly_red_deriv'][effective_row] < 0:
            passCriteria = True
        return passCriteria

    def daily_red_cross_below_zero(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['daily_red_deriv'][effective_row] < 0 and self.short_norm_stock_df['daily_red_deriv'][effective_row - 1] > 0:
            passCriteria = True
        return passCriteria

    def weekly_red_above_zero(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_red_deriv'][effective_row] > 0:
            passCriteria = True
        return passCriteria

    def daily_red_above_zero(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['daily_red_deriv'][effective_row] > 0:
            passCriteria = True
        return passCriteria

    def obv_volume_slope_up(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['obv_volume_slope'][effective_row] > 0:
            passCriteria = True
        return passCriteria

    def EMA200_slope_up(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['ema_200_slope'][effective_row] > 0:
            passCriteria = True
        return passCriteria

    def price_above_EMA200(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['close'][effective_row] >= self.short_norm_stock_df['ema_200'][effective_row]:
            passCriteria = True
        return passCriteria