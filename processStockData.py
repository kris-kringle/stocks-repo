import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from pandas import *
from mpl_finance import *
import matplotlib.dates as mdates
from matplotlib.dates import MONDAY
from matplotlib.dates import *
import datetime as dt
import iexfinance as iex
from dateutil import parser

class stock_data:

    def __init__(self):
        self.now = dt.datetime.now()
        self.start = datetime(self.now.year - 4, 1, 1)
        self.pull_years = 2
        self.row = 0
        self.stock = ""
        self.data_years = 0
        self.stock_df = pd.DataFrame()
        self.short_stock_df = pd.DataFrame()
        self.short_norm_stock_df = pd.DataFrame()

        self.trend_status = [None] * 6

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

        # self.stock_df['close'] = self.stock_df['close'].ewm(span=6, adjust=False).mean()

        # calculate MACD parameters
        self.stock_df['ema26'] = self.stock_df['close'].ewm(span=26, adjust=False).mean()
        self.stock_df['ema12'] = self.stock_df['close'].ewm(span=12, adjust=False).mean()
        self.stock_df['daily_black'] = self.stock_df['ema12'] - self.stock_df['ema26']
        self.stock_df['daily_red'] = self.stock_df['daily_black'].ewm(span=18, adjust=False).mean()
        self.stock_df['ema60'] = self.stock_df['close'].ewm(span=60, adjust=False).mean()
        self.stock_df['ema130'] = self.stock_df['close'].ewm(span=130, adjust=False).mean()
        self.stock_df['weekly_black'] = self.stock_df['ema60'] - self.stock_df['ema130']
        self.stock_df['weekly_red'] = self.stock_df['weekly_black'].ewm(span=45, adjust=False).mean()

        #self.stock_df['weekly_black'] = self.stock_df['weekly_black'].ewm(span=5, adjust=False).mean()

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
            self.stock_df['weekly_black_deriv'] = self.stock_df['weekly_black_deriv']#.ewm(span=8, adjust=False).mean()
            self.stock_df['weekly_black_deriv'] = self.stock_df['weekly_black_deriv']
            self.stock_df['weekly_red_deriv'] = np.gradient(self.stock_df['weekly_red'])
            self.stock_df['weekly_red_deriv'] = self.stock_df['weekly_red_deriv']#.ewm(span=8, adjust=False).mean()
            self.stock_df['weekly_red_deriv'] = self.stock_df['weekly_red_deriv']

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

        daily_max = self.short_norm_stock_df['daily_black'].abs().max()
        if daily_max < self.short_norm_stock_df['daily_red'].abs().max():
            daily_max = self.short_norm_stock_df['daily_red'].abs().max()

        self.short_norm_stock_df['daily_black'] = self.short_norm_stock_df['daily_black'] / daily_max
        self.short_norm_stock_df['daily_red'] = self.short_norm_stock_df['daily_red'] / daily_max

        self.short_norm_stock_df['ema60'] = self.short_norm_stock_df['close'].ewm(span=60, adjust=False).mean()
        self.short_norm_stock_df['ema130'] = self.short_norm_stock_df['close'].ewm(span=130, adjust=False).mean()

        weekly_max = self.short_norm_stock_df['weekly_black'].abs().max()
        if weekly_max < self.short_norm_stock_df['weekly_red'].abs().max():
            weekly_max = self.short_norm_stock_df['weekly_red'].abs().max()

        self.short_norm_stock_df['weekly_black'] = self.short_norm_stock_df['weekly_black'] / weekly_max
        self.short_norm_stock_df['weekly_red'] = self.short_norm_stock_df['weekly_red'] / weekly_max
        env_percent = .04
        self.short_norm_stock_df['fit_env'] = 0
        while self.short_norm_stock_df['fit_env'].mean() <= .9:
            self.short_norm_stock_df['ema26_highenv'] = self.short_norm_stock_df['ema26'] * (1 + env_percent)
            self.short_norm_stock_df['ema26_lowenv'] = self.short_norm_stock_df['ema26'] * (1 - env_percent)
            self.short_norm_stock_df['fit_env'] = np.where((self.short_norm_stock_df['ema26_highenv'] >= self.short_norm_stock_df['high']) &
                                                           (self.short_norm_stock_df['ema26_lowenv'] <= self.short_norm_stock_df['low']),1, 0)
            env_percent += .0005

        pd.options.mode.chained_assignment = None

        avg_volume = self.short_norm_stock_df['volume'][self.row - 10:].mean()
        self.short_norm_stock_df['close_from_top_env'] = self.short_norm_stock_df['ema26_highenv'] - self.short_norm_stock_df['close']
        if self.row > 5:
            self.short_norm_stock_df['weekly_black_deriv'] = np.gradient(self.short_norm_stock_df['weekly_black'])
            self.short_norm_stock_df['weekly_red_deriv'] = np.gradient(self.short_norm_stock_df['weekly_red'])
            self.short_norm_stock_df['weekly_black_double_deriv'] = np.gradient(self.short_norm_stock_df['weekly_black_deriv'])
            self.short_norm_stock_df['weekly_black_deriv'] = self.short_norm_stock_df['weekly_black_deriv'].ewm(span=6, adjust=False).mean()
            #self.short_norm_stock_df['weekly_black_deriv'] = self.short_norm_stock_df['weekly_black_deriv'].rolling(window=5).mean()

            #self.short_norm_stock_df['weekly_red_deriv'] = self.short_norm_stock_df['weekly_red_deriv'].ewm(span=2, adjust=False).mean()

            weekly_slope_max = self.short_norm_stock_df['weekly_black_deriv'].abs().max()
            if weekly_slope_max < self.short_norm_stock_df['weekly_red_deriv'].abs().max():
                weekly_slope_max = self.short_norm_stock_df['weekly_red_deriv'].abs().max()

            self.short_norm_stock_df['weekly_black_deriv'] = self.short_norm_stock_df['weekly_black_deriv'] / weekly_slope_max
            self.short_norm_stock_df['weekly_red_deriv'] = self.short_norm_stock_df['weekly_red_deriv'] / weekly_slope_max

            self.short_norm_stock_df['weekly_black_double_deriv'] = self.short_norm_stock_df['weekly_black_double_deriv'] / self.short_norm_stock_df['weekly_black_double_deriv'].abs().max()
            self.short_norm_stock_df['weekly_black_double_deriv'] = self.short_norm_stock_df['weekly_black_double_deriv'].ewm(span=6, adjust=False).mean()
            #self.short_norm_stock_df['weekly_red_double_deriv'] = np.gradient(self.short_norm_stock_df['weekly_red_deriv'])

        return False

    def plot_chart(self):

        months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
        monthsFmt = DateFormatter("%b")

        f1 = plt.figure(figsize=(12, 9))

        ax = plt.axes([0.05, 0.57, 0.9, 0.4])
        ax.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['ema26'], color='purple', label='ema26', linewidth=1.0)
        ax.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['ema26_highenv'], color='purple', label='ema26_highenv', linewidth=1.0)
        ax.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['ema26_lowenv'], color='purple', label='ema26_lowenv', linewidth=1.0)
        ax.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['close'])

        ax.set_title(self.stock)
        ax.grid(True, which='both')
        ax.tick_params(labelright=True)
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        ax.xaxis.set_ticklabels([])
        #ax.minorticks_on()
        #ax.get_xaxis().set_visible(False)



        ax2 = plt.axes([0.05, 0.46, 0.9, 0.1])
        ax5 = ax2.twinx()
        ax5 = plt.axes([0.05, 0.46, 0.9, 0.1])

        #ax2.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['weekly_black_double_deriv'], color='blue', label='blue')
        #ax2.plot(self.short_norm_stock_df.index, np.zeros(self.row), color='black', label='black')
        #ax2.plot(self.short_norm_stock_df.index, np.full((self.row,1), -.50), color='red', label='red')
        #ymin, ymax = ax2.get_ylim()
        #if abs(ymax) >= abs(ymin):
        #    ax2lim = abs(ymax)
        #else:
        #    ax2lim = abs(ymin)
        #ax2.set_ylim([-ax2lim, ax2lim])
        #ax2.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['weekly_red_double_deriv'], color='red', label='red')

        for j in range(1, self.row):
            if self.short_norm_stock_df['close'][j] >= self.short_norm_stock_df['close'][j - 1]:
                ax2.bar(self.short_norm_stock_df.index[j], self.short_norm_stock_df['volume'][j].astype('float'), color='black', width=1)
            elif self.short_norm_stock_df['close'][j] < self.short_norm_stock_df['close'][j - 1]:
                ax2.bar(self.short_norm_stock_df.index[j], self.short_norm_stock_df['volume'][j].astype('float'), color='red', width=1)

        ax5.plot(self.stock_df.index[self.data_years:], self.stock_df['obv_volume'][self.data_years:])
        ax5.get_yaxis().set_visible(False)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax2.grid(True, which='both')
        ax2.tick_params(labelright=True)
        ax2.xaxis.set_major_locator(months)
        ax2.xaxis.set_major_formatter(monthsFmt)
        ax2.xaxis.set_ticklabels([])

        ax3 = plt.axes([0.05, 0.35, 0.9, 0.1])
        ax3.plot(self.short_norm_stock_df.index, np.zeros(self.row), color='gray', label='gray')
        ax3.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['daily_black'], color='black', label='black')
        ax3.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['daily_red'], color='red', label='red')
        ymin, ymax = ax3.get_ylim()
        if abs(ymax) >= abs(ymin):
            ax3lim = abs(ymax)
        else:
            ax3lim = abs(ymin)
        ax3.set_ylim([-ax3lim, ax3lim])
        ax3.grid(True, which='both')
        ax3.tick_params(labelright=True)
        ax3.xaxis.set_major_locator(months)
        ax3.xaxis.set_major_formatter(monthsFmt)
        ax3.xaxis.set_ticklabels([])

        ax4 = plt.axes([0.05, 0.19, 0.9, 0.15])
        ax4.plot(self.short_norm_stock_df.index, np.zeros(self.row), color='gray', label='gray')
        ax4.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['weekly_black'], color='black', label='black')
        ax4.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['weekly_red'], color='red', label='red')
        ymin, ymax = ax4.get_ylim()
        if abs(ymax) >= abs(ymin):
            ax4lim = abs(ymax)
        else:
            ax4lim = abs(ymin)
        ax4.set_ylim([-ax4lim, ax4lim])
        ax4.grid(True, which='both')
        ax4.tick_params(labelright=True)
        ax4.xaxis.set_major_locator(months)
        ax4.xaxis.set_major_formatter(monthsFmt)
        ax4.xaxis.set_ticklabels([])

        ax6 = plt.axes([0.05, 0.03, 0.9, 0.15])
        ax6.plot(self.short_norm_stock_df.index, np.zeros(self.row), color='gray', label='gray')
        ax6.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['weekly_black_deriv'])
        ax6.plot(self.short_norm_stock_df.index, self.short_norm_stock_df['weekly_red_deriv'])
        ymin, ymax = ax6.get_ylim()
        if abs(ymax) >= abs(ymin):
            ax6lim = abs(ymax)
        else:
            ax6lim = abs(ymin)
        ax6.set_ylim([-ax6lim, ax6lim])
        ax6.grid(True, which='both')
        ax6.tick_params(labelright=True)
        ax6.xaxis.set_major_locator(months)
        ax6.xaxis.set_major_formatter(monthsFmt)

        stock_gain = self.slope_crossover_history()
        if len(stock_gain) == 0:
            print("length zero")
            stock_gain = pd.DataFrame(data=[0])
        print(round(stock_gain, 1))
        print("Mean gain: ", round(stock_gain.mean(), 1), "%")

        i = 0
        bought = False
        for i in range(1, self.row):
            if self.weekly_black_slope_crossover_zero(i) == True:
                print(self.check_trends(i))
                ax.axvline(x=str(self.short_norm_stock_df.index[i]), color='green', linewidth=2)
                ax3.axvline(x=str(self.short_norm_stock_df.index[i]), color='green', linewidth=2)
                ax4.axvline(x=str(self.short_norm_stock_df.index[i]), color='green', linewidth=2)
                ax6.axvline(x=str(self.short_norm_stock_df.index[i]), color='green', linewidth=2)
                bought = True

            elif self.weekly_black_slope_cross_below_red(i) and bought == True: # or self.weekly_black_slope_cross_below_red(i) == True:
                ax.axvline(x=str(self.short_norm_stock_df.index[i]), color='red', linewidth=2)
                ax3.axvline(x=str(self.short_norm_stock_df.index[i]), color='red', linewidth=2)
                ax4.axvline(x=str(self.short_norm_stock_df.index[i]), color='red', linewidth=2)
                ax6.axvline(x=str(self.short_norm_stock_df.index[i]), color='red', linewidth=2)
                bought = False

            # elif self.weekly_black_slope_cross_below_zero(i) and bought == True: # or self.weekly_black_slope_cross_below_red(i) == True:
            #     ax.axvline(x=str(self.short_norm_stock_df.index[i]),  color='orange', linewidth=1)
            #     ax3.axvline(x=str(self.short_norm_stock_df.index[i]), color='orange', linewidth=1)
            #     ax4.axvline(x=str(self.short_norm_stock_df.index[i]), color='orange', linewidth=1)
            #     ax6.axvline(x=str(self.short_norm_stock_df.index[i]), color='orange', linewidth=1)
            #     bought = False

        return f1, stock_gain

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

    def slope_crossover_history(self):
        # i = 0
        crossed_up = False
        stock_gain = pd.DataFrame()
        stock_max = pd.DataFrame()
        max_close = 0
        for i in range(1, self.row):
            if self.weekly_black_slope_crossover_zero(i) == True and crossed_up == False:
                price_up = self.short_norm_stock_df['close'][i]
                crossed_up = True
            elif self.weekly_black_slope_cross_below_red(i) and crossed_up == True: # (self.weekly_black_slope_cross_below_zero(i) == True or self.weekly_black_slope_cross_below_red(i)) and crossed_up == True:
                price_down = self.short_norm_stock_df['close'][i]
                stock_gain = stock_gain.append(pd.Series((price_down - price_up) / price_up), ignore_index=True)
                stock_max = stock_max.append(pd.Series((max_close - price_up) / price_up), ignore_index=True)
                crossed_up = False
                price_up = 0
                max_close = 0
            if self.short_norm_stock_df['close'][i] > max_close and crossed_up == True:
                max_close = self.short_norm_stock_df['close'][i]

        stock_gain = stock_gain * 100
        stock_max = stock_max * 100
        print(stock_max)

        return stock_gain

    def weekly_black_slope_crossover_red_slope(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] > self.short_norm_stock_df['weekly_red_deriv'][effective_row] and \
                self.short_norm_stock_df['weekly_black_deriv'][effective_row-1] < self.short_norm_stock_df['weekly_red_deriv'][effective_row-1]:
            passCriteria = True
        return passCriteria

    def check_trends(self, row1):

        trend_status = [None] * 8
        #row1, col1 = self.short_norm_stock_df.shape

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

        # print(self.short_norm_stock_df['obv_volume'][row1 - 1],self.short_norm_stock_df['obv_volume'][row1 - 20 - 1],self.short_norm_stock_df['obv_volume'][row1 - 1],self.short_norm_stock_df['obv_volume'][row1 - 60 - 1])

        return trend_status

    def weekly_black_and_red_above_zero(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] > 0 and self.short_norm_stock_df['weekly_red_deriv'][effective_row-1] > -0.2:
            passCriteria = True
        return passCriteria

    def weekly_black_or_red_below_zero(self, effective_row):
        passCriteria = False
        if self.short_norm_stock_df['weekly_black_deriv'][effective_row] < 0 or self.short_norm_stock_df['weekly_red_deriv'][effective_row-1] < 0:
            passCriteria = True
        return passCriteria