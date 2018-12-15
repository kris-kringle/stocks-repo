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
from iexfinance import Stock
from dateutil import parser
import processStockData as psd

# exec(open('stock_analyzer_4.py').read())

def pull(stock):
	
	now=dt.datetime.now()
	start = datetime(now.year-4, 1, 1)
	end = datetime(now.year, now.month, now.day)
	
	hist_prices(stock,start,end)
	#print(f_reg.index[row-1])
	#plot_chart(stock)

def plot_chart(_stock):
	
	global kris_index, ax4lim
	f1, (ax, ax2, ax3, ax4, ax6) = plt.subplots(5,1,figsize = (12,8),sharex=True)
	
	ax.plot(f_reg.index, f_reg['ema26'], color = 'purple', label = 'ema26',linewidth=1.0)
	ax.plot(f_reg.index, f_reg['ema26_highenv'], color = 'purple', label = 'ema26_highenv',linewidth=1.0)
	ax.plot(f_reg.index, f_reg['ema26_lowenv'], color = 'purple', label = 'ema26_lowenv',linewidth=1.0)
	ax.plot(f_reg.index, f_reg['close'])
	ax.set_position(matplotlib.transforms.Bbox([[0.1,0.48],[0.92,.96]]))
	ax.grid(True,which = 'both')
	ax.set_title(_stock)
	ax.minorticks_on()
	ax.tick_params(labelright=True)

	ax2.set_position(matplotlib.transforms.Bbox([[0.1,0.38],[0.92,0.48]]))
	ax5 = ax2.twinx()
	ax5.set_position(matplotlib.transforms.Bbox([[0.1,0.38],[0.92,0.48]]))
	for j in range(1,row):
		if f_reg['close'][j] >= f_reg['close'][j-1]:
			ax2.bar(f_reg.index[j], f_reg['volume'][j].astype('float'),color='black',width=1)
		elif f_reg['close'][j] < f_reg['close'][j-1]:
			ax2.bar(f_reg.index[j], f_reg['volume'][j].astype('float'),color='red',width=1)
	
	ax5.plot(f.index[data_years:], f['obv_volume'][data_years:])
	ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax2.grid(True,which = 'both')
	
	ax3.plot(f_reg.index, f_reg['daily_black'], color = 'black', label = 'black')
	ax3.plot(f_reg.index, f_reg['daily_red'], color = 'red', label = 'red')
	ax3.plot(f_reg.index, np.zeros(row), color = 'black', label = 'black')
	ax3.grid(True,which = 'both')
	ax3.set_position(matplotlib.transforms.Bbox([[0.1,0.26],[0.92,0.36]]))
	ymin, ymax = ax3.get_ylim()
	if abs(ymax) >= abs(ymin):
		ax3lim = abs(ymax)
	else:
		ax3lim = abs(ymin)
	ax3.set_ylim([-ax3lim,ax3lim])
	
	
	ax4.plot(f_reg.index, f_reg['weekly_black'], color = 'black', label = 'black')
	ax4.plot(f_reg.index, f_reg['weekly_red'], color = 'red', label = 'red')
	ax4.plot(f_reg.index, np.zeros(row), color = 'black', label = 'black')
	ax4.grid(True,which = 'both')
	ax4.set_position(matplotlib.transforms.Bbox([[0.1,0.16],[0.92,0.26]]))
	ymin, ymax = ax4.get_ylim()
	if abs(ymax) >= abs(ymin):
		ax4lim = abs(ymax)
	else:
		ax4lim = abs(ymin)
	ax4.set_ylim([-ax4lim,ax4lim])
	
	mondays = WeekdayLocator(MONDAY)
	months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
	monthsFmt = DateFormatter("%b")
	ax.xaxis.set_major_locator(months)
	ax.xaxis.set_major_formatter(monthsFmt)
		
	ax6.set_position(matplotlib.transforms.Bbox([[0.1,0.04],[0.92,.15]]))
	ax6.plot(f_reg.index,f_reg['weekly_black_deriv'])
	ax6.plot(f_reg.index,f_reg['weekly_red_deriv'])
	ax6.plot(f_reg.index, np.zeros(row), color = 'black', label = 'black')
	ax6.grid(True,which = 'both')
	ymin, ymax = ax6.get_ylim()
	if abs(ymax) >= abs(ymin):
		ax6lim = abs(ymax)
	else:
		ax6lim = abs(ymin)
	ax6.set_ylim([-ax6lim,ax6lim])
	
	weekly_slope_crossover_history()
	print(round(stock_gain,1))
	print("Mean gain: ", round(stock_gain.mean(),1), "%")
	
	criteria = "weekly slope crossover up"	
	i=0
	for i in range(1,row):
		if criteria == "weekly slope crossover up":
			result = weekly_slope_crossover_up(i)
		else:
			result = False
		if result == True:
			ax.axvline(x=str(f_reg.index[i]), color = 'green',linewidth=1)
			ax3.axvline(x=str(f_reg.index[i]), color = 'green',linewidth=1)
			ax4.axvline(x=str(f_reg.index[i]), color = 'green',linewidth=1)
			ax5.axvline(x=str(f_reg.index[i]), color = 'green',linewidth=1)
			ax6.axvline(x=str(f_reg.index[i]), color = 'green',linewidth=1)
			#print(f_reg.index[i])
	
	criteria = "weekly slope crossover down"
	i=0
	for i in range(1,row):
		if criteria == "weekly slope crossover down":
			result = weekly_slope_crossover_down(i)
		else:
			result = False
		if result == True:
			ax.axvline(x=str(f_reg.index[i]), color = 'red',linewidth=1)
			ax3.axvline(x=str(f_reg.index[i]), color = 'red',linewidth=1)
			ax4.axvline(x=str(f_reg.index[i]), color = 'red',linewidth=1)
			ax5.axvline(x=str(f_reg.index[i]), color = 'red',linewidth=1)
			ax6.axvline(x=str(f_reg.index[i]), color = 'red',linewidth=1)
			#print(f_reg.index[i])
	
	plt.show(block=False)
	
def hist_prices(stock,start,end,pull_years=1):

	#global f, row, f_short, ohlc, ohlc_vol, data_years, avg_volume, env_percent, ax_max, kris_index, f_reg
	stock = stock.upper()
	try:
		f = data.DataReader(stock, 'iex', start, end)
	except:
		print("ignore this stock---------------")
		#raise
	row, col = f.shape
	now=dt.datetime.now()
	end = datetime(now.year, now.month, now.day)
	if f.index[row-1] != end:
		stock_quote=Stock(stock)
		try:
			stock_pd=stock_quote.get_quote()
		except:
			print("ignore todays prices----------")
			#raise
		stock_date = stock_pd['latestTime']
		stock_date = str(parser.parse(stock_date))[0:10]
		stock_date = pd.to_datetime(stock_date)
		if stock_date == end:
			stock_open = stock_pd['open']
			stock_high = stock_pd['high']
			stock_low = stock_pd['low']
			stock_close = stock_pd['close']
			stock_volume = stock_pd['latestVolume']
			
			f_day = {'open':[stock_open],'high':[stock_high],'low':[stock_low],'close':[stock_close],'volume':[stock_volume]}
			date_index = pd.date_range(stock_date,periods=1, freq='D')
			f_day = pd.DataFrame(data=f_day, index=date_index)
			f_day.index.name = 'date'
			f = f.append(f_day.ix[0])
	
	f = f.set_index(pd.to_datetime(f.index))
	data_years = row - int(round(253*pull_years,0))
	f['date'] = f.index.map(mdates.date2num)
	
	# calculate MACD parameters
	f['ema26'] = f['close'].ewm(span=26,adjust=False).mean()
	f['ema12'] = f['close'].ewm(span=12,adjust=False).mean()
	f['daily_black'] = f['ema12'] - f['ema26']
	f['daily_red'] = f['daily_black'].ewm(span=18,adjust=False).mean()
	f['ema60'] = f['close'].ewm(span=60,adjust=False).mean()
	f['ema130'] = f['close'].ewm(span=130,adjust=False).mean()
	f['weekly_black'] = f['ema60'] - f['ema130']
	f['weekly_red'] = f['weekly_black'].ewm(span=45,adjust=False).mean()

	env_percent = .04
	f['fit_env'] = 0
	while f['fit_env'].mean() <= .9:
		f['ema26_highenv'] = f['ema26']*(1 + env_percent)
		f['ema26_lowenv'] = f['ema26']*(1 - env_percent)
		f['fit_env'] = np.where((f['ema26_highenv'] >= f['high']) & (f['ema26_lowenv'] <= f['low']),1,0)
		env_percent += .005
	
	ax_max = abs(f['weekly_black'][data_years:].max()*1.05)
	if abs(f['weekly_black'][data_years:].min()*1.05) > ax_max:
		ax_max = abs(f['weekly_black'][data_years:].min()*1.05)
	elif abs(f['weekly_red'][data_years:].max()*1.05)  > ax_max:
		ax_max = abs(f['weekly_red'][data_years:].max()*1.05)
	elif abs(f['weekly_red'][data_years:].min()*1.05)  > ax_max:
		ax_max = abs(f['weekly_red'][data_years:].max()*1.05)
	
	if row > 6:
		kris_index = ((f.ix[row-1]['weekly_black']-f.ix[row-5]['weekly_black'])/4)/ax_max
	else:
		kris_index = 0
	
	pd.options.mode.chained_assignment = None
	f['obv_volume'] = 0
	f['obv_volume'][0] = f['volume'][0]
	for i in range(1, row):
		if f['close'][i] > f['close'][i-1]:
			f['obv_volume'][i] = f['obv_volume'][i-1] + f['volume'][i]
		elif f['close'][i] < f['close'][i-1]:
			f['obv_volume'][i] = f['obv_volume'][i-1] - f['volume'][i]
		else:
			f['obv_volume'][i] = f['obv_volume'][i-1]
	
	f['volume'] = f['volume']/f['volume'].max()
	f['obv_volume'] = f['obv_volume']/f['obv_volume'].max()
	
	avg_volume = f['volume'][row-10:].mean()
	f['close_from_top_env'] = f['ema26_highenv']-f['close']
	if row > 5:
		f['weekly_black_deriv'] = np.gradient(f['weekly_black'])
		f['weekly_black_deriv'] = f['weekly_black_deriv'].ewm(span=8,adjust=False).mean()
		f['weekly_black_deriv'] = f['weekly_black_deriv']/f['weekly_black_deriv'].max()
		f['weekly_red_deriv'] = np.gradient(f['weekly_red'])
		f['weekly_red_deriv'] = f['weekly_red_deriv'].ewm(span=8,adjust=False).mean()
		f['weekly_red_deriv'] = f['weekly_red_deriv']/f['weekly_red_deriv'].max()
		
	f['decision']=0
	f['stock-name']=stock
	f['performance']=0
	
	f_short = f[data_years:]

	f_reg = f[data_years:]
	row, col = f_reg.shape
	f_reg['open'] = f_reg['open']/f_reg['open'].max()
	f_reg['high'] = f_reg['high']/f_reg['high'].max()
	f_reg['low'] = f_reg['low']/f_reg['low'].max()
	f_reg['close'] = f_reg['close']/f_reg['close'].max()
	f_reg['ema26'] = f_reg['close'].ewm(span=26,adjust=False).mean()
	f_reg['ema12'] = f_reg['close'].ewm(span=12,adjust=False).mean()
	f_reg['daily_black'] = f_reg['daily_black']/f_reg['daily_black'].max()
	f_reg['daily_red'] = f_reg['daily_red']/f_reg['daily_red'].max()
	f_reg['ema60'] = f_reg['close'].ewm(span=60,adjust=False).mean()
	f_reg['ema130'] = f_reg['close'].ewm(span=130,adjust=False).mean()
	f_reg['weekly_black'] = f_reg['weekly_black']/f_reg['weekly_black'].max()
	f_reg['weekly_red'] = f_reg['weekly_red']/f_reg['weekly_red'].max()
	env_percent = .04
	f_reg['fit_env'] = 0
	while f_reg['fit_env'].mean() <= .9:
		f_reg['ema26_highenv'] = f_reg['ema26']*(1 + env_percent)
		f_reg['ema26_lowenv'] = f_reg['ema26']*(1 - env_percent)
		f_reg['fit_env'] = np.where((f_reg['ema26_highenv'] >= f_reg['high']) & (f_reg['ema26_lowenv'] <= f_reg['low']),1,0)
		env_percent += .0005
	ax_max = abs(f_reg['weekly_black'].max()*1.05)
	if abs(f_reg['weekly_black'].min()*1.05) > ax_max:
		ax_max = abs(f_reg['weekly_black'].min()*1.05)
	elif abs(f_reg['weekly_red'].max()*1.05)  > ax_max:
		ax_max = abs(f_reg['weekly_red'].max()*1.05)
	elif abs(f_reg['weekly_red'].min()*1.05)  > ax_max:
		ax_max = abs(f_reg['weekly_red'].max()*1.05)
	if row > 6:
		kris_index = ((f_reg.ix[row-1]['weekly_black']-f_reg.ix[row-5]['weekly_black'])/4)/ax_max
	else:
		kris_index = 0
	pd.options.mode.chained_assignment = None
	f_reg['obv_volume'] = 0
	f_reg['obv_volume'][0] = f_reg['volume'][0]
	for i in range(1, row):
		if f_reg['close'][i] > f_reg['close'][i-1]:
			f_reg['obv_volume'][i] = f_reg['obv_volume'][i-1] + f_reg['volume'][i]
		elif f['close'][i] < f['close'][i-1]:
			f_reg['obv_volume'][i] = f_reg['obv_volume'][i-1] - f_reg['volume'][i]
		else:
			f_reg['obv_volume'][i] = f_reg['obv_volume'][i-1]
	f_reg['volume'] = f_reg['volume']/f_reg['volume'].max()
	f_reg['obv_volume'] = f_reg['obv_volume']/f_reg['obv_volume'].max()
	avg_volume = f_reg['volume'][row-10:].mean()
	f_reg['close_from_top_env'] = f_reg['ema26_highenv']-f_reg['close']
	if row > 5:
		f_reg['weekly_black_deriv'] = np.gradient(f_reg['weekly_black'])
		f_reg['weekly_black_deriv'] = f_reg['weekly_black_deriv'].ewm(span=8,adjust=False).mean()
		f_reg['weekly_black_deriv'] = f_reg['weekly_black_deriv']/f_reg['weekly_black_deriv'].max()
		f_reg['weekly_red_deriv'] = np.gradient(f_reg['weekly_red'])
		f_reg['weekly_red_deriv'] = f_reg['weekly_red_deriv'].ewm(span=8,adjust=False).mean()
		f_reg['weekly_red_deriv'] = f_reg['weekly_red_deriv']/f_reg['weekly_red_deriv'].max()
	
def weekly_slope_crossover_up(effective_row):
	passCriteria = False
	if f_reg['weekly_black_deriv'][effective_row] > f_reg['weekly_red_deriv'][effective_row] and f_reg['weekly_black_deriv'][effective_row-1] < f_reg['weekly_red_deriv'][effective_row-1]:
		passCriteria = True
	return passCriteria


def weekly_slope_crossover_history():
	global stock_gain
	i = 0
	crossed_up = False
	stock_gain = pd.DataFrame()
	for i in range(1, row):
		criteria = "weekly slope crossover up"
		if weekly_slope_crossover_up(i) == True and crossed_up == False:
			price_up = f_reg['close'][i - 1]
			crossed_up = True

		criteria = "weekly slope crossover down"
		if weekly_slope_crossover_down(i) == True and crossed_up == True:
			price_down = f_reg['close'][i]
			stock_gain = stock_gain.append(pd.Series((price_down - price_up) / price_up), ignore_index=True)
			crossed_up = False
			price_up = 0
			price_down = 0
	stock_gain = stock_gain * 100

def weekly_slope_crossover_down(effective_row):
	passCriteria = False
	if f_reg['weekly_black_deriv'][effective_row] < f_reg['weekly_red_deriv'][effective_row] and f_reg['weekly_black_deriv'][effective_row-1] > f_reg['weekly_red_deriv'][effective_row-1]:
		passCriteria = True
	return passCriteria

def pull_todays_buys(year="",month="",day=""):
	
	global tested_stocks, y, test
	criteria="weekly slope crossover up"
	stock_list=pd.read_csv('total_stock_list.csv',header=None)
	end = datetime(year,month,day)		
	start = datetime(year-4, 1, 1)

	tested_stocks = pd.DataFrame()
	sell_stocks = pd.DataFrame()
	hist_prices('aapl',start,end)
	reg_data = pd.DataFrame(index=range(0),columns=range(24))
	reg_data.columns = f.columns
	x=0
	for x in range(0,len(stock_list)):
		stock = stock_list[0][x]
		hist_prices(stock,start,end)
		weekly_slope_crossover_history()
		print(stock)
		if row > 200:
			result = weekly_slope_crossover_up(row-1)
			if result == True and len(stock_gain) != 0 and float(stock_gain.min()) > -20:
				tested_stocks=tested_stocks.append(pd.Series(stock), ignore_index=True)
				reg_data=reg_data.append(f_short.ix[row-1])
				print("possible buy", tested_stocks)
			result = weekly_slope_crossover_down(row-1)
			if result == True:
				sell_stocks=sell_stocks.append(pd.Series(stock), ignore_index=True)
				print("sell", sell_stocks)
			
	reg_data.to_csv('train_data - ' + str(end)[0:10] + '-no decision.csv')
	
	train_data = pd.DataFrame(index=range(0),columns=range(24))
	train_data.columns = f.columns
	y=0
	for y in range(0,len(tested_stocks)):
		stock = tested_stocks[0][y]
		hist_prices(stock,start,end)
		plot_chart(stock)

		train_data=train_data.append(f_short.ix[row-1])
		train_data['decision'][y] = input('Enter decision for ' + str(stock) + ': ').upper()
		plt.close()
		
	train_data.to_csv('train_data - ' + str(end)[0:10] + '.csv')	
		