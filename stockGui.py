import tkinter as tk
import datetime as dt
import tkinter.ttk
import matplotlib
import numpy as np
import os
from PIL import ImageTk, Image
matplotlib.use('TkAgg')
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class gui:

    def __init__(self, root):

        # Initialize station and title
        self.root = root
        self.root.title("Stock Analyzer")
        self.note = tk.ttk.Notebook(self.root)
        self.tab1 = tk.ttk.Frame(self.note)
        self.tab2 = tk.ttk.Frame(self.note)
        self.tab3 = tk.ttk.Frame(self.note)
        self.note.add(self.tab1, text="Tab One")
        self.note.add(self.tab2, text="Tab Two")
        self.note.add(self.tab3, text="Tab Three")
        self.note.grid(row=0, column=0)
        self.index = ""
        self.trend_status = [None] * 6
        self.pics_filepath = ""


        self.stock = "SPY"
        self.stock_gain = 0

        self.csv_variable = "zacks"

        self.now = dt.datetime.now()
        self.pull_list_date = str(dt.datetime(self.now.year, self.now.month, self.now.day))[0:10]
        self.pull_date = str(dt.datetime(self.now.year, self.now.month, self.now.day))[0:10]
        self.pull_list_stock = ""
        self.state = ""
        self.pull_list_dict = {}
        self.temp_stock_params = pd.DataFrame()
        self.pull_list_trend_dict = {}
        self.pull_list_stock_gain_dict = {}
        self.pull_list_stock_zacks_params_dict = {}
        self.market_performance = 0
        self.stock_performance = 0
        self.four_week_performance_dict = {}


        # Tab 1 - 1
        self.pull_stock_label = tk.Label(self.tab1, text=str("Check stock"), wraplength=150)
        self.pull_stock_label.grid(row=1, column = 0, pady=5)

        self.pull_stock_entry = tk.Entry(self.tab1)
        self.pull_stock_entry.insert(tk.END, self.stock)
        self.pull_stock_entry.bind('<Return>', self.update_pull_stock)
        self.pull_stock_entry.grid(row=1, column=1, pady=5)

        self.pull_stock_date = tk.Entry(self.tab1)
        self.pull_stock_date.insert(tk.END, self.pull_date)
        self.pull_stock_date.bind('<Return>', self.update_pull_stock)
        self.pull_stock_date.grid(row=1, column=2, pady=5)

        self.pull_stock_button = tk.Button(self.tab1, text="Enter")
        self.pull_stock_button.bind('<Button 1>', self.update_pull_stock)
        self.pull_stock_button.grid(row=1, column=3, pady=5)

        photo = Image.fromarray(np.uint8(np.empty((900, 1200, 3))))
        self.img_1 = ImageTk.PhotoImage(photo)
        self.label_img = tk.Label(self.tab1, image=self.img_1)
        self.label_img.image = self.img_1
        self.label_img.grid(row=2, column=0, rowspan=19, columnspan = 6, pady=5)

        self.zacks_rank_label = tk.Label(self.tab1, text=str("Zacks Rank: "))
        self.zacks_rank_label.grid(row=3, column = 8, pady=5)
        self.zacks_rank_value_label = tk.Label(self.tab1, text=str(self.trend_status[3]))
        self.zacks_rank_value_label.grid(row=3, column = 9, pady=5)

        self.value_score_label = tk.Label(self.tab1, text=str("Value Score: "))
        self.value_score_label.grid(row=4, column = 8, pady=5)
        self.value_score_value_label = tk.Label(self.tab1, text=str(self.trend_status[3]))
        self.value_score_value_label.grid(row=4, column = 9, pady=5)

        self.growth_score_label = tk.Label(self.tab1, text=str("Growth Score: "))
        self.growth_score_label.grid(row=5, column = 8, pady=5)
        self.growth_score_value_label = tk.Label(self.tab1, text=str(self.trend_status[3]))
        self.growth_score_value_label.grid(row=5, column = 9, pady=5)

        self.momentum_score_label = tk.Label(self.tab1, text=str("Momentum Score: "))
        self.momentum_score_label.grid(row=6, column = 8, pady=5)
        self.momentum_score_value_label = tk.Label(self.tab1, text=str(self.trend_status[3]))
        self.momentum_score_value_label.grid(row=6, column = 9, pady=5)

        self.VGM_score_label = tk.Label(self.tab1, text=str("VGM Score: "))
        self.VGM_score_label.grid(row=7, column = 8, pady=5)
        self.VGM_score_value_label = tk.Label(self.tab1, text=str(self.trend_status[3]))
        self.VGM_score_value_label.grid(row=7, column = 9, pady=5)

        self.industry_label = tk.Label(self.tab1, text=str("Industry Rank: "))
        self.industry_label.grid(row=8, column=8, pady=5)
        self.industry_value_label = tk.Label(self.tab1, text=str(self.trend_status[3]))
        self.industry_value_label.grid(row=8, column=9, pady=5)

        self.earnings_esp_label = tk.Label(self.tab1, text=str("Earnings ESP %: "))
        self.earnings_esp_label.grid(row=9, column = 8, pady=5)
        self.earnings_esp_value_label = tk.Label(self.tab1, text=str(self.trend_status[0]))
        self.earnings_esp_value_label.grid(row=9, column = 9, pady=5)

        self.last_earnings_esp_label = tk.Label(self.tab1, text=str("Last Earnings ESP %:"))
        self.last_earnings_esp_label.grid(row=10, column = 8, pady=5)
        self.last_earnings_esp_value_label = tk.Label(self.tab1, text=str(self.trend_status[1]))
        self.last_earnings_esp_value_label.grid(row=10, column = 9, pady=5)

        self.Q1_estimate_label = tk.Label(self.tab1, text=str("Q1 Estimate Change %: "))
        self.Q1_estimate_label.grid(row=11, column = 8, pady=5)
        self.Q1_estimate_value_label = tk.Label(self.tab1, text=str(self.trend_status[2]))
        self.Q1_estimate_value_label.grid(row=11, column = 9, pady=5)

        self.Q2_estimate_label = tk.Label(self.tab1, text=str("Q2 Estimate Change %: "))
        self.Q2_estimate_label.grid(row=12, column = 8, pady=5)
        self.Q2_estimate_value_label = tk.Label(self.tab1, text=str(self.trend_status[3]))
        self.Q2_estimate_value_label.grid(row=12, column = 9, pady=5)

        self.eps_date_label = tk.Label(self.tab1, text=str("Next EPS Date: "))
        self.eps_date_label.grid(row=13, column = 8, pady=5)
        self.eps_date_value_label = tk.Label(self.tab1, text=str(self.trend_status[3]))
        self.eps_date_value_label.grid(row=13, column = 9, pady=5)

        self.stock_performance_label = tk.Label(self.tab1, text=str("4 Week Stock Gain %: "))
        self.stock_performance_label.grid(row=14, column = 8, pady=5)
        self.stock_performance_value_label = tk.Label(self.tab1, text=str(self.stock_performance))
        self.stock_performance_value_label.grid(row=14, column = 9, pady=5)

        self.market_performance_label = tk.Label(self.tab1, text=str("4 Week SPY Gain %: "))
        self.market_performance_label.grid(row=15, column = 8, pady=5)
        self.market_performance_value_label = tk.Label(self.tab1, text=str(self.market_performance))
        self.market_performance_value_label.grid(row=15, column = 9, pady=5)

        self.stock_gain_listbox_tab1 = tk.Listbox(self.tab1)
        self.stock_gain_listbox_tab1.insert(tk.END, "stock_gain")
        self.stock_gain_listbox_tab1.grid(row=16, column=8, rowspan = 10, pady=5)


        # Tab 2
        self.pull_list_date_label = tk.Label(self.tab2, text=str("Pull List Date"))
        self.pull_list_date_label.grid(row=1, column = 1, pady=15)
        self.pull_list_date_entry = tk.Entry(self.tab2)
        self.pull_list_date_entry.insert(tk.END, self.pull_list_date)
        self.pull_list_date_entry.bind('<Return>', self.update_pull_list_date)
        self.pull_list_date_entry.grid(row=1, column=2, pady=15)
        self.pull_list_date_button = tk.Button(self.tab2, text="Enter")
        self.pull_list_date_button.bind('<Button 1>', self.update_pull_list_date)
        self.pull_list_date_button.grid(row=1, column=4, pady=15)

        self.csv_var = tk.StringVar(self.tab2)
        self.csv_var.set(self.csv_variable)  # default value

        self.csv_menu = tk.OptionMenu(self.tab2, self.csv_var, "total_stock_list", "industry_stock_list", "XLE-energy", "XLB-materials", "XLI-industrials", "XLP-consumer discretionary", "XLY-consumer staples", "XLV-health care", "XLF-financials", "XLK-technology", "XLC-telecommunication", "XLU-utilities", "XLRE-real estate", "portfolio", "zacks", "zacks test")
        self.csv_menu.grid(row=1, column=3, pady=15)


        self.label_img_2 = tk.Label(self.tab2, image=self.img_1)
        self.label_img_2.image = self.img_1
        self.label_img_2.grid(row=2, column=0, rowspan=19, columnspan = 6, pady=5)

        self.pull_list_status_label = tk.Label(self.tab2, text="Status: " + str(self.pull_list_stock), wraplength=150)
        self.pull_list_status_label.grid(row=1, column = 5, pady=15)

        self.zacks_rank_label_2 = tk.Label(self.tab2, text=str("Zacks Rank: "))
        self.zacks_rank_label_2.grid(row=3, column=6, padx=5, pady=5)
        self.zacks_rank_value_label_2 = tk.Label(self.tab2, text=str(self.trend_status[3]))
        self.zacks_rank_value_label_2.grid(row=3, column=7, padx=5, pady=5)

        self.value_score_label_2 = tk.Label(self.tab2, text=str("Value Score: "))
        self.value_score_label_2.grid(row=4, column=6, pady=5)
        self.value_score_value_label_2 = tk.Label(self.tab2, text=str(self.trend_status[3]))
        self.value_score_value_label_2.grid(row=4, column=7, pady=5)

        self.growth_score_label_2 = tk.Label(self.tab2, text=str("Growth Score: "))
        self.growth_score_label_2.grid(row=5, column=6, pady=5)
        self.growth_score_value_label_2 = tk.Label(self.tab2, text=str(self.trend_status[3]))
        self.growth_score_value_label_2.grid(row=5, column=7, pady=5)

        self.momentum_score_label_2 = tk.Label(self.tab2, text=str("Momentum Score: "))
        self.momentum_score_label_2.grid(row=6, column=6, pady=5)
        self.momentum_score_value_label_2 = tk.Label(self.tab2, text=str(self.trend_status[3]))
        self.momentum_score_value_label_2.grid(row=6, column=7, pady=5)

        self.VGM_score_label_2 = tk.Label(self.tab2, text=str("VGM Score: "))
        self.VGM_score_label_2.grid(row=7, column=6, pady=5)
        self.VGM_score_value_label_2 = tk.Label(self.tab2, text=str(self.trend_status[3]))
        self.VGM_score_value_label_2.grid(row=7, column=7, pady=5)

        self.industry_label_2 = tk.Label(self.tab2, text=str("Industry Rank: "))
        self.industry_label_2.grid(row=8, column=6, pady=5)
        self.industry_value_label_2 = tk.Label(self.tab2, text=str(self.trend_status[3]))
        self.industry_value_label_2.grid(row=8, column=7, pady=5)

        self.earnings_esp_label_2 = tk.Label(self.tab2, text=str("Earnings ESP %: "))
        self.earnings_esp_label_2.grid(row=9, column=6, pady=5)
        self.earnings_esp_value_label_2 = tk.Label(self.tab2, text=str(self.trend_status[0]))
        self.earnings_esp_value_label_2.grid(row=9, column=7, pady=5)

        self.last_earnings_esp_label_2 = tk.Label(self.tab2, text=str("Last Earnings ESP %:"))
        self.last_earnings_esp_label_2.grid(row=10, column=6, pady=5)
        self.last_earnings_esp_value_label_2 = tk.Label(self.tab2, text=str(self.trend_status[1]))
        self.last_earnings_esp_value_label_2.grid(row=10, column=7, pady=5)

        self.Q1_estimate_label_2 = tk.Label(self.tab2, text=str("Q1 Estimate Change %: "))
        self.Q1_estimate_label_2.grid(row=11, column=6, pady=5)
        self.Q1_estimate_value_label_2 = tk.Label(self.tab2, text=str(self.trend_status[2]))
        self.Q1_estimate_value_label_2.grid(row=11, column=7, pady=5)

        self.Q2_estimate_label_2 = tk.Label(self.tab2, text=str("Q2 Estimate Change %: "))
        self.Q2_estimate_label_2.grid(row=12, column=6, pady=5)
        self.Q2_estimate_value_label_2 = tk.Label(self.tab2, text=str(self.trend_status[3]))
        self.Q2_estimate_value_label_2.grid(row=12, column=7, pady=5)

        self.eps_date_label_2 = tk.Label(self.tab2, text=str("Next EPS Date: "))
        self.eps_date_label_2.grid(row=13, column = 6, pady=5)
        self.eps_date_value_label_2 = tk.Label(self.tab2, text=str(self.trend_status[3]))
        self.eps_date_value_label_2.grid(row=13, column = 7, pady=5)

        self.stock_performance_label_2 = tk.Label(self.tab2, text=str("4 Week Stock Gain %: "))
        self.stock_performance_label_2.grid(row=14, column = 6, pady=5)
        self.stock_performance_value_label_2 = tk.Label(self.tab2, text=str(self.stock_performance))
        self.stock_performance_value_label_2.grid(row=14, column = 7, pady=5)

        self.market_performance_label_2 = tk.Label(self.tab2, text=str("4 Week SPY Gain %: "))
        self.market_performance_label_2.grid(row=15, column = 6, pady=5)
        self.market_performance_value_label_2 = tk.Label(self.tab2, text=str(self.market_performance))
        self.market_performance_value_label_2.grid(row=15, column = 7, pady=5)

        self.buy_listbox = tk.Listbox(self.tab2, height=30)
        self.buy_listbox.grid(row=3, column=8, rowspan=10, padx=5, pady=15)

        self.sell_listbox = tk.Listbox(self.tab2, height=30)
        self.sell_listbox.grid(row=3, column=9, rowspan=10, padx=5, pady=15)

        self.yes_pull_list_button = tk.Button(self.tab2, text="Yes", command=self.update_yes_pull_list)
        self.yes_pull_list_button.grid(row=3, column=10, pady=5)
        self.maybe_pull_list_button = tk.Button(self.tab2, text="Maybe", command=self.update_maybe_pull_list)
        self.maybe_pull_list_button.grid(row=4, column=10, pady=5)
        self.no_pull_list_button = tk.Button(self.tab2, text="No", command=self.update_no_pull_list)
        self.no_pull_list_button.grid(row=5, column=10, pady=5)

        self.yes_listbox = tk.Listbox(self.tab2, height=15)
        self.yes_listbox.grid(row=16, column=8, padx=5, pady=15)
        self.maybe_listbox = tk.Listbox(self.tab2, height=15)
        self.maybe_listbox.grid(row=16, column=9, padx=5, pady=15)
        self.no_listbox = tk.Listbox(self.tab2, height=15)
        self.no_listbox.grid(row=16, column=10, padx=5, pady=15)

        self.update_decisions_button = tk.Button(self.tab2, text="Save Decisions", command=self.save_stock_decisions)
        self.update_decisions_button.grid(row=17, column=9, pady=15)

        self.stock_gain_listbox_tab2 = tk.Listbox(self.tab2, height=15)
        self.stock_gain_listbox_tab2.insert(tk.END, "stock_gain")
        self.stock_gain_listbox_tab2.grid(row=16, column=6, pady=15)

        self.test_var = ""

        self.backtest_button = tk.Button(self.tab3, text="Back test")
        self.backtest_button.bind('<Button 1>', self.update_backtest)
        self.backtest_button.grid(row=1, column=1, pady=15)


    def update_every_time(self):
        buy_index = self.buy_listbox.curselection()
        sell_index = self.sell_listbox.curselection()
        yes_index = self.yes_listbox.curselection()
        maybe_index = self.maybe_listbox.curselection()
        no_index = self.no_listbox.curselection()
        self.csv_variable = self.csv_var.get()
        selected = False
        old_stock = self.pull_list_stock
        if len(buy_index) > 0:
            self.pull_list_stock = self.buy_listbox.get(buy_index[0])
            selected = True
        elif len(sell_index) > 0:
            self.pull_list_stock = self.sell_listbox.get(sell_index[0])
            selected = True
        elif len(yes_index) > 0:
            self.pull_list_stock = self.yes_listbox.get(yes_index[0])
            selected = True
        elif len(maybe_index) > 0:
            self.pull_list_stock = self.maybe_listbox.get(maybe_index[0])
            selected = True
        elif len(no_index) > 0:
            self.pull_list_stock = self.no_listbox.get(no_index[0])
            selected = True
        else:
            selected = False

        # temp_stock_params = self.zacks_total_params[self.zacks_total_params['Ticker'] == str(self.stock.upper())]
        # temp_stock_params.reset_index(drop=True)



        if selected == True and (self.pull_list_stock != old_stock):
            self.plot_fig_tab2()
            self.plot_stock_gain_tab2(self.pull_list_stock_gain_dict[self.pull_list_stock])
            # self.one_month_trend_value_2.config(text=str(self.pull_list_trend_dict[self.pull_list_stock][0]))
            # self.three_month_trend_value_2.config(text=str(self.pull_list_trend_dict[self.pull_list_stock][1]))
            # self.one_month_obv_trend_value_2.config(text=str(self.pull_list_trend_dict[self.pull_list_stock][2]))
            # self.three_month_obv_trend_value_2.config(text=str(self.pull_list_trend_dict[self.pull_list_stock][3]))


        # self.one_month_trend_value_1.config(text=str(self.trend_status[0]))
        # self.three_month_trend_value_1.config(text=str(self.trend_status[1]))
        # self.one_month_obv_trend_value_1.config(text=str(self.trend_status[2]))
        # self.three_month_obv_trend_value_1.config(text=str(self.trend_status[3]))


    def update_pull_list_status(self):
        self.pull_list_status_label.config(text="Status: " + str(self.pull_list_stock))


    def plot_fig_tab1(self):

        photo = Image.open(os.path.join(self.pics_filepath, str(self.stock) + '.png'))
        self.img_1 = ImageTk.PhotoImage(photo)
        self.label_img.configure(image=self.img_1)

        row_num, garbage_col = self.temp_stock_params.shape

        if row_num == 1:

            index_val = int(self.temp_stock_params.index.values.astype(int))

            self.earnings_esp_value_label.config(text=str(self.temp_stock_params['Earnings ESP'][index_val]))
            self.last_earnings_esp_value_label.config(text=str(self.temp_stock_params['Last EPS Surprise (%)'][index_val]))
            self.Q1_estimate_value_label.config(text=str(self.temp_stock_params['% Change Q1 Est. (4 weeks)'][index_val]))
            self.Q2_estimate_value_label.config(text=str(self.temp_stock_params['% Change Q2 Est. (4 weeks)'][index_val]))
            self.zacks_rank_value_label.config(text=str(self.temp_stock_params['Zacks Rank'][index_val]))
            self.value_score_value_label.config(text=str(self.temp_stock_params['Value Score'][index_val]))
            self.growth_score_value_label.config(text=str(self.temp_stock_params['Growth Score'][index_val]))
            self.momentum_score_value_label.config(text=str(self.temp_stock_params['Momentum Score'][index_val]))
            self.VGM_score_value_label.config(text=str(self.temp_stock_params['VGM Score'][index_val]))
            self.industry_value_label.config(text=str(self.temp_stock_params['Zacks Industry Rank'][index_val]))
            self.eps_date_value_label.config(text=str(self.temp_stock_params['Next EPS Report Date '][index_val])[4:6] + '/' + str(self.temp_stock_params['Next EPS Report Date '][index_val])[6:8] + '/' + str(self.temp_stock_params['Next EPS Report Date '][index_val])[0:4])
            self.stock_performance_value_label.config(text=str(self.stock_performance))
            self.market_performance_value_label.config(text=str(self.market_performance))


        elif row_num == 0:
            self.earnings_esp_value_label.config(text='N/A')
            self.last_earnings_esp_value_label.config(text='N/A')
            self.Q1_estimate_value_label.config(text='N/A')
            self.Q2_estimate_value_label.config(text='N/A')
            self.zacks_rank_value_label.config(text='N/A')
            self.value_score_value_label.config(text='N/A')
            self.growth_score_value_label.config(text='N/A')
            self.momentum_score_value_label.config(text='N/A')
            self.VGM_score_value_label.config(text='N/A')
            self.industry_value_label.config(text='N/A')
            self.eps_date_value_label.config(text='N/A')


    def plot_stock_gain_tab1(self, stock_gain):

        self.stock_gain_listbox_tab1.delete(0, tk.END)
        for i in range(1,len(stock_gain)+1):
            self.stock_gain_listbox_tab1.insert(tk.END, "Buy sign " + str(i) + ": " + str(round(stock_gain[0][i-1],1)) + "%")
        self.stock_gain_listbox_tab1.insert(tk.END, "")
        self.stock_gain_listbox_tab1.insert(tk.END, "Average gain: " + str(round(stock_gain[0].mean(),1)) + "%")


    def plot_fig_tab2(self):

        photo = Image.open(os.path.join(self.pics_filepath, str(self.pull_list_stock) + '.png'))
        self.img_2 = ImageTk.PhotoImage(photo)
        self.label_img_2.configure(image=self.img_2)



        zacks_temp = self.pull_list_stock_zacks_params_dict[self.pull_list_stock]

        row_num, garbage_col = zacks_temp.shape
        print(zacks_temp, row_num)

        if row_num == 1:

            index_val = int(zacks_temp.index.values.astype(int))

            self.earnings_esp_value_label_2.config(text=str(zacks_temp['Earnings ESP'][index_val]))
            self.last_earnings_esp_value_label_2.config(text=str(zacks_temp['Last EPS Surprise (%)'][index_val]))
            self.Q1_estimate_value_label_2.config(text=str(zacks_temp['% Change Q1 Est. (4 weeks)'][index_val]))
            self.Q2_estimate_value_label_2.config(text=str(zacks_temp['% Change Q2 Est. (4 weeks)'][index_val]))
            self.zacks_rank_value_label_2.config(text=str(zacks_temp['Zacks Rank'][index_val]))
            self.value_score_value_label_2.config(text=str(zacks_temp['Value Score'][index_val]))
            self.growth_score_value_label_2.config(text=str(zacks_temp['Growth Score'][index_val]))
            self.momentum_score_value_label_2.config(text=str(zacks_temp['Momentum Score'][index_val]))
            self.VGM_score_value_label_2.config(text=str(zacks_temp['VGM Score'][index_val]))
            self.industry_value_label_2.config(text=str(zacks_temp['Zacks Industry Rank'][index_val]))
            self.eps_date_value_label_2.config(text=str(zacks_temp['Next EPS Report Date '][index_val])[4:6] + '/' + str(zacks_temp['Next EPS Report Date '][index_val])[6:8] + '/' + str(zacks_temp['Next EPS Report Date '][index_val])[0:4])
            self.stock_performance_value_label_2.config(text=str(self.four_week_performance_dict[self.pull_list_stock]))
            self.market_performance_value_label_2.config(text=str(self.market_performance))

        elif row_num == 0:
            self.earnings_esp_value_label_2.config(text='N/A')
            self.last_earnings_esp_value_2_label.config(text='N/A')
            self.Q1_estimate_value_label_2.config(text='N/A')
            self.Q2_estimate_value_label_2.config(text='N/A')
            self.zacks_rank_value_label_2.config(text='N/A')
            self.value_score_value_label_2.config(text='N/A')
            self.growth_score_value_label_2.config(text='N/A')
            self.momentum_score_value_label_2.config(text='N/A')
            self.VGM_score_value_label_2.config(text='N/A')
            self.industry_value_label_2.config(text='N/A')
            self.eps_date_value_label_2.config(text='N/A')

    def plot_stock_gain_tab2(self, stock_gain):

        self.stock_gain_listbox_tab2.delete(0, tk.END)
        for i in range(1,len(stock_gain)+1):
            self.stock_gain_listbox_tab2.insert(tk.END, "Buy sign " + str(i) + ": " + str(round(stock_gain[0][i-1],1)) + "%")
        self.stock_gain_listbox_tab2.insert(tk.END, "")
        self.stock_gain_listbox_tab2.insert(tk.END, "Average gain: " + str(round(stock_gain[0].mean(),1)) + "%")


    def update_pull_list_date(self, event):

        self.pull_list_date = self.pull_list_date_entry.get()
        self.buy_listbox.delete(0, tk.END)
        self.sell_listbox.delete(0, tk.END)
        self.yes_listbox.delete(0, tk.END)
        self.maybe_listbox.delete(0, tk.END)
        self.no_listbox.delete(0, tk.END)
        self.state = "pull list"

    def update_backtest(self, event):

        self.state = "back test"

    def update_pull_stock(self, event):

        self.stock = self.pull_stock_entry.get()
        self.pull_date = self.pull_stock_date.get()
        self.state = "pull"


    def update_yes_pull_list(self):

        self.yes_listbox.insert(tk.END, self.pull_list_stock)


    def update_maybe_pull_list(self):

        self.maybe_listbox.insert(tk.END, self.pull_list_stock)


    def update_no_pull_list(self):

        self.no_listbox.insert(tk.END, self.pull_list_stock)


    def update_buy_list(self):

        self.buy_listbox.insert(tk.END, self.pull_list_stock)


    def update_sell_list(self):

        self.sell_listbox.insert(tk.END, self.pull_list_stock)


    def save_stock_decisions(self):

        save_yes_decisions = self.yes_listbox.get(0, tk.END)
        length = len(save_yes_decisions)
        save_yes_decisions = np.array(save_yes_decisions).reshape(length, 1)
        yes_column = np.full((length, 1), "yes")
        save_yes_decisions = np.concatenate((save_yes_decisions, yes_column), axis=1)

        save_maybe_decisions = self.maybe_listbox.get(0, tk.END)
        length = len(save_maybe_decisions)
        save_maybe_decisions = np.array(save_maybe_decisions).reshape(length, 1)
        maybe_column = np.full((length, 1), "maybe")
        save_maybe_decisions = np.concatenate((save_maybe_decisions, maybe_column), axis=1)

        save_no_decisions = self.no_listbox.get(0, tk.END)
        length = len(save_no_decisions)
        save_no_decisions = np.array(save_no_decisions).reshape(length, 1)
        no_column = np.full((length, 1), "no")
        save_no_decisions = np.concatenate((save_no_decisions, no_column), axis=1)

        decision_stack = np.concatenate((save_yes_decisions, save_maybe_decisions))
        decision_stack = np.concatenate((decision_stack, save_no_decisions))

        filepath = os.getcwd()
        decisions_path = '\\decisions\\'
        filepath = filepath + decisions_path


        np.savetxt(os.path.join(filepath,"" + str(self.pull_list_date) + " decisions.csv"), decision_stack, fmt = ['%s', '%s'], delimiter=",")

        print(decision_stack)
