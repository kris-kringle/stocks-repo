import tkinter as tk
import datetime as dt
import tkinter.ttk
import matplotlib
import numpy as np
import os
from PIL import ImageTk, Image
matplotlib.use('TkAgg')
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

        self.stock = "DIA"
        self.stock_gain = 0

        self.csv_variable = "total_stock_list"

        self.now = dt.datetime.now()
        self.pull_list_date = str(dt.datetime(self.now.year, self.now.month, self.now.day))[0:10]
        self.pull_date = str(dt.datetime(self.now.year, self.now.month, self.now.day))[0:10]
        self.pull_list_stock = ""
        self.state = ""
        self.pull_list_dict = {}
        self.pull_list_trend_dict = {}
        self.pull_list_stock_gain_dict = {}


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

        self.one_month_trend_text_1 = tk.Label(self.tab1, text=str("One Month Price Trend"))
        self.one_month_trend_text_1.grid(row=8, column = 8, pady=5)
        self.one_month_trend_value_1 = tk.Label(self.tab1, text=str(self.trend_status[0]))
        self.one_month_trend_value_1.grid(row=8, column = 9, pady=5)

        self.three_month_trend_text_1 = tk.Label(self.tab1, text=str("Three Month Price Trend"))
        self.three_month_trend_text_1.grid(row=9, column = 8, pady=5)
        self.three_month_trend_value_1 = tk.Label(self.tab1, text=str(self.trend_status[1]))
        self.three_month_trend_value_1.grid(row=9, column = 9, pady=5)

        self.one_month_obv_trend_text_1 = tk.Label(self.tab1, text=str("One Month OBV Trend"))
        self.one_month_obv_trend_text_1.grid(row=10, column = 8, pady=5)
        self.one_month_obv_trend_value_1 = tk.Label(self.tab1, text=str(self.trend_status[2]))
        self.one_month_obv_trend_value_1.grid(row=10, column = 9, pady=5)

        self.three_month_obv_trend_text_1 = tk.Label(self.tab1, text=str("Three Month OBV Trend"))
        self.three_month_obv_trend_text_1.grid(row=11, column = 8, pady=5)
        self.three_month_obv_trend_value_1 = tk.Label(self.tab1, text=str(self.trend_status[3]))
        self.three_month_obv_trend_value_1.grid(row=11, column = 9, pady=5)

        self.stock_gain_listbox_tab1 = tk.Listbox(self.tab1)
        self.stock_gain_listbox_tab1.insert(tk.END, "stock_gain")
        self.stock_gain_listbox_tab1.grid(row=13, column=8, rowspan = 10, pady=5)


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

        self.csv_menu = tk.OptionMenu(self.tab2, self.csv_var, "total_stock_list", "industry_stock_list", "XLE-energy", "XLB-materials", "XLI-industrials", "XLP-consumer discretionary", "XLY-consumer staples", "XLV-health care", "XLF-financials", "XLK-technology", "XLC-telecommunication", "XLU-utilities", "XLRE-real estate")
        self.csv_menu.grid(row=1, column=3, pady=15)


        self.label_img_2 = tk.Label(self.tab2, image=self.img_1)
        self.label_img_2.image = self.img_1
        self.label_img_2.grid(row=2, column=0, rowspan=19, columnspan = 6, pady=5)

        self.pull_list_status_label = tk.Label(self.tab2, text="Status: " + str(self.pull_list_stock), wraplength=150)
        self.pull_list_status_label.grid(row=1, column = 5, pady=15)

        self.buy_listbox = tk.Listbox(self.tab2, height=20)
        self.buy_listbox.grid(row=3, column=8, rowspan=3, pady=15)

        self.sell_listbox = tk.Listbox(self.tab2, height=20)
        self.sell_listbox.grid(row=3, column=9, rowspan=3, pady=15)

        self.yes_pull_list_button = tk.Button(self.tab2, text="Yes", command=self.update_yes_pull_list)
        self.yes_pull_list_button.grid(row=3, column=10, pady=15)
        self.maybe_pull_list_button = tk.Button(self.tab2, text="Maybe", command=self.update_maybe_pull_list)
        self.maybe_pull_list_button.grid(row=4, column=10, pady=15)
        self.no_pull_list_button = tk.Button(self.tab2, text="No", command=self.update_no_pull_list)
        self.no_pull_list_button.grid(row=5, column=10, pady=15)

        self.yes_listbox = tk.Listbox(self.tab2)
        self.yes_listbox.grid(row=7, column=8, pady=15)
        self.maybe_listbox = tk.Listbox(self.tab2)
        self.maybe_listbox.grid(row=7, column=9, pady=15)
        self.no_listbox = tk.Listbox(self.tab2)
        self.no_listbox.grid(row=7, column=10, pady=15)

        self.update_decisions_button = tk.Button(self.tab2, text="Save Decisions", command=self.save_stock_decisions)
        self.update_decisions_button.grid(row=6, column=9, pady=15)

        self.stock_gain_listbox_tab2 = tk.Listbox(self.tab2)
        self.stock_gain_listbox_tab2.insert(tk.END, "stock_gain")
        self.stock_gain_listbox_tab2.grid(row=8, column=8, rowspan = 10, pady=15)

        self.test_var = ""

        self.one_month_trend_text_2 = tk.Label(self.tab2, text=str("One Month Price Trend"))
        self.one_month_trend_text_2.grid(row=8, column = 9, pady=15)
        self.one_month_trend_value_2 = tk.Label(self.tab2, text=str("N/A"))
        self.one_month_trend_value_2.grid(row=8, column = 10, pady=15)

        self.three_month_trend_text_2 = tk.Label(self.tab2, text=str("Three Month Price Trend"))
        self.three_month_trend_text_2.grid(row=9, column = 9, pady=15)
        self.three_month_trend_value_2 = tk.Label(self.tab2, text=str("N/A"))
        self.three_month_trend_value_2.grid(row=9, column = 10, pady=15)

        self.one_month_obv_trend_text_2 = tk.Label(self.tab2, text=str("One Month OBV Trend"))
        self.one_month_obv_trend_text_2.grid(row=10, column = 9, pady=15)
        self.one_month_obv_trend_value_2 = tk.Label(self.tab2, text=str("N/A"))
        self.one_month_obv_trend_value_2.grid(row=10, column = 10, pady=15)

        self.three_month_obv_trend_text_2 = tk.Label(self.tab2, text=str("Three Month OBV Trend"))
        self.three_month_obv_trend_text_2.grid(row=11, column = 9, pady=15)
        self.three_month_obv_trend_value_2 = tk.Label(self.tab2, text=str("N/A"))
        self.three_month_obv_trend_value_2.grid(row=11, column = 10, pady=15)


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

        if selected == True and (self.pull_list_stock != old_stock):
            self.plot_fig_tab2()
            self.plot_stock_gain_tab2(self.pull_list_stock_gain_dict[self.pull_list_stock])
            self.one_month_trend_value_2.config(text=str(self.pull_list_trend_dict[self.pull_list_stock][0]))
            self.three_month_trend_value_2.config(text=str(self.pull_list_trend_dict[self.pull_list_stock][1]))
            self.one_month_obv_trend_value_2.config(text=str(self.pull_list_trend_dict[self.pull_list_stock][2]))
            self.three_month_obv_trend_value_2.config(text=str(self.pull_list_trend_dict[self.pull_list_stock][3]))


        self.one_month_trend_value_1.config(text=str(self.trend_status[0]))
        self.three_month_trend_value_1.config(text=str(self.trend_status[1]))
        self.one_month_obv_trend_value_1.config(text=str(self.trend_status[2]))
        self.three_month_obv_trend_value_1.config(text=str(self.trend_status[3]))


    def update_pull_list_status(self):
        self.pull_list_status_label.config(text="Status: " + str(self.pull_list_stock))


    def plot_fig_tab1(self):

        # self.particle_canvas_tab1 = FigureCanvasTkAgg(fig, self.tab1)
        # # fig.tight_layout()
        # self.particle_canvas_tab1.get_tk_widget().grid(row=12, column=0, rowspan=15, columnspan = 7, pady=5)
        photo = Image.open(os.path.join(self.pics_filepath, str(self.stock) + '.png'))
        self.img_1 = ImageTk.PhotoImage(photo)
        self.label_img.configure(image=self.img_1)


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
