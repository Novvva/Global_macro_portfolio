# Import modules
from tkinter import *
from tkinter.ttk import Combobox
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import *

from portfolio_optimizer import portfolio_optimizer

# Define a series of functions for buttons
def selectPath1():
    filename1 = filedialog.askopenfilename()
    path1.set(filename1)
def selectPath2():
    filename2 = filedialog.askopenfilename()
    path2.set(filename2)
def selectPath3():
    filename3 = filedialog.askopenfilename()
    path3.set(filename3)

def load_data(path_1, path_2, path_3):
    global rets, riskfree, benchmark_ret, benchmark_excess, benchmark_pnl, rf_bm
    try:
        rets = pd.read_csv(path_1)[1:]
        riskfree = pd.read_csv(path_2)
        benchmark = pd.read_csv(path_3)

        rets.index = pd.to_datetime(rets['Date'])
        del rets['Date']

        # Delete any outliers
        rets[rets > 1] = 0
        rets[rets < -1] = 0

        riskfree['Date'] = pd.to_datetime(riskfree['Date'])
        riskfree = riskfree[riskfree['Date'].isin(rets.index.values)].set_index('Date')

        # daily risk free
        riskfree = (1 + riskfree/100).pow(1 / 252) - 1

        benchmark['Date'] = pd.to_datetime(benchmark['Date'])

        # calculate returns
        benchmark['return'] = benchmark['Adj_Close'].pct_change()

        # match the date
        benchmark = benchmark[benchmark['Date'].isin(rets.index.values)].set_index('Date')

        # calculate excess returns
        rf_bm = riskfree[riskfree.index.isin(benchmark.index.values)]
        benchmark_ret = benchmark['return']
        benchmark_excess = benchmark['return'].subtract(rf_bm['RFR'], axis=0)

        # Benchmark PnL
        k = benchmark_ret
        benchmark_pnl = pd.DataFrame()
        benchmark_pnl['return'] = k
        benchmark_pnl['capital'] = 100000
        dates_to_split = pd.date_range(rets.index[0], rets.index[-1], freq='6M')
        for i in range(len(dates_to_split) - 1):
            benchmark_pnl.loc[dates_to_split[i]:dates_to_split[i + 1], 'capital'] = 100000 + (i + 1) * 10000
        benchmark_pnl.loc[dates_to_split[len(dates_to_split) - 1]:, 'capital'] = 100000 + len(dates_to_split) * 10000
        benchmark_pnl['PnL'] = benchmark_pnl['return'] * benchmark_pnl['capital']

        messagebox.showinfo(title='Success', message='Data Imported')
    except:
        messagebox.showinfo(title='Warning', message='Data Import Failed')

def back_test(initial_capital, freq, cutoff, optimization, start_date):
    global dollar_full_portfolio_bt, PnL_bt, usdrisk_bt, cadrisk_bt, overallrisk_bt, max_drawdown_bt, SR_bt, w_bt, optimization_type
    try:
        metrics.delete(1.0, END)
        optimization_type = optimization

        start_date = datetime.strptime(str(start_date), '%m/%d/%Y')
        start_date = start_date + relativedelta(months=-1)
        start_date = start_date.strftime('%m/%d/%Y')

        # Risk mapping Parameter
        VaRcutoff = {'VaR95': 5000, 'VaR99': 5800, 'CVaR95': 5700, 'CVaR99': 5400}

        # Back-testing
        rets_bt = rets[rets.index <= start_date]
        dates_to_split = pd.date_range(rets_bt.index[0], rets_bt.index[-1], freq=freq)
        # Split on these dates
        semiannual_bt = {}
        for i in range(len(dates_to_split) - 1):
            semiannual_bt[i] = rets_bt[dates_to_split[i]:dates_to_split[i + 1]]
        top_bt = {}
        for i in range(1, len(semiannual_bt)):
            corrs = ((1 + semiannual_bt[i - 1]).cumprod() - 1).iloc[-1]
            top_bt[i] = [x for x in corrs.sort_values(ascending=False, axis=0).index if "USA_" in x][:int(cutoff)]
            top_bt[i] += [x for x in corrs.sort_values(ascending=False, axis=0).index if "CAN_" in x][:int(cutoff)]

        opt_bt = portfolio_optimizer(semiannual_bt)
        if optimization == 'MVO':
            dollar_full_portfolio_bt, PnL_bt, usdrisk_bt, cadrisk_bt, overallrisk_bt, max_drawdown_bt, SR_bt, w_bt = opt_bt.portfolio_simulator(int(initial_capital), riskfree, top_bt, int(cutoff), VaRcutoff, optimization, benchmark_excess)
        else:
            dollar_full_portfolio_bt, PnL_bt, usdrisk_bt, cadrisk_bt, overallrisk_bt, max_drawdown_bt, SR_bt, w_bt = opt_bt.portfolio_simulator(int(initial_capital), riskfree, top_bt, int(cutoff), VaRcutoff, optimization)

        messagebox.showinfo(title='Success', message='Portfolio back-tested!')

    except:
        messagebox.showinfo(title='Warning', message='Portfolio back-test Failed')

def portfolio_perform(initial_capital, freq, cutoff, optimization, start_date):
    global dollar_full_portfolio_p, PnL_p, usdrisk_p, cadrisk_p, overallrisk_p, max_drawdown_p, SR_p, w_p, optimization_type
    try:
        metrics_1.delete(1.0, END)
        optimization_type = optimization

        start_date = datetime.strptime(str(start_date), '%m/%d/%Y')
        start_date = start_date + relativedelta(months=-1)
        start_date = start_date.strftime('%m/%d/%Y')

        # Risk mapping Parameter
        VaRcutoff = {'VaR95': 5000, 'VaR99': 5800, 'CVaR95': 5700, 'CVaR99': 5400}

        # Back-testing
        rets_p = rets[rets.index >= start_date]
        dates_to_split = pd.date_range(rets_p.index[0], rets_p.index[-1], freq=freq)
        # Split on these dates
        semiannual_p = {}
        for i in range(len(dates_to_split) - 1):
            semiannual_p[i] = rets_p[dates_to_split[i]:dates_to_split[i + 1]]
        top_p = {}
        for i in range(1, len(semiannual_p)):
            corrs = ((1 + semiannual_p[i - 1]).cumprod() - 1).iloc[-1]
            top_p[i] = [x for x in corrs.sort_values(ascending=False, axis=0).index if "USA_" in x][:int(cutoff)]
            top_p[i] += [x for x in corrs.sort_values(ascending=False, axis=0).index if "CAN_" in x][:int(cutoff)]

        opt_p = portfolio_optimizer(semiannual_p)
        if optimization == 'MVO':
            dollar_full_portfolio_p, PnL_p, usdrisk_p, cadrisk_p, overallrisk_p, max_drawdown_p, SR_p, w_p = opt_p.portfolio_simulator(
                int(initial_capital), riskfree, top_p, int(cutoff), VaRcutoff, optimization, benchmark_excess)
        else:
            dollar_full_portfolio_p, PnL_p, usdrisk_p, cadrisk_p, overallrisk_p, max_drawdown_p, SR_p, w_p = opt_p.portfolio_simulator(
                int(initial_capital), riskfree, top_p, int(cutoff), VaRcutoff, optimization)

        messagebox.showinfo(title='Success', message='Portfolio optimized!')

    except:
        messagebox.showinfo(title='Warning', message='Portfolio optimization Failed')

def get_metrics_bt():
    m = 'Your optimal portfolio has Sharpe Ratio of ' + str(SR_bt) + '\n Max drawdown of ' + str(max_drawdown_bt)
    metrics.insert(END, m)

def plot_PnL_bt(start_date):
    # plot PnL chart:

    PnL_bt.cumsum().plot(figsize=(15,8), label=optimization_type)
    benchmark_pnl_1 = benchmark_pnl[benchmark_pnl.index < str(start_date)]
    benchmark_pnl_1['PnL'].cumsum().plot(figsize=(15,8), label='Benchmark - MSCI ACWI')
    plt.xlabel('Date')
    plt.ylabel('PnL')
    plt.legend()
    plt.title('Back testing PnL using ' + optimization_type + ' v.s. Benchmark')

    filename = 'Back_test_PnL_plot_' + optimization_type + '.png'
    plt.savefig(filename)
    plt.close()

    messagebox.showinfo(title='Success', message='PnL saved!')

def plot_pie_bt():
    # Plot pie chart:
    threshold = 0.01
    w_bt_2 = w_bt[w_bt['final_weights'] > threshold]
    w_bt_2.plot.pie(y='final_weights', figsize=(10, 10), labeldistance=1.2, autopct='%1.0f%%', fontsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(1.3, 0.5))
    plt.title('Back testing portfolio final weights using ' + optimization_type)
    filename_1 = 'Back_test_pie_plot_' + optimization_type + '.png'
    plt.savefig(filename_1)
    plt.close()

    messagebox.showinfo(title='Success', message='Pie chart saved!')

def get_metrics_p():
    m = 'Your optimal portfolio has Sharpe Ratio of ' + str(SR_p) + '\n Max drawdown of ' + str(max_drawdown_p)
    metrics_1.insert(END, m)


def plot_PnL_p(start_date):
    # plot PnL chart:

    PnL_p.cumsum().plot(figsize=(15, 8), label=optimization_type)
    benchmark_pnl_1 = benchmark_pnl[benchmark_pnl.index >= str(start_date)]
    benchmark_pnl_1['PnL'].cumsum().plot(figsize=(15, 8), label='Benchmark - MSCI ACWI')
    plt.xlabel('Date')
    plt.ylabel('PnL')
    plt.legend()
    plt.title('Optimal portfolio PnL using ' + optimization_type + ' v.s. Benchmark')

    filename = 'Optimal_PnL_plot_' + optimization_type + '.png'
    plt.savefig(filename)
    plt.close()

    messagebox.showinfo(title='Success', message='PnL saved!')

def plot_pie_p():
    # Plot pie chart:
    threshold = 0.01
    w_bt_2 = w_p[w_p['final_weights'] > threshold]
    w_bt_2.plot.pie(y='final_weights', figsize=(10, 10), labeldistance=1.2, autopct='%1.0f%%', fontsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(1.3, 0.5))
    plt.title('Optimal portfolio final weights using ' + optimization_type)
    filename_1 = 'Optimal_pie_plot_' + optimization_type + '.png'
    plt.savefig(filename_1)
    plt.close()

    messagebox.showinfo(title='Success', message='Pie chart saved!')


# GUI design
window = Tk()
image = Image.open('robo-adv.png')
image = ImageTk.PhotoImage(image)

window.title('Robo Advisor Platform')
window.geometry('1024x440')
background_label = Label(window, image=image)
background_label.place(x=0, y=33, relwidth=1, relheight=1)
l = Label(window, text='Welcome to Global Macro Portfolio Robo Advisor', bg='light blue', font=('calibre', 15, 'bold'), width=1024, height=2)
l.pack()

Label(window, text='Please load all the data:', width=30,
      font=('calibre', 12, 'bold')).place(bordermode=OUTSIDE, x=5, y=50)

path1 = StringVar()
path2 = StringVar()
path3 = StringVar()

entryPath1 = Entry(window, width=20, textvariable=path1)
entryPath1.place(bordermode=OUTSIDE, x=5, y=80)

button1 = Button(window, width=10, text="Select return", font=('calibre', 12, 'bold'), command=selectPath1)
button1.place(bordermode=OUTSIDE, x=5, y=110)

entryPath2 = Entry(window, width=25, textvariable=path2)
entryPath2.place(bordermode=OUTSIDE, x=200, y=80)

button3 = Button(window, width=15, text="Select risk free rate", font=('calibre', 12, 'bold'), command=selectPath2)
button3.place(bordermode=OUTSIDE, x=200, y=110)

entryPath3 = Entry(window, width=25, textvariable=path3)
entryPath3.place(bordermode=OUTSIDE, x=440, y=80)

button5 = Button(window, width=15, text="Select benchmark", font=('calibre', 12, 'bold'), command=selectPath3)
button5.place(bordermode=OUTSIDE, x=440, y=110)

button6 = Button(window, width=10, text="Load data", font=('calibre', 12, 'bold'),
                 command=lambda: load_data(entryPath1.get(), entryPath2.get(), entryPath3.get()))
button6.place(bordermode=OUTSIDE, x=700, y=85)

Label(window, text='Please enter your initial capital:', width=25,
      font=('calibre', 12, 'bold')).place(bordermode=OUTSIDE, x=5, y=160)
entry1 = Entry(window, width=10, textvariable=StringVar())
entry1.place(bordermode=OUTSIDE, x=5, y=190)

Label(window, text='Please choose your cutoff:', width=20,
      font=('calibre', 12, 'bold')).place(bordermode=OUTSIDE, x=300, y=160)
combobox1 = Combobox(window, values=np.arange(5, 31).tolist(), width=10)
combobox1.current(0)
combobox1.place(bordermode=OUTSIDE, x=300, y=190)

Label(window, text='Please choose your preferred optimization method:', width=35,
      font=('calibre', 12, 'bold')).place(bordermode=OUTSIDE, x=500, y=160)
combobox2 = Combobox(window, values=['Equally weighted', 'MVO', 'Risk parity', 'Sharpe ratio maximization'])
combobox2.current(0)
combobox2.place(bordermode=OUTSIDE, x=500, y=190)

Label(window, text='Rebalancing period', width=20,
      font=('calibre', 12, 'bold')).place(bordermode=OUTSIDE, x=830, y=160)
combobox3 = Combobox(window, values=['M', '3M', '6M', '9M', '12M'], width=10)
combobox3.current(0)
combobox3.place(bordermode=OUTSIDE, x=830, y=190)

Label(window, text='Start date (mm/dd/yy)', width=20,
      font=('calibre', 12, 'bold')).place(bordermode=OUTSIDE, x=5, y=230)
entry2 = Entry(window, width=10, textvariable=StringVar())
entry2.place(bordermode=OUTSIDE, x=5, y=260)

Label(window, text='Please select your risk appetite', width=25,
      font=('calibre', 12, 'bold')).place(bordermode=OUTSIDE, x=300, y=230)
combobox5 = Combobox(window, values=['Risk averse', 'Moderate', 'Risk Seeking'], width=10)
combobox5.current(0)
combobox5.place(bordermode=OUTSIDE, x=300, y=260)

button8 = Button(window, width=25, text="Hit me to back-test your portfolio!",
                 font=('calibre', 12, 'bold'), command=lambda: back_test(entry1.get(), combobox3.get(), combobox1.get(), combobox2.get(), entry2.get()))
button8.place(bordermode=OUTSIDE, x=600, y=230)

button9 = Button(window, width=25, text="Hit me to optimize your portfolio!",
                 font=('calibre', 12, 'bold'), command=lambda: portfolio_perform(entry1.get(), combobox3.get(), combobox1.get(), combobox2.get(), entry2.get()))
button9.place(bordermode=OUTSIDE, x=600, y=260)

metrics = Text(window, height=10, width=25)
metrics.place(bordermode=OUTSIDE, x=240, y=300)

Display = Button(window, height=2, width=25, text="Show back-test metrics", font=('calibre', 12, 'bold'), command=get_metrics_bt)
Display.place(bordermode=OUTSIDE, x=5, y=300)

Plot_button = Button(window, height=2, width=25, text="Plot back-testing PnL v.s. Benchmark", font=('calibre', 12, 'bold'), command=lambda: plot_PnL_bt(entry2.get()))
Plot_button.place(bordermode=OUTSIDE, x=5, y=350)

Plot_button_1 = Button(window, height=2, width=25, text="Plot Pie chart", font=('calibre', 12, 'bold'), command=plot_pie_bt)
Plot_button_1.place(bordermode=OUTSIDE, x=5, y=400)

metrics_1 = Text(window, height=10, width=25)
metrics_1.place(bordermode=OUTSIDE, x=735, y=300)

Display_1 = Button(window, height=2, width=30, text="Show your optimized portfolio metrics", font=('calibre', 12, 'bold'), command=get_metrics_p)
Display_1.place(bordermode=OUTSIDE, x=450, y=300)

Plot_button_2 = Button(window, height=2, width=30, text="Plot your portfolio PnL v.s. Benchmark", font=('calibre', 12, 'bold'), command=lambda: plot_PnL_p(entry2.get()))
Plot_button_2.place(bordermode=OUTSIDE, x=450, y=350)

Plot_button_3 = Button(window, height=2, width=25, text="Plot Pie chart", font=('calibre', 12, 'bold'), command=plot_pie_p)
Plot_button_3.place(bordermode=OUTSIDE, x=450, y=400)

window.mainloop()

