from tkinter import *
from tkinter.ttk import Combobox
from tkinter import messagebox, filedialog

import numpy as np
import pandas as pd
from portfolio_optimizer import portfolio_optimizer
from PIL import Image, ImageTk

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    global rets, riskfree, benchmark_ret, benchmark_excess
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
        riskfree = (1 + riskfree).pow(1 / 252) - 1

        benchmark['Date'] = pd.to_datetime(benchmark['Date'])

        # calculate returns
        benchmark['return'] = benchmark['Adj_Close'].pct_change()

        # match the date
        benchmark = benchmark[benchmark['Date'].isin(rets.index.values)].set_index('Date')

        # calculate excess returns
        rf_bm = riskfree[riskfree.index.isin(benchmark.index.values)]
        benchmark_ret = benchmark['return']
        benchmark_excess = benchmark['return'].subtract(rf_bm['RFR'], axis=0)

        messagebox.showinfo(title='Success', message='Data Imported')
    except:
        messagebox.showinfo(title='Warning', message='Data Import Failed')

def load_capital(capital):
    global initial_capital
    try:
        initial_capital = capital
        messagebox.showinfo(title='Success', message='Capital Imported')
    except:
        messagebox.showinfo(title='Warning', message='Capital Import Failed')

def portfolio_calculator(freq, cutoff, optimization):
    global dollar_full_portfolio, PnL, usdrisk, cadrisk, overallrisk, max_drawdown, optimization_type
    try:
        optimization_type = optimization

        dates_to_split = pd.date_range(rets.index[0], rets.index[-1], freq=freq)
        # Split on these dates
        semiannual = {}
        for i in range(len(dates_to_split) - 1):
            semiannual[i] = rets[dates_to_split[i]:dates_to_split[i + 1]]
        top = {}
        for i in range(1, len(semiannual)):
            corrs = ((1 + semiannual[i - 1]).cumprod() - 1).iloc[-1]
            top[i] = [x for x in corrs.sort_values(ascending=False, axis=0).index if "USA_" in x][:int(cutoff)]
            top[i] += [x for x in corrs.sort_values(ascending=False, axis=0).index if "CAN_" in x][:int(cutoff)]

        # Risk mapping Parameter
        VaRcutoff = {'VaR95': 5000, 'VaR99': 5800, 'CVaR95': 5700, 'CVaR99': 5400}

        opt = portfolio_optimizer(semiannual)
        if optimization == 'MVO':
            dollar_full_portfolio, PnL, usdrisk, cadrisk, overallrisk, max_drawdown = opt.portfolio_simulator(
                int(initial_capital), riskfree, top, int(cutoff), VaRcutoff, optimization, benchmark_excess)
        else:
            dollar_full_portfolio, PnL, usdrisk, cadrisk, overallrisk, max_drawdown = opt.portfolio_simulator(
                int(initial_capital), riskfree, top, int(cutoff), VaRcutoff, optimization)

        messagebox.showinfo(title='Success', message='Portfolio optimized!')

    except:
        messagebox.showinfo(title='Warning', message='Portfolio optimization Failed')

def get_metrics():
    metrics.insert(END, 'Your optimal portfolio has Max drawdown of ' + str(max_drawdown))

def plot_PnL():
    PnL.cumsum().plot(figsize=(15,8), label=optimization_type)
    ((benchmark_ret * 100000).cumsum().plot(figsize=(15, 8), label='Benchmark - MSCI ACWI'))
    plt.xlabel('Date')
    plt.ylabel('PnL')
    plt.title('Optimal Portfolio PnL using ' + optimization_type + ' v.s. Benchmark')

    filename = 'PnL_plot_' + optimization_type + '.png'
    plt.savefig(filename)

    messagebox.showinfo(title='Success', message='Plot saved!')

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

Label(window, text='Please choose your initial capital:', width=30,
      font=('calibre', 12, 'bold')).place(bordermode=OUTSIDE, x=5, y=160)
entry1 = Entry(window, width=10, textvariable=StringVar())
entry1.place(bordermode=OUTSIDE, x=5, y=190)
button7 = Button(window, width=10, text="Load capital", font=('calibre', 12, 'bold'), command=lambda: load_capital(entry1.get()))
button7.place(bordermode=OUTSIDE, x=110, y=192)

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

button8 = Button(window, width=50, text="Hit me to have your optimal investment portfolio suggestion",
                 font=('calibre', 12, 'bold'), command=lambda: portfolio_calculator(combobox3.get(), combobox1.get(), combobox2.get()))
button8.place(bordermode=OUTSIDE, x=300, y=250)

metrics = Text(window, height=6, width=50)
metrics.place(bordermode=OUTSIDE, x=200, y=300)

Display = Button(window, height=2, width=20, text="Show portfolio metrics", font=('calibre', 12, 'bold'), command=get_metrics)
Display.place(bordermode=OUTSIDE, x=5, y=300)

Plot_button = Button(window, height=2, width=20, text="Save PnL plot", font=('calibre', 12, 'bold'), command=plot_PnL)
Plot_button.place(bordermode=OUTSIDE, x=5, y=350)

window.mainloop()

