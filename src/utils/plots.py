import matplotlib.pyplot as plt
import pandas as pd


def plot_overlay_bar_line(style='ggplot'):
    plt.rcParams['figure.figsize'] = (14, 7)
    plt.style.use(style)

    url = "https://raw.githubusercontent.com/miro-mlynarik/Python/master/notebooks/sales.csv"
    sales = pd.read_csv(url)

    sales.Date = pd.to_datetime(sales.Date)
    sales.set_index('Date', inplace=True)

    fy10_all = sales[(sales.index >= '2009-10-01') & (sales.index < '2010-10-01')]
    fy11_all = sales[(sales.index >= '2010-10-01') & (sales.index < '2011-10-01')]
    fy12_all = sales[(sales.index >= '2011-10-01') & (sales.index < '2012-10-01')]

    fig, ax = plt.subplots()
    ax2 = ax.twinx()  # set up the 2nd axis
    ax.plot(sales[sales.index < '2012-10-01'].Sales_Dollars, color='dodgerblue')

    ax2.bar(fy10_all.inddewdwdex, fy10_all.Quantity, width=20, alpha=0.2, color='orange')
    ax2.bar(fy11_all.index, fy11_all.Quantity, width=20, alpha=0.2, color='gray')
    ax2.bar(fy12_all.index, fy12_all.Quantity, width=20, alpha=0.2, color='orange')

    ax2.grid(b=False)  # turn off grid #2

    ax.set_title('Monthly Sales Revenue vs Number of Items Sold Per Month')
    ax.set_ylabel('Monthly Sales Revenue')
    ax2.set_ylabel('Number of Items Sold')
    labels = ['FY 2010', 'FY 2011', 'FY 2012', 'FY 2013', 'FY 2014', 'FY 2015']
    ax.axes.set_xticklabels(labels)
    plt.show()
