# tsa.py - time series analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from scipy.special import inv_boxcox
from IPython.display import display
from typing import Optional


def get_major_locator(date_freq, xticks_freq):
    if date_freq == 'M':
        major_locator = mdates.MonthLocator(interval=xticks_freq)
    elif date_freq == 'Y':
        major_locator = mdates.YearLocator(base=xticks_freq)
    elif date_freq == 'W':
        major_locator = mdates.WeekdayLocator(interval=xticks_freq)
    elif date_freq == 'D':
        major_locator = mdates.DayLocator(interval=xticks_freq)
    return major_locator


def plot_many_ts(
    data: pd.DataFrame,
    y: str,
    id: str,
    title: str,
    nrows: int,
    ncols: int = 4,
    figsize_coef: int = 6,
    xticks_fmt: str = '%Y-%m',
    date_freq: str = 'M',
    xticks_freq: int = 2,
    rotation: int = 45
) -> None:
    """Plot multiple time series in 4-plots-per-row figure

    Parameters
    ----------
    data : pd.DateFrame
        Dataframe in long format containing time series with index
        containing the date/time information used for output figure.
    y : str
        Column name of the actual time series data
        The actual column data type must be float.
    id : str
        Column name of identifier of the diplayed time series.
        The actual column data type must be integer.
    title : str
        Column name of textual description of the time series.
    nrows : int
        Number of rows to display in output figure, automatically adjusted
        for the number of plots available in data.
    ncols: int, optional
        Number of columns to display in output figure.
    figsize_coef : int, optional
        Resizing coefficient of output figure.
    """
    data = data.sort_values(by=id)
    series_ids = list(data[id].unique())

    n_plots = min(nrows * ncols, len(series_ids))
    nrows_actual = n_plots // ncols + 1 if n_plots % ncols > 0 else int(n_plots/ncols)
    figsize_width = figsize_coef * ncols
    figsize_height = (figsize_coef - 1) * nrows_actual
    figsize = (figsize_width, figsize_height)

    _, axs = plt.subplots(nrows=nrows_actual, ncols=ncols, figsize=figsize)
    for i in range(n_plots):
        ax = axs[i // ncols, i % ncols] if nrows > 1 else axs[i]
        series_data = data[data[id] == series_ids[i]]
        series_data[y].plot(ax=ax)
        ax.xaxis.set_major_locator(
            get_major_locator(date_freq=date_freq, xticks_freq=xticks_freq)
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter(xticks_fmt))
        ax.tick_params(axis="x", rotation=rotation)
        ax.set_title(f"{int(series_data[id][0])}: {series_data[title][0][:50]}")
    plt.tight_layout()


def plot_one_ts(
    y: pd.Series,
    title: str,
    figsize: tuple = (14, 10),
    rotation: int = 45,
    hist_bins: int = 10,
    xticks_fmt: str = '%Y-%m',
    date_freq: str = 'M',
    xticks_freq: int = 3
) -> None:
    """Plot lineplot and histogram for given time series for exploratory purposes.

    Parameters
    ----------
    y : pd.Series
        Time series to be displayed
    title : str
        Title of the plot
    figsize : tuple
        Optional argument for larger displays
    xticks_freq : int, optional
        Frequency of xaxis grid points
    hist_bins : int, optional
        Number of bins in histrogram plot
    rotation: int, optional
        X-ticks labels rotation
    """

    _, axs = plt.subplots(2, 1, figsize=figsize)
    y.plot(ax=axs[0])
    axs[0].tick_params(axis="x", rotation=rotation)
    axs[0].xaxis.set_major_locator(
        get_major_locator(date_freq=date_freq, xticks_freq=xticks_freq)
    )
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter(xticks_fmt))
    axs[0].set_xlim(y.index.min(), y.index.max())
    axs[0].grid(True, alpha=0.2)
    axs[0].set_title(title)

    axs[1].hist(y, bins=hist_bins)
    axs[1].set_title(title)
    plt.subplots_adjust(hspace=0.3)


def decompose_ts(
    y: pd.Series,
    period: int = 12,
    method: str = 'stl',
    type_: str = 'additive',
    stl_trend: int = None,
    stl_seasonal: int = 7,
    stl_seasonal_deg: int = 1,
    stl_trend_deg: int = 1,
    stl_low_pass_deg: int = 1,
    title: str = None,
    xticks_freq: int = 3,
    xticks_fmt: str = '%Y-%m',
    date_freq: str = 'M',
    rotation: int = 45,
    figsize: tuple = (12, 18),
    return_type: str = 'both'
):
    """Decompose time series and plot respective trend, seasonal and remainder components.

    Parameters
    ----------
    y :
        Time series variable to be decomposed
    period : int
        Number of observations in seasonal period (for monthly data = 12)
    method : str
        Either 'stl' or 'classical'
    type_ : str
        Either 'additive' or 'multiplicative'. Only applies to classic method
    figsize : tuple
        Optional argument for larger displays
    xticks_frequency : str
        Optional argument for frequency of xaxis grid
    date_format : str
        Format of displayed dates on x axis
    rotation : int
        Optional argument for xticks labels rotation
    return_type : str
        'plot', 'data' or 'both'
    """
    if method == 'stl':
        decomposed = tsa.STL(
            endog=y, period=period,
            trend=stl_trend,
            seasonal=stl_seasonal,
            seasonal_deg=stl_seasonal_deg,
            trend_deg=stl_trend_deg, low_pass_deg=stl_low_pass_deg
        ).fit()
        sa = decomposed.observed - decomposed.seasonal

    elif method == 'classical':
        decomposed = tsa.seasonal_decompose(x=y, model=type_, period=period)
        is_additive = (type_ == 'additive')
        if is_additive:
            sa = decomposed.observed - decomposed.seasonal
        else:
            sa = decomposed.observed / decomposed.seasonal

    components = pd.DataFrame({'observed': decomposed.observed,
                               'trend': decomposed.trend,
                               'seasonal': decomposed.seasonal,
                               'residual': decomposed.resid,
                               'seasonally adjusted': sa
                               }, index=y.index)

    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=figsize, sharex=True)
    for i, column in enumerate(components.columns):
        axs[i].plot(components[column])
        axs[i].grid(True, alpha=1, color='#FFFFFF', linewidth=1.7)
        axs[i].set_ylabel(column)
        axs[i].set_xticks([])
        axs[i].patch.set_facecolor('#F5F5F5')

    axs[0].set_title('Time series decomposition using ' + method.upper() + ' method: ' + title)
    axs[4].tick_params(axis="x", rotation=rotation)
    axs[4].xaxis.set_major_locator(get_major_locator(
        date_freq=date_freq, xticks_freq=xticks_freq))
    axs[4].xaxis.set_major_formatter(mdates.DateFormatter(xticks_fmt))
    axs[4].set_xlim(components.index.min(), components.index.max())
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()

    if return_type == 'data':
        plt.close(fig)
        return components
    elif return_type == 'plot':
        return None
    elif return_type == 'both':
        return components


def plot_forecast(
    data: pd.DataFrame,
    title: str,
    columns: list = ["observed", "predicted_mean", "lower", "upper"],
    figsize: tuple = (16, 8),
    rotation: int = 45,
    date_fmt: str = '%Y-%m',
    date_freq: str = 'M',
    xticks_freq: int = 3,
    start_idx: int = 0
) -> None:
    """Plot time series forecasts.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing columns representing observed, predicted, and confidence
        interval limits variables
    title : str
        Title of the plot
    columns : list, optional
        Plotted column names
    figsize : tuple
        Displayed figure size
    xticks_freq : str
        Frequency of xaxis grid
    hist_bins : int
        Number of bins in histrogram plot
    rotation : int
        Xticks labels rotation
    """
    _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(data[columns[1]][start_idx:], label="prediction")
    ax.plot(data[columns[0]][start_idx:], marker='o', markerfacecolor='k', markeredgecolor='k',
            markersize=4, linewidth=0, label="observed")
    ax.fill_between(
        x=data.index[start_idx:],
        y1=data[columns[2]][start_idx:],
        y2=data[columns[3]][start_idx:],
        alpha=0.2,
        label="prediction interval"
    )

    ax.tick_params(axis="x", rotation=rotation)
    ax.xaxis.set_major_locator(get_major_locator(date_freq=date_freq, xticks_freq=xticks_freq))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    # ax.set_xlim(data.index.min(), data.index.max())
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    plt.subplots_adjust(hspace=0.3)


def print_model_stats(model):
    print(f"AICc= {model.aicc:.2f}\n"
          f"AIC = {model.aic:.2f} \n"
          f"MSE = {model.mse:.2f}"
          f"{model.summary()}")


def plot_residuals_stats(model, y, alpha, nlags):
    """Model residuals diagnostic plots and (P)ACF plots for original time series."""
    model.plot_diagnostics(figsize=(14, 8), lags=nlags)
    _, ax = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(y, lags=nlags, alpha=alpha, use_vlines=True, title='ACF',
             zero=True, vlines_kwargs=None, marker=None, fft=False, ax=ax[0])
    plot_pacf(y, lags=nlags, alpha=alpha, use_vlines=True, title='PACF',
              zero=True, vlines_kwargs=None, marker=None, ax=ax[1])


def print_residuals_acf_details(model, alpha, nlags):
    residual_acf = pd.DataFrame(
        {'residual_acf': tsa.acf(model.filter_results.standardized_forecasts_error[0],
                                 adjusted=False, nlags=20, qstat=True, alpha=alpha)[0][1:],
         'q_stat': model.test_serial_correlation(method='ljungbox', lags=20)[0][0],
         'p_value': model.test_serial_correlation(method='ljungbox', lags=20)[0][1]
         }, index=np.arange(1, nlags + 1)).T
    display(residual_acf)


def stl_arimax_forecast(
    y: pd.Series,
    arima_params: dict,          # order, seasonal_order, trend
    forecast_plot_params: dict,  # date_freq, date_fmt, xticks_freq, title, start_idx
    stl_trend=None,
    stl_seasonal=7,
    start=0,
    steps=None,
    exog=None,
    dynamic=None,
    alpha=0.05,
    nlags=20,
    lmbda=None,
) -> None:
    """ARIMAX model forecasts applied to seasonally adjusted time series using STL method."""
    arima_params['freq'] = y.index.freq
    model = tsa.STLForecast(endog=y,
                            model=ARIMA,
                            trend=stl_trend,
                            seasonal=stl_seasonal,
                            model_kwargs=arima_params
                            ).fit()

    print_model_stats(model.model_result)
    print_residuals_acf_details(model.model_result, alpha, nlags)
    plot_residuals_stats(model.model_result, y, alpha, nlags)

    end = (steps or 0) + len(y)
    prediction = model.get_prediction(start=start, end=end, exog=exog, dynamic=dynamic)

    sa = pd.Series(model.result.resid + model.result.trend, name=y.name + '_seasonal_adjusted')
    predicted = pd.Series(prediction.predicted_mean, name='predicted_raw')
    regression_residual = pd.Series(model.model_result.resid, name='regression_residual')
    arima_residual = pd.Series(list(
        model.model_result.filter_results.standardized_forecasts_error[0]), index=y.index,
        name='arima_residual'
    )
    conf_int = pd.DataFrame(prediction.conf_int(alpha=alpha))
    calculated = pd.concat([sa, predicted, regression_residual, arima_residual, conf_int], axis=1)

    forecast = pd.merge(y, calculated, left_index=True, right_index=True, how='right')
    forecast.columns = list(forecast.columns[:-2]) + ['pi_lower_raw', 'pi_upper_raw']

    if lmbda is not None:
        forecast['predicted_raw'] = inv_boxcox(forecast['predicted_raw'], lmbda)
        forecast['pi_lower_raw'] = inv_boxcox(forecast['pi_lower_raw'], lmbda)
        forecast['pi_upper_raw'] = inv_boxcox(forecast['pi_upper_raw'], lmbda)

    plot_forecast(data=forecast,
                  title=forecast_plot_params['title'],
                  start_idx=forecast_plot_params['start_idx'],
                  xticks_freq=forecast_plot_params['xticks_freq'],
                  date_freq=forecast_plot_params['date_freq'],
                  columns=forecast.columns[[0, 2, 5, 6]],
                  date_fmt=forecast_plot_params['date_fmt'])

    display(forecast.head())


def ets_forecast(
    data: pd.DataFrame,
    y: str,
    title: str,
    forecast_plot_params: dict,  # date_freq, date_fmt, xticks_freq, title
    forecast_params: dict,       # start, steps, dynamic, alpha, last_obs_only
    ets_params: dict = None,     # trend, damped_trend, seasonal, seasonal_periods
    nlags: int = 20,
    lmbda: Optional[float] = None
) -> None:
    """ETS model forecasts applied to raw time series."""
    used_obs = forecast_params.get('last_obs_only', len(data))
    data = data[-used_obs:]

    y_col, y = y, data[y]
    y.index.freq = forecast_params['date_freq']
    ets_params = ets_params or {}

    model = ExponentialSmoothing(endog=y,
                                 trend=ets_params.get('trend', False),
                                 seasonal=ets_params.get('seasonal'),
                                 damped_trend=ets_params.get('damped_trend', False),
                                 ).fit()

    print_model_stats(model)
    print_residuals_acf_details(model, forecast_params.get('alpha', 0.05), nlags)

    end = forecast_params.get('steps', 0) + len(y) - 1
    prediction = model.get_prediction(start=forecast_params.get('start', 0),
                                      end=end,
                                      dynamic=forecast_params.get('dynamic'))

    predicted = pd.Series(prediction.predicted_mean, name='predicted_raw')
    regression_residual = pd.Series(model.resid, name='regression_residual')
    arima_residual = pd.Series(list(
        model.filter_results.standardized_forecasts_error[0]), index=y.index,
        name='arima_residual'
    )
    conf_int = pd.DataFrame(prediction.conf_int(alpha=forecast_params.get('alpha', 0.05)))
    calculated = pd.concat([predicted, regression_residual, arima_residual, conf_int], axis=1)

    forecast = pd.merge(y, calculated, left_index=True, right_index=True, how='right')
    forecast.columns = list(forecast.columns[:-2]) + ['pi_lower_raw', 'pi_upper_raw']

    if lmbda is not None:
        forecast['predicted_raw'] = inv_boxcox(forecast['predicted_raw'], lmbda)
        forecast['pi_lower_raw'] = inv_boxcox(forecast['pi_lower_raw'], lmbda)
        forecast['pi_upper_raw'] = inv_boxcox(forecast['pi_upper_raw'], lmbda)

    title = f"{data[title][0]}, {y_col} \nForecast: {forecast_params['steps']} days, " + \
            f"based on {used_obs} observations"

    plot_forecast(data=forecast,
                  columns=forecast.columns[[0, 1, 4, 5]],
                  title=title,
                  start_idx=forecast_params.get('start', 0),
                  xticks_freq=forecast_plot_params['xticks_freq'],
                  date_freq=forecast_plot_params['date_freq'],
                  date_fmt=forecast_plot_params['date_fmt'])

    display(forecast.tail(10))
