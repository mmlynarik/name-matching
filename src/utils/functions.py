import itertools as it
import pickle

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as m
import sklearn.tree as t
import pmdarima as pa
from pandas.api.types import is_numeric_dtype
from sklearn import linear_model as lm
from sklearn import model_selection as ms
from sklearn import preprocessing as pp
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sqlalchemy import create_engine

as_float = ["count"]
as_pct = ["pct"]
dict_pct = {i: "{:,.2%}" for i in as_pct}
dict_flt = {i: "{:,.0f}" for i in as_float}
dict_fmt = {**dict_pct, **dict_flt}


def classification_table(actual_response, predicted_prob, threshold=0.5):
    predicted_response = pp.binarize([predicted_prob], threshold)[0]
    return confusion_matrix(actual_response, predicted_response)


def classification_statistics(actual_response, predicted_prob, threshold=0.5):
    predicted_response = pp.binarize([predicted_prob], threshold)[0]
    return classification_report(actual_response, predicted_response)


def plot_classification_table(
    cm, classes, title="Classification table", cmap=plt.cm.Blues
):

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks([-0.5, 1.5], [0, 1])

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt), horizontalalignment="center", size=13,
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel("Actual response")
    plt.xlabel("Predicted response")
    plt.show()


def category_ordering(data, column, prefix=None):
    if is_numeric_dtype(data[column]) and prefix:
        data[column] = prefix + data[column].astype("str") + prefix

    ordered_categories = pd.api.types.CategoricalDtype(
        categories=data[column], ordered=True
    )
    data[column] = data[column].astype("category")
    data[column] = data[column].astype(ordered_categories)
    return data[column]


def make_categorical(data, predictors, bins=10, rounding=2):
    for predictor in predictors:
        binned_predictor_raw = pd.qcut(data[predictor], bins, duplicates="drop")
        binned_predictor_upper = pd.Series(
            binned_predictor_raw.apply(lambda x: x.right).astype("float").round(rounding),
            name=predictor + "_bin_upp")
        binned_predictor_lower = pd.Series(
            binned_predictor_raw.apply(lambda x: x.left).astype("float").round(rounding),
            name=predictor + "_bin_low")
        binned_predictor_id = pd.Series(
            pd.qcut(data[predictor], bins, duplicates="drop", labels=False), name=predictor +
            "_bin")
        binned_predictor_rng = pd.Series(
            "(" + binned_predictor_lower.astype(str) + ", " + binned_predictor_upper.astype(str) +
            ")", name=predictor + "bin_rng")

        binned_predictor = pd.concat(
            [binned_predictor_id, binned_predictor_rng, binned_predictor_lower,
             binned_predictor_upper], axis=1
        )
        data = pd.concat([data, binned_predictor], axis=1)

    return data


def train_test_statistics(
    train_sample,
    test_sample,
    total_sample,
    groupby,
    response,
    event_name="event",
    event_rate_name="er",
):
    train_test_stats = pd.concat(
        [
            pd.Series(train_sample.groupby(groupby).size(), name="train_obs"),
            pd.Series(test_sample.groupby(groupby).size(), name="test_obs"),
            pd.Series(total_sample.groupby(groupby).size(), name="total_obs"),
            pd.Series(round(
                train_sample.groupby(groupby).size() / total_sample.groupby(groupby).size() * 100,
                2), name="train_obs (%)"),
            pd.Series(train_sample.groupby(groupby)[response].sum(), name="train_guar"),
            pd.Series(test_sample.groupby(groupby)[response].sum(), name="test_guar"),
            pd.Series(total_sample.groupby(groupby)[response].sum(), name="total_guar"),
            pd.Series(round(
                train_sample.groupby(groupby)[response].sum() / total_sample.groupby(groupby)
                [response].sum() * 100, 2), name="train_guar (%)"),
            pd.Series(
                round(train_sample.groupby(groupby)[response].sum() / train_sample.groupby(
                    groupby).size() * 100, 2,), name="gr_train (%)"),
            pd.Series(
                round(
                    test_sample.groupby(groupby)[response].sum()
                    / test_sample.groupby(groupby).size()
                    * 100,
                    2,
                ),
                name="gr_test (%)",
            ),
            pd.Series(
                round(
                    total_sample.groupby(groupby)[response].sum()
                    / total_sample.groupby(groupby).size()
                    * 100,
                    2,
                ),
                name="gr_total (%)",
            ),
        ],
        axis=1,
    )

    train_test_stats.reset_index(inplace=True)
    train_test_stats_sum = pd.DataFrame(train_test_stats.sum()).transpose()
    train_test_stats_sum.loc[0, [groupby]] = "Total"
    train_test_stats_sum.loc[0, "train_obs (%)"] = round(
        train_test_stats_sum.loc[0, "train_obs"]
        / train_test_stats_sum.loc[0, "total_obs"]
        * 100,
        2,
    )
    train_test_stats_sum.loc[0, "train_guar (%)"] = round(
        train_test_stats_sum.loc[0, "train_guar"]
        / train_test_stats_sum.loc[0, "total_guar"]
        * 100,
        2,
    )
    train_test_stats_sum.loc[0, "gr_train (%)"] = round(
        train_test_stats_sum.loc[0, "train_guar"]
        / train_test_stats_sum.loc[0, "train_obs"]
        * 100,
        2,
    )
    train_test_stats_sum.loc[0, "gr_test (%)"] = round(
        train_test_stats_sum.loc[0, "test_guar"]
        / train_test_stats_sum.loc[0, "test_obs"]
        * 100,
        2,
    )
    train_test_stats_sum.loc[0, "gr_total (%)"] = round(
        train_test_stats_sum.loc[0, "total_guar"]
        / train_test_stats_sum.loc[0, "total_obs"]
        * 100,
        2,
    )

    train_test_stats[groupby] = train_test_stats[groupby].astype(str)
    train_test_stats_final = pd.concat(
        [train_test_stats, train_test_stats_sum], axis=0, ignore_index=True
    )
    train_test_stats_final.fillna(0, inplace=True)
    train_test_stats_final = train_test_stats_final.astype(
        {
            "train_obs": "int",
            "test_obs": "int",
            "total_obs": "int",
            "train_guar": "int",
            "test_guar": "int",
            "total_guar": "int",
        }
    )
    train_test_stats_final.columns = [
        groupby,
        "train_obs",
        "test_obs",
        "total_obs",
        "train_obs (%)",
        "train_" + event_name,
        "test_" + event_name,
        "total_" + event_name,
        "train_" + event_name + " (%)",
        event_rate_name + "_train (%)",
        event_rate_name + "_test (%)",
        event_rate_name + "_total (%)",
    ]
    return train_test_stats_final


def univariate_analysis_continuous(
    train_sample,
    test_sample,
    total_sample,
    predictor,
    response,
    rotation=45,
    rounding=2,
    bins=10,
    figsize=(24, 6),
    event_name="event",
    event_rate_name="er",
    plots_only=False,
    bins_details_only=False,
):
    train_sample_bin = make_categorical(
        data=train_sample, predictors=[predictor], bins=bins, rounding=rounding
    )
    test_sample_bin = make_categorical(
        data=test_sample, predictors=[predictor], bins=bins, rounding=rounding
    )
    total_sample_bin = make_categorical(
        data=total_sample, predictors=[predictor], bins=bins, rounding=rounding
    )

    predictor_binned = predictor + "_bin"
    predictor_binned_range = predictor + "_bin_rng"

    train_test_stats = pd.concat(
        [
            pd.Series(
                train_sample_bin.groupby(predictor_binned).size(), name="count_train"
            ),
            pd.Series(
                test_sample_bin.groupby(predictor_binned).size(), name="count_test"
            ),
            pd.Series(
                total_sample_bin.groupby(predictor_binned).size(), name="count_total"
            ),
            pd.Series(
                round(
                    train_sample_bin.groupby(predictor_binned).size()
                    / total_sample_bin.groupby(predictor_binned).size()
                    * 100,
                    2,
                ),
                name="count_train (%)",
            ),
            pd.Series(
                train_sample_bin.groupby(predictor_binned)[response].sum(),
                name=event_name + "_train",
            ),
            pd.Series(
                test_sample_bin.groupby(predictor_binned)[response].sum(),
                name=event_name + "_test",
            ),
            pd.Series(
                total_sample_bin.groupby(predictor_binned)[response].sum(),
                name=event_name + "_total",
            ),
            pd.Series(
                round(
                    train_sample_bin.groupby(predictor_binned)[response].sum()
                    / total_sample_bin.groupby(predictor_binned)[response].sum()
                    * 100,
                    2,
                ),
                name=event_name + "_train (%)",
            ),
            pd.Series(
                round(
                    train_sample_bin.groupby(predictor_binned)[response].sum()
                    / train_sample_bin.groupby(predictor_binned).size()
                    * 100,
                    2,
                ),
                name=event_rate_name + "_train (%)",
            ),
            pd.Series(
                round(
                    test_sample_bin.groupby(predictor_binned)[response].sum()
                    / test_sample_bin.groupby(predictor_binned).size()
                    * 100,
                    2,
                ),
                name=event_rate_name + "_test (%)",
            ),
            pd.Series(
                round(
                    total_sample_bin.groupby(predictor_binned)[response].sum()
                    / total_sample_bin.groupby(predictor_binned).size()
                    * 100,
                    2,
                ),
                name=event_rate_name + "_total (%)",
            ),
        ],
        axis=1,
    )

    train_test_stats.reset_index(inplace=True)

    train_test_stats_sum = pd.DataFrame(train_test_stats.sum()).transpose()
    train_test_stats_sum.loc[0, [predictor_binned]] = "Total"
    train_test_stats_sum.loc[0, "count_train (%)"] = round(
        train_test_stats_sum.loc[0, "count_train"]
        / train_test_stats_sum.loc[0, "count_total"]
        * 100,
        2,
    )
    train_test_stats_sum.loc[0, event_name + "_train (%)"] = round(
        train_test_stats_sum.loc[0, event_name + "_train"]
        / train_test_stats_sum.loc[0, event_name + "_total"]
        * 100,
        2,
    )
    train_test_stats_sum.loc[0, event_rate_name + "_train (%)"] = round(
        train_test_stats_sum.loc[0, event_name + "_train"]
        / train_test_stats_sum.loc[0, "count_train"]
        * 100,
        2,
    )
    train_test_stats_sum.loc[0, event_rate_name + "_test (%)"] = round(
        train_test_stats_sum.loc[0, event_name + "_test"]
        / train_test_stats_sum.loc[0, "count_test"]
        * 100,
        2,
    )
    train_test_stats_sum.loc[0, event_rate_name + "_total (%)"] = round(
        train_test_stats_sum.loc[0, event_name + "_total"]
        / train_test_stats_sum.loc[0, "count_total"]
        * 100,
        2,
    )

    train_test_stats_final = pd.concat(
        [train_test_stats, train_test_stats_sum], axis=0, sort=False, ignore_index=True
    )
    train_test_stats_final.fillna(0, inplace=True)
    train_test_stats_final = train_test_stats_final.astype(
        {
            "count_train": "int",
            "count_test": "int",
            "count_total": "int",
            event_name + "_train": "int",
            event_name + "_test": "int",
            event_name + "_total": "int",
        }
    )

    train_test_stats_plot = train_test_stats

    train_rng = train_sample_bin[
        [predictor_binned, predictor_binned_range]
    ].drop_duplicates()
    test_rng = test_sample_bin[
        [predictor_binned, predictor_binned_range]
    ].drop_duplicates()
    total_rng = total_sample_bin[
        [predictor_binned, predictor_binned_range]
    ].drop_duplicates()

    train_rng.rename(
        columns={predictor_binned_range: predictor_binned_range + "_train"},
        inplace=True,
    )
    test_rng.rename(
        columns={predictor_binned_range: predictor_binned_range + "_test"}, inplace=True
    )
    total_rng.rename(
        columns={predictor_binned_range: predictor_binned_range + "_total"},
        inplace=True,
    )

    predictor_info = train_rng.merge(test_rng, on=predictor_binned, how="inner").merge(
        total_rng, on=predictor_binned, how="inner", sort=True
    )
    train_test_stats_plot = train_test_stats_plot.merge(
        predictor_info, on=predictor_binned, how="inner"
    )

    train_test_stats_plot.columns = [
        predictor_binned,
        "count_train",
        "count_test",
        "count_total",
        "count_train (%)",
        event_name + "_train",
        event_name + "_test",
        event_name + "_total",
        event_name + "_train (%)",
        event_rate_name + "_train (%)",
        event_rate_name + "_test (%)",
        event_rate_name + "_total (%)",
        predictor_binned_range + "_train",
        predictor_binned_range + "_test",
        predictor_binned_range + "_total",
    ]

    train_test_stats_plot[predictor_binned_range + "_train"] = category_ordering(
        data=train_test_stats_plot, column=predictor_binned_range + "_train"
    )
    train_test_stats_plot[predictor_binned_range + "_test"] = category_ordering(
        data=train_test_stats_plot, column=predictor_binned_range + "_test"
    )
    train_test_stats_plot[predictor_binned_range + "_total"] = category_ordering(
        data=train_test_stats_plot, column=predictor_binned_range + "_total"
    )

    if bins_details_only:
        return train_test_stats_plot

    train_test_line_bar_plot(
        x_train=predictor_binned_range + "_train",
        y_train=event_name + "_train",
        y2_train=event_rate_name + "_train (%)",
        x_test=predictor_binned_range + "_test",
        y_test=event_name + "_test",
        y2_test=event_rate_name + "_test (%)",
        x_total=predictor_binned_range + "_total",
        y_total=event_name + "_total",
        y2_total=event_rate_name + "_total (%)",
        data=train_test_stats_plot,
        ylabel=event_name.capitalize(),
        y2label=event_name.capitalize() + " rate (%)",
        title=event_name.capitalize() + " rate (%) by " + predictor,
        figsize=figsize,
        rotation=rotation,
    )
    if plots_only:
        return None
    else:
        return train_test_stats_final


def univariate_analysis_categorical(
    train_sample,
    test_sample,
    total_sample,
    predictor,
    response,
    rotation=0,
    prefix=None,
    figsize=(24, 6),
    filter_sample_id=None,
    filter_object=None,
    filter_minimum=None,
    order_by=None,
    order_ascending=True,
    event_name="event",
    event_rate_name="event_rate",
    plots_only=False,
):

    train_test_stats = pd.concat(
        [
            pd.Series(train_sample.groupby(predictor).size(), name="count_train"),
            pd.Series(test_sample.groupby(predictor).size(), name="count_test"),
            pd.Series(total_sample.groupby(predictor).size(), name="count_total"),
            pd.Series(
                round(
                    train_sample.groupby(predictor).size()
                    / total_sample.groupby(predictor).size()
                    * 100,
                    2,
                ),
                name="count_train (%)",
            ),
            pd.Series(
                train_sample.groupby(predictor)[response].sum(),
                name=event_name + "_train",
            ),
            pd.Series(
                test_sample.groupby(predictor)[response].sum(),
                name=event_name + "_test",
            ),
            pd.Series(
                total_sample.groupby(predictor)[response].sum(),
                name=event_name + "_total",
            ),
            pd.Series(
                round(
                    train_sample.groupby(predictor)[response].sum()
                    / total_sample.groupby(predictor)[response].sum()
                    * 100,
                    2,
                ),
                name=event_name + "_train (%)",
            ),
            pd.Series(
                round(
                    train_sample.groupby(predictor)[response].sum()
                    / train_sample.groupby(predictor).size()
                    * 100,
                    2,
                ),
                name=event_rate_name + "_train (%)",
            ),
            pd.Series(
                round(
                    test_sample.groupby(predictor)[response].sum()
                    / test_sample.groupby(predictor).size()
                    * 100,
                    2,
                ),
                name=event_rate_name + "_test (%)",
            ),
            pd.Series(
                round(
                    total_sample.groupby(predictor)[response].sum()
                    / total_sample.groupby(predictor).size()
                    * 100,
                    2,
                ),
                name=event_rate_name + "_total (%)",
            ),
        ],
        axis=1,
        sort=False,
    )

    train_test_stats.index.name = predictor
    train_test_stats.reset_index(inplace=True)

    if filter_minimum:
        train_test_stats = train_test_stats[
            train_test_stats[filter_object + "_" + filter_sample_id] >= filter_minimum
        ]

    if order_by:
        train_test_stats.sort_values(
            by=order_by, ascending=order_ascending, inplace=True
        )

    train_test_stats[predictor] = category_ordering(
        data=train_test_stats, column=predictor, prefix=prefix
    )

    train_test_line_bar_plot(
        x_train=predictor,
        y_train=event_name + "_train",
        y2_train=event_rate_name + "_train (%)",
        x_test=predictor,
        y_test=event_name + "_test",
        y2_test=event_rate_name + "_test (%)",
        x_total=predictor,
        y_total=event_name + "_total",
        y2_total=event_rate_name + "_total (%)",
        data=train_test_stats,
        ylabel=event_name.capitalize(),
        y2label=event_name.capitalize() + " rate (%)",
        title=event_name.capitalize() + " rate (%) by " + predictor,
        figsize=figsize,
        rotation=rotation,
    )
    if plots_only:
        return None

    train_test_stats_sum = pd.DataFrame(train_test_stats.sum()).transpose()
    train_test_stats_sum.loc[0, [predictor]] = "Total"
    train_test_stats_sum.loc[0, "count_train (%)"] = round(
        train_test_stats_sum.loc[0, "count_train"]
        / train_test_stats_sum.loc[0, "count_total"]
        * 100,
        2,
    )
    train_test_stats_sum.loc[0, event_name + "_train (%)"] = round(
        train_test_stats_sum.loc[0, event_name + "_train"]
        / train_test_stats_sum.loc[0, event_name + "_total"]
        * 100,
        2,
    )
    train_test_stats_sum.loc[0, event_rate_name + "_train (%)"] = round(
        train_test_stats_sum.loc[0, event_name + "_train"]
        / train_test_stats_sum.loc[0, "count_train"]
        * 100,
        2,
    )
    train_test_stats_sum.loc[0, event_rate_name + "_test (%)"] = round(
        train_test_stats_sum.loc[0, event_name + "_test"]
        / train_test_stats_sum.loc[0, "count_test"]
        * 100,
        2,
    )
    train_test_stats_sum.loc[0, event_rate_name + "_total (%)"] = round(
        train_test_stats_sum.loc[0, event_name + "_total"]
        / train_test_stats_sum.loc[0, "count_total"]
        * 100,
        2,
    )

    train_test_stats_final = pd.concat(
        [train_test_stats, train_test_stats_sum], axis=0, sort=False, ignore_index=True
    )
    train_test_stats_final.fillna(0, inplace=True)
    train_test_stats_final = train_test_stats_final.astype(
        {
            "count_train": "int",
            "count_test": "int",
            "count_total": "int",
            event_name + "_train": "int",
            event_name + "_test": "int",
            event_name + "_total": "int",
        }
    )

    return train_test_stats_final


def model_performance_initialise(
    model,
    X_train,
    X_test,
    df_train,
    df_test,
    response,
    lost_opportunity,
    saved_cost,
    agg_func_saved_cost,
    predicted_probability,
    other_columns,
    xgboost=False,
):
    if xgboost:
        pred_prob_train = pd.Series(model.predict(X_train), name=predicted_probability)
        pred_prob_test = pd.Series(model.predict(X_test), name=predicted_probability)
    else:
        pred_prob_train = pd.Series(
            model.predict_proba(X_train)[:, 1], name=predicted_probability
        )
        pred_prob_test = pd.Series(
            model.predict_proba(X_test)[:, 1], name=predicted_probability
        )

    train_scored = pd.concat([df_train, pred_prob_train], axis=1)
    test_scored = pd.concat([df_test, pred_prob_test], axis=1)

    train_scored_bid = pd.concat(
        [
            train_scored.groupby("bid")[lost_opportunity].agg(max),
            train_scored.groupby("bid")[saved_cost].agg(agg_func_saved_cost),
            train_scored.groupby("bid")[predicted_probability].agg(sum),
            train_scored.groupby("bid")[response].agg(max),
            train_scored.groupby("bid").agg({i: max for i in other_columns}),
        ],
        axis=1,
    )

    test_scored_bid = pd.concat(
        [
            test_scored.groupby("bid")[lost_opportunity].agg(max),
            test_scored.groupby("bid")[saved_cost].agg(agg_func_saved_cost),
            test_scored.groupby("bid")[predicted_probability].agg(sum),
            test_scored.groupby("bid")[response].agg(max),
            test_scored.groupby("bid").agg({i: max for i in other_columns}),
        ],
        axis=1,
    )

    train_scored_bid.reset_index(inplace=True)
    test_scored_bid.reset_index(inplace=True)
    return train_scored_bid, test_scored_bid


def model_performance(
    train_scored_bid,
    test_scored_bid,
    prediction,
    response,
    event_name,
    predicted_probability,
    lost_opportunity,
    saved_cost,
    trip_type,
    display_results=False,
    threshold=0.5,
):
    prediction_train_bid = pd.Series(
        pp.binarize([train_scored_bid[predicted_probability]], threshold)[0],
        name=prediction,
    )
    prediction_test_bid = pd.Series(
        pp.binarize([test_scored_bid[predicted_probability]], threshold)[0],
        name=prediction,
    )

    train_predicted_bid = pd.concat([train_scored_bid, prediction_train_bid], axis=1)
    test_predicted_bid = pd.concat([test_scored_bid, prediction_test_bid], axis=1)
    total_predicted_bid = pd.concat([train_scored_bid, test_scored_bid], axis=0)
    evaluate_guarantee_model
    auroc_train = round(
        m.roc_auc_score(
            train_predicted_bid[response], train_predicted_bid[predicted_probability]
        ),
        4,
    )
    auroc_test = round(
        m.roc_auc_score(
            test_predicted_bid[response], test_predicted_bid[predicted_probability]
        ),
        4,
    )
    somers_d_train = round(2 * auroc_train - 1, 4)
    somers_d_test = round(2 * auroc_test - 1, 4)

    ct_train = classification_table(
        actual_response=train_predicted_bid[response],
        predicted_prob=train_predicted_bid[predicted_probability],
        threshold=threshold,
    )
    ct_test = classification_table(
        actual_response=test_predicted_bid[response],
        predicted_prob=test_predicted_bid[predicted_probability],
        threshold=threshold,
    )

    train_predicted_bid_tp = train_predicted_bid[
        (train_predicted_bid[prediction] == 1) & (train_predicted_bid[response] == 1)
    ]
    train_predicted_bid_fp = train_predicted_bid[
        (train_predicted_bid[prediction] == 1) & (train_predicted_bid[response] == 0)
    ]
    test_predicted_bid_tp = test_predicted_bid[
        (test_predicted_bid[prediction] == 1) & (test_predicted_bid[response] == 1)
    ]
    test_predicted_bid_fp = test_predicted_bid[
        (test_predicted_bid[prediction] == 1) & (test_predicted_bid[response] == 0)
    ]

    results_train_test = pd.DataFrame(
        {
            "threshold": [threshold, threshold],
            "id": [1, 2],
            "trip_type": [trip_type, trip_type],
            "sample_id": ["TRAIN", "TEST"],
            "level": ["BID", "BID"],
            "count": [len(train_predicted_bid), len(test_predicted_bid)],
            event_name: [
                train_predicted_bid[response].sum(),
                test_predicted_bid[response].sum(),
            ],
            event_name
            + "_rate": [
                round(
                    100
                    * train_predicted_bid[response].sum()
                    / len(train_predicted_bid),
                    2,
                ),
                round(
                    100 * test_predicted_bid[response].sum() / len(test_predicted_bid),
                    2,
                ),
            ],
            "total_cost": [
                train_predicted_bid[saved_cost].sum(),
                test_predicted_bid[saved_cost].sum(),
            ],
            "tp": [ct_train[1][1], ct_test[1][1]],
            "fp": [ct_train[0][1], ct_test[0][1]],
            "tn": [ct_train[0][0], ct_test[0][0]],
            "fn": [ct_train[1][0], ct_test[1][0]],
            "fp/tp": [
                round(ct_train[0][1] / (ct_train[1][1] + 1), 2),
                round(ct_test[0][1] / (ct_test[1][1] + 1), 2),
            ],
            "auroc": [auroc_train, auroc_test],
            "somers_d": [somers_d_train, somers_d_test],
            "saved_cost": [
                round(train_predicted_bid_tp[saved_cost].sum(), 2),
                round(test_predicted_bid_tp[saved_cost].sum(), 2),
            ],
            "lost_margin": [
                round(train_predicted_bid_fp[lost_opportunity].sum(), 2),
                round(test_predicted_bid_fp[lost_opportunity].sum(), 2),
            ],
            "net_savings": [
                round(
                    train_predicted_bid_tp[saved_cost].sum()
                    - train_predicted_bid_fp[lost_opportunity].sum(),
                    2,
                ),
                round(
                    test_predicted_bid_tp[saved_cost].sum()
                    - test_predicted_bid_fp[lost_opportunity].sum(),
                    2,
                ),
            ],
        }
    )

    results_total = pd.DataFrame(
        {
            "threshold": threshold,
            "id": 3,
            "trip_type": trip_type,
            "sample_id": "TOTAL",
            "level": "BID",
            "count": len(total_predicted_bid),
            event_name: results_train_test[event_name].sum(),
            event_name
            + "_rate": round(
                100 * results_train_test[event_name].sum() / len(total_predicted_bid), 2
            ),
            "total_cost": results_train_test.total_cost.sum(),
            "tp": results_train_test.tp.sum(),
            "fp": results_train_test.fp.sum(),
            "tn": results_train_test.tn.sum(),
            "fn": results_train_test.fn.sum(),
            "fp/tp": round(
                results_train_test.fp.sum() / (results_train_test.tp.sum() + 1), 2
            ),
            "auroc": round(results_train_test.auroc.mean(), 4),
            "somers_d": round(results_train_test.somers_d.mean(), 4),
            "saved_cost": results_train_test.saved_cost.sum(),
            "lost_margin": results_train_test.lost_margin.sum(),
            "net_savings": results_train_test.net_savings.sum(),
        },
        index=[0],
    )

    results = (
        pd.concat([results_train_test, results_total], axis=0, sort=False)
        .sort_values(by=["threshold", "id"])
        .drop("id", axis=1)
    )
    results.reset_index(drop=True, inplace=True)

    if display_results:
        sns.set("notebook")
        plot_classification_table(
            ct_train, [0, 1], title="Classification table", cmap=plt.cm.Blues
        )
        plot_classification_table(
            ct_test, [0, 1], title="Classification table", cmap=plt.cm.Blues
        )
        print(
            classification_statistics(
                actual_response=train_predicted_bid[response],
                predicted_prob=train_predicted_bid[predicted_probability],
                threshold=threshold,
            )
        )
        print(
            classification_statistics(
                actual_response=test_predicted_bid[response],
                predicted_prob=test_predicted_bid[predicted_probability],
                threshold=threshold,
            )
        )

        return results, train_predicted_bid, test_predicted_bid
    return results


def model_performance_thresholds(
    train_scored_bid,
    test_scored_bid,
    prediction,
    response,
    event_name,
    predicted_probability,
    lost_opportunity,
    saved_cost,
    trip_type,
    thresholds,
):
    results_threshold = 0
    for i, threshold in enumerate(thresholds):
        results = model_performance(
            train_scored_bid,
            test_scored_bid,
            prediction,
            response=response,
            event_name=event_name,
            predicted_probability=predicted_probability,
            lost_opportunity=lost_opportunity,
            trip_type=trip_type,
            saved_cost=saved_cost,
            threshold=threshold,
        )
        if i == 0:
            results_threshold = results
        else:
            results_threshold = pd.concat([results_threshold, results], axis=0)
    results_threshold.reset_index(drop=True, inplace=True)
    return results_threshold


def train_test_line_bar_plot(
    x_train,
    y_train,
    y2_train,
    x_test,
    y_test,
    y2_test,
    x_total,
    y_total,
    y2_total,
    data,
    ylabel,
    y2label,
    title,
    rotation=None,
    figsize=None,
):
    plt.figure(figsize=figsize)
    sns.set_style("dark")
    plt.subplot(1, 3, 1)
    ax = sns.barplot(x=x_train, y=y_train, data=data, palette=["#F9971E"])
    ax2 = plt.twinx()
    ax2 = sns.lineplot(
        x=x_train,
        y=y2_train,
        data=data,
        marker="o",
        color="#009882",
        linewidth=2.5,
        label="line 1",
    )

    ax.set(xlabel="")
    ax.tick_params(axis="x", labelsize=10.5)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title(title + ", TRAIN", fontdict={"fontsize": 13.5, "fontweight": "bold"})
    ax.set_ylabel(ylabel + "s", fontdict={"fontsize": 12.5, "fontweight": "medium"})

    ax2.set(xlabel="", ylabel="")
    ax2.tick_params(axis="y", labelsize=11)
    plt.legend(
        labels=[ylabel + " rate (%)"],
        labelspacing=1,
        loc=2,
        fontsize="12",
        title_fontsize="12",
    )

    if rotation:
        for item in ax.get_xticklabels():
            item.set_rotation(rotation)

    ax3 = plt.subplot(1, 3, 2)
    ax3 = sns.barplot(x=x_test, y=y_test, data=data, palette=["#F9971E"])

    ax4 = plt.twinx()
    ax4 = sns.lineplot(
        x=x_test,
        y=y2_test,
        data=data,
        marker="o",
        color="#009882",
        linewidth=2.5,
        label="line 1",
    )

    ax3.set(xlabel="", ylabel="")
    ax3.tick_params(axis="x", labelsize=10.5)
    ax3.tick_params(axis="y", labelsize=11)
    ax3.set_title(title + ", TEST", fontdict={"fontsize": 13.5, "fontweight": "bold"})

    if rotation:
        ax3.tick_params(axis="x", rotation=rotation)
        # for item in ax3.get_xticklabels():
        #     item.set_rotation(rotation)

    ax4.set(xlabel="", ylabel="")
    ax4.tick_params(axis="y", labelsize=11)

    plt.legend(
        labels=[ylabel + " rate (%)"],
        labelspacing=1,
        loc=2,
        fontsize="12",
        title_fontsize="12",
    )

    ax5 = plt.subplot(1, 3, 3)
    ax5 = sns.barplot(x=x_total, y=y_total, data=data, palette=["#F9971E"])

    ax5.set(xlabel="", ylabel="")

    ax6 = plt.twinx()
    ax6 = sns.lineplot(
        x=x_total,
        y=y2_total,
        data=data,
        marker="o",
        color="#009882",
        linewidth=2.5,
        label="line 1",
    )

    ax5.tick_params(axis="x", labelsize=10.5)
    ax5.tick_params(axis="y", labelsize=11)
    ax5.set_title(title + ", TOTAL", fontdict={"fontsize": 13.5, "fontweight": "bold"})

    if rotation:
        for item in ax5.get_xticklabels():
            item.set_rotation(rotation)

    ax6.set(xlabel="", ylabel="")
    ax6.tick_params(axis="y", labelsize=11)
    ax6.set_ylabel(y2label, fontdict={"fontsize": 12.5, "fontweight": "medium"})

    plt.legend(
        labels=[ylabel + " rate (%)"],
        labelspacing=1,
        loc=2,
        fontsize="12",
        title_fontsize="12",
    )


def line_bar_plot(x, y, y2, data, ylabel, y2label, title, rotation, figsize):
    plt.figure(figsize=figsize)
    sns.set_style("dark")
    ax = sns.barplot(x=x, y=y, data=data, palette=["#F9971E"])

    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), ".2f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )

    ax2 = plt.twinx()
    ax2 = sns.lineplot(
        x=x, y=y2, data=data, marker="o", color="#009882", linewidth=2.5, label="line 1"
    )

    ax.set(xlabel="")
    ax.tick_params(axis="x", labelsize=10.5)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title(title, fontdict={"fontsize": 13.5, "fontweight": "bold"})
    ax.set_ylabel(ylabel, fontdict={"fontsize": 12.5, "fontweight": "medium"})

    ax2.set(xlabel="")
    ax2.set_ylabel(y2label, fontdict={"fontsize": 12.5, "fontweight": "medium"})
    ax2.tick_params(axis="y", labelsize=11)

    plt.legend(
        labels=[y2label], labelspacing=1, loc=9, fontsize="12", title_fontsize="12"
    )

    if rotation:
        for item in ax.get_xticklabels():
            item.set_rotation(rotation)

    return None


def line_bar_plot_ts(x, y, y2, data, ylabel, y2label, title, rotation, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("dark")
    ax = sns.barplot(x=data[x].dt.date, y=y, data=data, palette=["#F9971E"])
    fig.autofmt_xdate()
    ax2 = plt.twinx()
    ax2 = sns.lineplot(
        x=data.index,
        y=y2,
        data=data,
        marker="o",
        color="#009882",
        linewidth=2.5,
        label="line 1",
    )

    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), ".0f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )

    ax.set(xlabel="")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title(title, fontdict={"fontsize": 20, "fontweight": "bold"})
    ax.set_ylabel(ylabel, fontdict={"fontsize": 14, "fontweight": "medium"})

    ax2.set(xlabel="")
    ax2.set_ylabel(y2label, fontdict={"fontsize": 14, "fontweight": "medium"})
    ax2.tick_params(axis="y", labelsize=11)
    fig.autofmt_xdate()
    plt.legend(
        labels=[y2label], labelspacing=1, loc=1, fontsize="12", title_fontsize="12"
    )

    if rotation:
        for item in ax.get_xticklabels():
            item.set_rotation(rotation)

    return None


def weight_of_evidence_categorical(
    train_sample,
    predictor,
    response,
    event_name,
    figsize=(8, 6),
    rotation=0,
    order_by="event_rate (%)",
    ascending=True,
    prefix=" ",
):
    woe = (
        train_sample[[predictor, response]]
        .groupby(predictor)[response]
        .agg(["count", "sum"])
    )
    woe.index.name = predictor
    woe.reset_index(inplace=True)
    woe.columns = [predictor, "count", "events"]

    woe["non_events"] = woe["count"] - woe["events"]
    woe["events (%)"] = round(np.maximum(woe.events, 0.5) / woe.events.sum() * 100, 2)
    woe["non_events (%)"] = round(
        np.maximum(woe.non_events, 0.5) / woe.non_events.sum() * 100, 2
    )
    woe["event_rate (%)"] = round(woe["events"] / woe["count"] * 100, 2)
    woe["woe"] = round(np.log(woe["non_events (%)"] / woe["events (%)"]), 4)
    woe.sort_values(by=order_by, ascending=ascending, inplace=True)

    woe[predictor + "_category"] = woe[predictor]
    woe[predictor + "_category"] = category_ordering(
        data=woe, column=predictor + "_category", prefix=prefix
    )

    line_bar_plot(
        x=predictor + "_category",
        y="woe",
        y2="event_rate (%)",
        data=woe,
        ylabel="WOE",
        y2label=event_name.capitalize() + " rate (%)",
        title="WOE & " + event_name.capitalize() + " rate (%) for " + predictor,
        figsize=figsize,
        rotation=rotation,
    )

    woe.reset_index(inplace=True, drop=True)
    woe_sum = pd.DataFrame(woe.sum()).transpose()
    woe_sum.loc[0, [predictor]] = "Total"
    woe_sum.loc[0, "event_rate (%)"] = round(
        woe_sum.loc[0, "events"] / woe_sum.loc[0, "count"] * 100, 2
    )
    woe_sum.loc[0, "woe"] = 0
    woe_final = pd.concat([woe, woe_sum], axis=0, sort=False, ignore_index=True)
    woe_final = woe_final.astype(
        {
            "count": "int",
            "events": "int",
            "non_events": "int",
            "non_events (%)": "float",
            "events (%)": "float",
            "event_rate (%)": "float",
            "woe": "float",
        }
    )

    return woe_final


def univariate_analysis_somers_d(
    train_sample, test_sample, total_sample, predictors, response, ascending=False
):
    ua_somers_d_train = []
    ua_somers_d_test = []
    ua_somers_d_total = []
    ua_somers_d_abs_train = []

    for predictor in predictors:
        ua_somers_d_train.append(
            round(
                2 * m.roc_auc_score(train_sample[response], train_sample[predictor])
                - 1,
                4,
            )
            * 100
        )
        ua_somers_d_abs_train.append(
            abs(
                round(
                    2 * m.roc_auc_score(train_sample[response], train_sample[predictor])
                    - 1,
                    4,
                )
                * 100
            )
        )
        ua_somers_d_test.append(
            round(
                2 * m.roc_auc_score(test_sample[response], test_sample[predictor]) - 1,
                4,
            )
            * 100
        )
        ua_somers_d_total.append(
            round(
                2 * m.roc_auc_score(total_sample[response], total_sample[predictor])
                - 1,
                4,
            )
            * 100
        )

    ua_somers_d = (
        pd.DataFrame(
            {
                "train": ua_somers_d_train,
                "test": ua_somers_d_test,
                "total": ua_somers_d_total,
                "train_abs": ua_somers_d_abs_train,
            },
            index=predictors,
        )
        .sort_values(by="train_abs", ascending=ascending)
        .drop("train_abs", axis=1)
    )
    return ua_somers_d


def make_continuous_monotonous(
    train_sample,
    test_sample,
    total_sample,
    predictor,
    bins,
    response,
    event_name,
    event_rate_name,
    rounding=3,
    spike="up",
    fixed_value=None,
):
    ua = univariate_analysis_continuous(
        train_sample=train_sample,
        test_sample=test_sample,
        total_sample=total_sample,
        response=response,
        predictor=predictor,
        bins=bins,
        event_name=event_name,
        event_rate_name=event_rate_name,
        rounding=rounding,
        bins_details_only=True,
    )
    if fixed_value:
        train_sample[predictor + "_mnt"] = np.abs(train_sample[predictor] - fixed_value)
        test_sample[predictor + "_mnt"] = np.abs(test_sample[predictor] - fixed_value)
        total_sample[predictor + "_mnt"] = np.abs(total_sample[predictor] - fixed_value)
    else:
        if spike == "up":
            bin_rng_str = (
                ua.loc[
                    ua[event_rate_name + "_total (%)"].idxmax(skipna=True),
                    predictor + "_bin_rng_total",
                ]
                .replace("(", "")
                .replace(")", "")
            )
            print(
                "Curve shape:",
                spike,
                "\nBin range with the maximum",
                event_rate_name,
                "on total sample:",
                ua.loc[
                    ua[event_rate_name + "_total (%)"].idxmax(skipna=True),
                    predictor + "_bin_rng_total",
                ],
            )
        else:
            bin_rng_str = (
                ua.loc[
                    ua[event_rate_name + "_total (%)"].idxmin(skipna=True),
                    predictor + "_bin_rng_total",
                ]
                .replace("(", "")
                .replace(")", "")
            )
            print(
                "Curve shape:",
                spike,
                "\nBin range with the minimum",
                event_rate_name,
                " on total sample:",
                ua.loc[
                    ua[event_rate_name + "_total (%)"].idxmax(skipna=True),
                    predictor + "_bin_rng_total",
                ],
            )

        bin_rng_lower = float(bin_rng_str[: bin_rng_str.find(",")])
        bin_rng_upper = float(bin_rng_str[bin_rng_str.find(",") + 1:])
        bin_rng_mid = (bin_rng_upper + bin_rng_lower) / 2
        print("Bin range mid-point used to adjust predictor values:", bin_rng_mid)

        train_sample[predictor + "_mnt"] = np.abs(train_sample[predictor] - bin_rng_mid)
        test_sample[predictor + "_mnt"] = np.abs(test_sample[predictor] - bin_rng_mid)
        total_sample[predictor + "_mnt"] = np.abs(total_sample[predictor] - bin_rng_mid)

    return train_sample, test_sample, total_sample


def display_roc_curve(response, probability):
    fpr, tpr, thresholds = roc_curve(response, probability)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    return thresholds


def outlier_analysis(data, predictors, percentile_range=5, rounding=2):
    percentiles = pd.DataFrame()
    for i in it.chain(
        range(0, percentile_range + 1),
        range(25, 26),
        range(50, 51),
        range(75, 76),
        range(100 - percentile_range, 101),
    ):
        if i == 0:
            percentiles = pd.DataFrame(
                data=pd.Series(
                    round(data[predictors].quantile(i / 100), rounding),
                    name="p" + str(i),
                )
            )
        else:
            p = pd.Series(
                data=round(data[predictors].quantile(i / 100), rounding),
                name="p" + str(i),
            )
            percentiles = pd.concat([percentiles, p], axis=1, sort=False)
    return percentiles


def model_performance_plot(
    data, savings_column, net_savings_min_total, net_savings_tick, subcategory
):
    figure_thresholds = data[
        (data.sample_id == "TOTAL") & (data[savings_column] > net_savings_min_total)
    ].threshold
    figure_data = data[data.threshold.isin(figure_thresholds)]
    net_savings_max = round(figure_data.net_savings.max() + net_savings_tick, -3)
    net_savings_min = round(figure_data.net_savings.min(), -4)

    sns.set_style("darkgrid", {"axes.facecolor": "#E5EAEF"})
    sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 2})

    plt.figure(figsize=(12, 8))
    ax = sns.lineplot(
        x="threshold",
        y="net_savings",
        hue="sample_id",
        data=figure_data,
        marker="o",
        palette=["#F9971E", "#009882", "#660000"],
    )

    ax.set(
        xlabel="Cut-off threshold",
        ylabel="Net guarantee savings",
        title="Net guarantee costs savings for different cut-off thresholds",
        xticks=figure_thresholds,
        yticks=np.arange(net_savings_min, net_savings_max, net_savings_tick),
    )

    ax.tick_params(which="both", width=2)

    plt.legend(
        labels=["TRAIN", "TEST", "TOTAL"],
        labelspacing=1.1,
        title="Sample ID",
        loc=1,
        fontsize="12",
        title_fontsize="12",
    )
    return None


def lin_reg_best_p_subset(
    train_sample,
    test_sample,
    predictors,
    p,
    response,
    best_p_subset_criterion,
    prediction,
    fit_intercept,
    order_ascending=True,
):
    subsets_of_predictors = list(it.combinations(predictors, p))
    results = pd.DataFrame(data=0, columns=["p"], index=subsets_of_predictors)

    for i in subsets_of_predictors:
        shortlist = list(i)
        X_train, y_train = train_sample[shortlist], train_sample[response]
        X_test, y_test = test_sample[shortlist], test_sample[response]

        model = lm.LinearRegression(fit_intercept=fit_intercept)
        model.fit(X=X_train, y=y_train)

        train_scored = pd.concat(
            [train_sample, pd.Series(model.predict(X_train).round(2), name=prediction)],
            axis=1,
        )
        test_scored = pd.concat(
            [test_sample, pd.Series(model.predict(X_test).round(2), name=prediction)],
            axis=1,
        )
        train_scored["error"] = train_scored[prediction] - train_scored[response]
        test_scored["error"] = test_scored[prediction] - test_scored[response]

        results["best_of"] = len(subsets_of_predictors)
        results["p"] = p

        for name, scored, X, y in [
            ("train", train_scored, X_train, y_train),
            ("test", test_scored, X_test, y_test),
        ]:
            results.loc[i, name + "_" + prediction] = scored[prediction].sum().round(2)
            results.loc[i, name + "_sum_error"] = scored["error"].sum().round(2)
            results.loc[i, name + "_mae"] = scored["error"].abs().mean().round(2)
            results.loc[i, name + "_mse"] = (
                scored["error"].apply(lambda x: x * x).mean().round(2)
            )
            results.loc[i, name + "_r_squared"] = round(model.score(X, y), 4)
    results.sort_values(
        by=best_p_subset_criterion, ascending=order_ascending, inplace=True
    )
    return results


def lin_reg_best_subset(
    train_sample,
    test_sample,
    predictors,
    response,
    fit_intercept,
    best_p_subset_criterion,
    prediction,
    order_ascending=True,
):
    result = pd.DataFrame()
    for i in range(1, len(predictors) + 1):
        df = lin_reg_best_p_subset(
            train_sample=train_sample,
            test_sample=test_sample,
            predictors=predictors,
            prediction=prediction,
            p=i,
            response=response,
            best_p_subset_criterion=best_p_subset_criterion,
            fit_intercept=fit_intercept,
            order_ascending=order_ascending,
        )
        result = pd.concat([result, df.head(1)], axis=0)
    return result


def reg_tree_estimates_variability(
    sample,
    predictors,
    response,
    n,
    train_size,
    stratify,
    min_sample_leaf,
    criterion,
    prediction,
):
    results = pd.DataFrame(data=0, columns=["split_id"], index=range(1, n + 1))
    for i in range(1, n + 1):
        df_train_raw, df_test_raw = ms.train_test_split(
            sample,
            train_size=train_size,
            stratify=sample[stratify],
            random_state=100 * i,
        )
        df_train_raw.reset_index(drop=True, inplace=True)
        df_test_raw.reset_index(drop=True, inplace=True)

        df_train = pd.DataFrame(
            df_train_raw.loc[:, ~df_train_raw.columns.isin(predictors)]
        )
        df_test = pd.DataFrame(
            df_test_raw.loc[:, ~df_test_raw.columns.isin(predictors)]
        )

        for j in predictors:
            df_train[j] = df_train_raw[j].fillna(df_train_raw[j].mean())
            df_test[j] = df_test_raw[j].fillna(df_train_raw[j].mean())

        X_train, y_train = df_train[predictors], df_train[response]
        X_test, y_test = df_test[predictors], df_test[response]

        dtr = t.DecisionTreeRegressor(
            criterion=criterion,
            min_samples_leaf=min_sample_leaf,
            random_state=100,
            max_leaf_nodes=None,
        )  # max_depth=5,
        dtr.fit(X_train, y_train)

        train_scored = pd.concat(
            [df_train, pd.Series(dtr.predict(X_train).round(2), name=prediction)],
            axis=1,
        )
        test_scored = pd.concat(
            [df_test, pd.Series(dtr.predict(X_test).round(2), name=prediction)], axis=1
        )

        train_scored["error"] = train_scored[prediction] - train_scored[response]
        test_scored["error"] = test_scored[prediction] - test_scored[response]

        results["split_id"] = results.index
        results["leaves"] = dtr.get_n_leaves()

        for name, scored, X, y in [
            ("train", train_scored, X_train, y_train),
            ("test", test_scored, X_test, y_test),
        ]:
            results.loc[i, name + "_" + prediction] = scored[prediction].sum().round(2)
            results.loc[i, name + "_sum_error"] = scored["error"].sum().round(2)
            results.loc[i, name + "_mae"] = scored["error"].abs().mean().round(2)
            results.loc[i, name + "_mse"] = (
                scored["error"].apply(lambda x: x * x).mean().round(2)
            )
            results.loc[i, name + "_r_squared"] = round(dtr.score(X, y), 4)
    return results


def estimated_guarantee_cost(sample):
    sample["kiwi_guarantee_woe"] = sample["kiwi_guarantee"].apply(
        lambda x: 2.1450 if x == 0 else -0.1699
    )
    sample["arr_departure_hour" + "_mnt"] = np.abs(sample["arr_departure_hour"] - 16)
    sample["oneway"] = sample["trip_type"].apply(lambda x: 1 if x == "oneway" else 0)
    sample["booking_window_sq"] = sample["booking_window"] ** 2
    sample["gr"] = sample["gr"].mask(pd.isnull, 0)

    pg_shortlist = [
        "stopover",
        "mct",
        "gr",
        "booking_window",
        "booking_window_sq",
        "arr_departure_hour_mnt",
        "bid_total_distance",
        "kiwi_guarantee_woe",
    ]
    cgg_shortlist = ["oneway", "kiwi_guarantees_orig", "passengers", "token_price"]

    mean = np.load(file="pg_standardize_mean.npy")
    std = np.load(file="pg_standardize_std.npy")

    X_pg = sample[pg_shortlist]
    X_pg -= mean
    X_pg /= std

    reg = pickle.load(open("pg_logistic_regression.bin", "rb"))
    dtr = pickle.load(open("cgg_decision_tree.bin", "rb"))

    scored_pg = pd.concat(
        [sample, pd.Series(reg.predict_proba(X_pg)[:, 1], name="pg")], axis=1
    )
    scored_pg_bid = pd.concat(
        [
            scored_pg.groupby(["bid", "trip_type", "estimated_costs"]).agg(
                {"total_cost": max, "pg": sum}
            ),
            scored_pg.groupby(["bid", "trip_type", "estimated_costs"]).agg(
                {i: max for i in cgg_shortlist}
            ),
        ],
        axis=1,
    )
    scored_pg_bid.reset_index(inplace=True)

    X_cgg = scored_pg_bid[
        ["oneway", "kiwi_guarantees_orig", "passengers", "token_price"]
    ]
    scored_pg_bid_cgg = pd.concat(
        [scored_pg_bid, pd.Series(dtr.predict(X_cgg).round(2), name="cgg")], axis=1
    )

    scored_pg_bid_cgg["estimated_cost"] = round(
        scored_pg_bid_cgg["cgg"] * scored_pg_bid_cgg["pg"], 4
    )
    scored_pg_bid_cgg["bid"] = scored_pg_bid_cgg["bid"].astype("int")

    if sample.total_cost.any():
        scored_pg_bid_cgg["flg_guarantee"] = scored_pg_bid_cgg["total_cost"].apply(
            lambda x: 1 if x > 0 else 0
        )
    return scored_pg_bid_cgg


def evaluate_guarantee_model(sample, description, estimated_cost_threshold=None):
    scored_output = estimated_guarantee_cost(sample)
    if estimated_cost_threshold:
        scored_output = scored_output.loc[
            scored_output.estimated_cost <= estimated_cost_threshold, :
        ]

    estimated_cost_trip_type = scored_output.groupby(["trip_type"]).agg(
        {
            "bid": "count",
            "total_cost": "sum",
            "estimated_cost": "sum",
            "pg": "mean",
            "flg_guarantee": "mean",
            "cgg": "mean",
        }
    )
    estimated_cost_trip_type.reset_index(inplace=True)
    estimated_cost_trip_type.rename(columns={"bid": "count"}, inplace=True)
    estimated_cost_trip_type["pg"] = estimated_cost_trip_type["pg"].apply(
        lambda x: 100 * x
    )
    estimated_cost_trip_type.rename(
        columns={"flg_guarantee": "gr (%)", "pg": "pg (%)"}, inplace=True
    )
    estimated_cost_trip_type["gr (%)"] = estimated_cost_trip_type["gr (%)"].apply(
        lambda x: 100 * x
    )
    estimated_cost_trip_type["diff (%)"] = (
        estimated_cost_trip_type.estimated_cost / estimated_cost_trip_type.total_cost
        - 1
    ) * 100

    estimated_cost_total = pd.DataFrame(
        scored_output.agg(
            {
                "bid": "count",
                "total_cost": "sum",
                "estimated_cost": "sum",
                "pg": "mean",
                "flg_guarantee": "mean",
                "cgg": "mean",
            }
        )
    ).transpose()
    estimated_cost_total["bid"] = estimated_cost_total.bid.astype("int")
    estimated_cost_total.rename(columns={"bid": "count"}, inplace=True)
    estimated_cost_total["pg"] = estimated_cost_total["pg"].apply(lambda x: 100 * x)
    estimated_cost_total.rename(
        columns={"flg_guarantee": "gr (%)", "pg": "pg (%)"}, inplace=True
    )
    estimated_cost_total["gr (%)"] = estimated_cost_total["gr (%)"].apply(
        lambda x: 100 * x
    )
    estimated_cost_total["diff (%)"] = (
        estimated_cost_total.estimated_cost / estimated_cost_total.total_cost - 1
    ) * 100
    estimated_cost_total["trip_type"] = "total"

    estimated_cost = pd.concat(
        [estimated_cost_trip_type, estimated_cost_total], axis=0, sort=False
    )

    print(
        "ESTIMATED COST SUMMARY -",
        description,
        "SAMPLE, BID LEVEL: cost statistics, percentiles and histogram:\n",
    )
    sns.set_style("darkgrid")
    scored_output.estimated_cost.hist(bins=20, figsize=(6, 4))
    sns.despine()
    # display(
    #     pd.concat(
    #         [
    #             pd.DataFrame(
    #                 scored_output.estimated_cost.quantile(np.arange(0, 1.05, 0.05))
    #             )
    #             .transpose()
    #             .set_index([pd.Index(["NEW"])]),
    #             pd.DataFrame(
    #                 scored_output.estimated_costs.quantile(np.arange(0, 1.05, 0.05))
    #             )
    #             .transpose()
    #             .set_index([pd.Index(["OLD"])]),
    #         ]
    #     )
    # )

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        scored_output.estimated_cost,
        clip=(0, estimated_cost_threshold),
        shade=True,
        label="New model",
    )
    sns.kdeplot(
        scored_output.estimated_costs,
        clip=(0, estimated_cost_threshold),
        shade=True,
        label="Old model",
    )

    print(
        f"OLD MODEL ESTIMATED COSTS: {round(scored_output.estimated_costs.sum(), 2):,.2f} EUR"
    )
    print(
        f"NEW MODEL ESTIMATED COSTS: {round(scored_output.estimated_cost.sum(), 2):,.2f} EUR"
    )

    return scored_output


def logistic_regression_weights(model, shortlist):
    weights = pd.Series(
        np.round(100 * abs(model.coef_[0]) / abs(model.coef_[0]).sum(), 4),
        name="weight (%)",
        index=shortlist,
    )
    coefficient_estimates = pd.concat(
        [
            pd.Series(
                np.append(model.intercept_, model.coef_[0]),
                index=["intercept"] + shortlist,
                name="coefficient estimate",
            ),
            weights,
        ],
        axis=1,
        sort=False,
    )
    coefficient_estimates.loc[["intercept"], "weight (%)"] = 0

    return coefficient_estimates


# Population stability index


def psi_categorical(
    sample_1,
    sample_2,
    variable,
    sample_1_name="sample_1",
    sample_2_name="sample_2",
    figsize=(7, 4),
    rotation=0,
):
    sample_1_output = pd.DataFrame(
        pd.Series(
            sample_1.groupby(variable)[variable].agg("count"), name="count_sample_1"
        )
    )
    sample_1_output["percent_sample_1"] = (
        sample_1_output.count_sample_1 / sample_1_output.count_sample_1.sum()
    )
    sample_1_output.reset_index(inplace=True)

    sample_2_output = pd.DataFrame(
        pd.Series(
            sample_2.groupby(variable)[variable].agg("count"), name="count_sample_2"
        )
    )
    sample_2_output["percent_sample_2"] = (
        sample_2_output.count_sample_2 / sample_2_output.count_sample_2.sum()
    )
    sample_2_output.reset_index(inplace=True)

    output = sample_1_output.merge(sample_2_output, on=variable, how="outer")
    output["psi"] = (output.percent_sample_1 - output.percent_sample_2) * np.log(
        output.percent_sample_1 / output.percent_sample_2
    )
    output = output[
        [
            variable,
            "count_sample_1",
            "count_sample_2",
            "percent_sample_1",
            "percent_sample_2",
            "psi",
        ]
    ]

    psi_plot_raw = output[["percent_sample_1", "percent_sample_2", variable]]
    psi_plot_raw.rename(
        columns={"percent_sample_1": sample_1_name, "percent_sample_2": sample_2_name},
        inplace=True,
    )
    psi_plot = psi_plot_raw.melt(id_vars=[variable]).rename(
        columns={"variable": "Sample", "value": "Percent"}
    )

    output = pd.concat(
        [output, pd.DataFrame(output.sum()).transpose()], axis=0
    ).reset_index(drop=True)
    output.iloc[-1, 0] = "total"

    output.fillna(0, inplace=True)
    output = output.astype({"count_sample_1": "int", "count_sample_2": "int"})

    plt.figure(figsize=figsize)
    ax = sns.barplot(x=variable, y="Percent", hue="Sample", data=psi_plot)
    ax.set(
        xlabel=None,
        ylabel="Proportion",
        title="Population stability index - "
        + variable
        + " = "
        + str(round(output.iloc[-1, -1], 4)),
    )
    plt.legend(loc=1)
    for item in ax.get_xticklabels():
        item.set_rotation(rotation)

    output.rename(
        columns={
            "count_sample_1": "count_" + sample_1_name,
            "percent_sample_1": "percent_" + sample_1_name,
        },
        inplace=True,
    )
    output.rename(
        columns={
            "count_sample_2": "count_" + sample_2_name,
            "percent_sample_2": "percent_" + sample_2_name,
        },
        inplace=True,
    )
    return output


def psi_continuous(
    sample_1,
    sample_2,
    variable,
    bins=10,
    sample_1_name="sample_1",
    sample_2_name="sample_2",
    figsize=(7, 4),
    rotation=45,
    bins_definition="jointly",
):
    if bins_definition == "jointly":
        # Bins edges based on joint (combined) sample percentiles
        sample = pd.concat(
            [
                pd.DataFrame({variable: sample_1[variable], "id": 1}),
                pd.DataFrame({variable: sample_2[variable], "id": 2}),
            ],
            axis=0,
        )
        sample_bin = make_categorical(data=sample, predictors=[variable], bins=bins)
        sample_1_bin, sample_2_bin = (
            sample_bin[sample_bin.id == 1],
            sample_bin[sample_bin.id == 2],
        )

        sample_1_output = pd.DataFrame(
            pd.Series(
                sample_1_bin.groupby([variable + "_bin", variable + "_bin_rng"])[
                    variable
                ].agg("count"),
                name="count_sample_1",
            )
        )
        sample_1_output["percent_sample_1"] = (
            sample_1_output.count_sample_1 / sample_1_output.count_sample_1.sum()
        )
        sample_1_output.reset_index(inplace=True)

        sample_2_output = pd.DataFrame(
            pd.Series(
                sample_2_bin.groupby([variable + "_bin", variable + "_bin_rng"])[
                    variable
                ].agg("count"),
                name="count_sample_2",
            )
        )
        sample_2_output["percent_sample_2"] = (
            sample_2_output.count_sample_2 / sample_2_output.count_sample_2.sum()
        )
        sample_2_output.reset_index(inplace=True)
        output = sample_1_output.merge(
            sample_2_output, on=variable + "_bin_rng", how="outer"
        )
    else:
        # Bins edges based on sample_1 (i.e. reference) percentiles
        sample_1_bin = make_categorical(data=sample_1, predictors=[variable], bins=bins)
        sample_1_output = pd.DataFrame(
            pd.Series(
                sample_1_bin.groupby([variable + "_bin", variable + "_bin_rng"])[
                    variable
                ].agg("count"),
                name="count_sample_1",
            )
        )
        sample_1_output["percent_sample_1"] = (
            sample_1_output.count_sample_1 / sample_1_output.count_sample_1.sum()
        )
        sample_1_output.reset_index(inplace=True)

        bins_edges = sorted(
            np.append(
                sample_1_bin[variable + "_bin_low"].unique(),
                sample_1_bin[variable + "_bin_upp"].max(),
            )
        )
        sample_2_output = pd.DataFrame(
            {"count_sample_2": np.histogram(sample_2[variable], bins_edges)[0]}
        )
        sample_2_output["percent_sample_2"] = (
            sample_2_output.count_sample_2 / sample_2_output.count_sample_2.sum()
        )
        sample_2_output.reset_index(inplace=True)
        sample_2_output.rename(columns={"index": variable + "_bin"}, inplace=True)
        output = sample_1_output.merge(
            sample_2_output, on=variable + "_bin", how="outer"
        )

    output["psi"] = (output.percent_sample_1 - output.percent_sample_2) * np.log(
        output.percent_sample_1 / output.percent_sample_2
    )
    output = output[
        [
            variable + "_bin_rng",
            "count_sample_1",
            "count_sample_2",
            "percent_sample_1",
            "percent_sample_2",
            "psi",
        ]
    ]

    psi_plot_raw = output[
        ["percent_sample_1", "percent_sample_2", variable + "_bin_rng"]
    ]
    psi_plot_raw.rename(
        columns={"percent_sample_1": sample_1_name, "percent_sample_2": sample_2_name},
        inplace=True,
    )
    psi_plot = psi_plot_raw.melt(id_vars=[variable + "_bin_rng"]).rename(
        columns={"variable": "Sample", "value": "Percent"}
    )

    output = pd.concat(
        [output, pd.DataFrame(output.sum()).transpose()], axis=0
    ).reset_index(drop=True)
    output.iloc[-1, 0] = "total"

    plt.figure(figsize=figsize)
    ax = sns.barplot(x=variable + "_bin_rng", y="Percent", hue="Sample", data=psi_plot)
    ax.set(
        xlabel=None,
        ylabel="Proportion",
        title="Population stability index - "
        + variable
        + " = "
        + str(round(output.iloc[-1, -1], 4)),
    )
    plt.legend(loc=1)
    for item in ax.get_xticklabels():
        item.set_rotation(rotation)

    output.fillna(0, inplace=True)
    output = output.astype({"count_sample_1": "int", "count_sample_2": "int"})

    output.rename(
        columns={
            "count_sample_1": "count_" + sample_1_name,
            "percent_sample_1": "percent_" + sample_1_name,
        },
        inplace=True,
    )
    output.rename(
        columns={
            "count_sample_2": "count_" + sample_2_name,
            "percent_sample_2": "percent_" + sample_2_name,
        },
        inplace=True,
    )
    return output


def cluster_summary_categorical(
    sample, cluster, variables, binary=False, separate_pcts_counts=False
):
    K = sample[cluster].max() + 1
    output = pd.DataFrame()
    for k in range(K):
        k_sample = sample[sample[cluster] == k]
        k_output = pd.DataFrame()
        for variable in variables:
            single_output = pd.DataFrame(
                data={
                    "C"
                    + str(k): k_sample[variable].value_counts().sort_values().values,
                    "C"
                    + str(k)
                    + "-PCT": k_sample[variable].value_counts().sort_values().values
                    / k_sample[variable].value_counts().sum()
                    * 100,
                },
                index=pd.MultiIndex.from_tuples(
                    [
                        (variable, j)
                        for j in k_sample[variable].value_counts().sort_values().index
                    ],
                    names=["variable", "value"],
                ),
            )
            k_output = pd.concat([k_output, single_output], axis=0)
        output = pd.concat([output, k_output], axis=1)

    k_sample = sample
    k_output = pd.DataFrame()
    for variable in variables:
        single_output = pd.DataFrame(
            data={
                "SAMPLE": k_sample[variable].value_counts().sort_values().values,
                "SAMPLE"
                + "-PCT": k_sample[variable].value_counts().sort_values().values
                / k_sample[variable].value_counts().sum()
                * 100,
            },
            index=pd.MultiIndex.from_tuples(
                [
                    (variable, j)
                    for j in k_sample[variable].value_counts().sort_values().index
                ],
                names=["variable", "value"],
            ),
        )
        k_output = pd.concat([k_output, single_output], axis=0)
    output = pd.concat([output, k_output], axis=1)

    totals = pd.DataFrame(
        output.iloc[output.index.get_level_values("variable") == variables[0]].sum()
    ).T
    totals = totals.set_index(
        pd.MultiIndex.from_tuples([("Total", "Total")], names=("variable", "value"))
    )

    output = pd.concat([output, totals], axis=0)
    output.replace(np.nan, 0, inplace=True)
    output = output.astype({"C" + str(i): "int" for i in range(K)}).astype(
        {"SAMPLE": "int"}
    )

    if binary:
        output = output.iloc[output.index.get_level_values("value") != 0]
    if separate_pcts_counts:
        output = output[
            ["C" + str(i) for i in range(K)]
            + ["SAMPLE"]
            + ["C" + str(i) + "-PCT" for i in range(K)]
            + ["SAMPLE-PCT"]
        ]

    return output


def cluster_summary_continuous(sample, cluster, variables, median_only=False):
    K = sample[cluster].max() + 1
    output = pd.DataFrame()
    for variable in variables:
        single_output = pd.concat(
            [
                pd.Series(
                    sample[sample[cluster] == i][variable].describe(), name="C" + str(i)
                )
                for i in range(K)
            ],
            axis=1,
        )
        single_output.index = pd.MultiIndex.from_tuples(
            [(variable, j) for j in single_output.index], names=["variable", "value"]
        )
        single_output = single_output.iloc[[3, 4, 5, 1, 6, 7]]
        single_output.rename(
            index={"25%": "1Q", "75%": "3Q", "50%": "med", "mean": "avg"}, inplace=True
        )
        output = pd.concat([output, single_output], axis=0)

    if median_only:
        output = output.loc[output.index.get_level_values("value") == "avg"]

    return output


def cluster_summary(
    sample, cluster, continuous=None, categorical=None, delimiter="======"
):
    K = sample[cluster].max() + 2
    for k in range(K):
        k_output = pd.DataFrame({delimiter: ""}, index=range(6))
        k_output.columns = pd.MultiIndex.from_tuples([("", delimiter)])
        k_N = len(sample[sample[cluster] == k])
        if continuous:
            for variable in continuous:
                tuples = [
                    (variable, "statistic"),
                    (variable, "value"),
                    (" ", delimiter),
                ]
                if k == K - 1:
                    single_output = pd.DataFrame(
                        sample[sample[cluster] <= k][variable].describe()
                    )
                else:
                    single_output = pd.DataFrame(
                        sample[sample[cluster] == k][variable].describe()
                    )
                single_output.rename(
                    index={
                        "min": "MIN:",
                        "25%": "1Q:",
                        "75%": "3Q:",
                        "50%": "MEDIAN:",
                        "mean": "MEAN:",
                        "max": "MAX:",
                    },
                    inplace=True,
                )
                single_output = single_output.iloc[[3, 4, 5, 1, 6, 7]].reset_index()
                single_output.rename(
                    columns={"index": "statistic", variable: "value"}, inplace=True
                )
                single_output.loc[:, delimiter] = ""
                single_output.columns = pd.MultiIndex.from_tuples(tuples)
                k_output = pd.concat([k_output, single_output], axis=1)

        if categorical:
            for variable in categorical:
                tuples = [
                    (variable, "category"),
                    (variable, "count"),
                    (variable, "percent"),
                    (" ", delimiter),
                ]
                if k == K - 1:
                    value_counts = (
                        sample[sample[cluster] <= k][variable]
                        .value_counts()
                        .sort_values(ascending=False)
                    )
                else:
                    value_counts = (
                        sample[sample[cluster] == k][variable]
                        .value_counts()
                        .sort_values(ascending=False)
                    )
                single_output = pd.DataFrame(value_counts.head(5))
                other = pd.Series(
                    value_counts.sum() - value_counts.head(5).sum(),
                    name="Other",
                    index=[variable],
                )
                if other.sum() > 0:
                    single_output = single_output.append(other)
                single_output["percent"] = (
                    single_output.transform(lambda x: x / sum(x)) * 100
                )
                single_output = single_output.reset_index()
                single_output.loc[:, delimiter] = ""
                single_output.columns = pd.MultiIndex.from_tuples(tuples)
                k_output = pd.concat([k_output, single_output], axis=1)

        k_output.replace(np.nan, "", inplace=True)
        if k == K - 1:
            k_output.columns.names = ["GRAND_TOTAL", "N = " + str(len(sample))]
        else:
            k_output.columns.names = ["CLUSTER_ID:" + str(k), "N = " + str(k_N)]
        k_output
    return None


def pivot(sample, groupby, n=None):
    output = (
        sample.groupby(groupby)
        .agg(count=(sample.columns[0], "count"))
        .sort_values(by="count", ascending=False)
        .reset_index()
        .head(n)
    )
    output["pct"] = output["count"] / output["count"].sum()
    output = output.style.format(dict_fmt)
    return output


def get_major_locator(date_frequency, xticks_frequency):
    if date_frequency == 'month':
        major_locator = mdates.MonthLocator(interval=xticks_frequency)
    elif date_frequency == 'year':
        major_locator = mdates.YearLocator(base=xticks_frequency)
    elif date_frequency == 'week':
        major_locator = mdates.WeekdayLocator(interval=xticks_frequency)
    elif date_frequency == 'day':
        major_locator = mdates.DayLocator(interval=xticks_frequency)
    return major_locator


def time_series_decompose(
    y: pd.Series,
    period: int = 12,
    method: str = 'stl',
    model_type: str = 'additive',
    stl_trend: int = None,
    stl_seasonal: int = 7,
    stl_seasonal_deg: int = 1,
    stl_trend_deg: int = 1,
    stl_low_pass_deg: int = 1,
    title: str = None,
    xticks_frequency: int = 3,
    date_format: str = '%Y-%m',
    date_frequency: str = 'month',
    rotation: int = 45,
    figsize: tuple = (12, 18),
    return_type: str = 'both'
):
    """Decompose time series and plot respective trend, seasonal and remainder components.

    Arguments:
    y: Time series variable to be decomposed
    period: Number of observations in seasonal period (for monthly data = 12)
    method: Either 'stl' or 'classical'
    model_type: Either 'additive' or 'multiplicative'. Only applies to classic method
    figsize: Optional argument for larger displays
    xticks_frequency: Optional argument for frequency of xaxis grid
    date_format: Format of displayed dates on x axis
    rotation: Optional argument for xticks labels rotation
    return_type: 'plot', 'data' or 'both'
    """
    if method == 'stl':
        decomposed = tsa.STL(
            endog=y, period=period,
            trend=stl_trend,
            seasonal=stl_seasonal,
            seasonal_deg=stl_seasonal_deg,
            trend_deg=stl_trend_deg, low_pass_deg=stl_low_pass_deg
        ).fit()
        components = pd.DataFrame({'observed': decomposed.observed,
                                   'trend': decomposed.trend,
                                   'seasonal': decomposed.seasonal,
                                   'residual': decomposed.resid,
                                   'seasonally adjusted': (
                                       decomposed.observed - decomposed.seasonal
                                    )
                                   })
    elif method == 'classical':
        decomposed = pa.arima.decompose(x=y.to_numpy(), type_=model_type, m=period)
        is_additive = (model_type == 'additive')
        sa = decomposed[0] - decomposed[2] if is_additive else decomposed[0]/decomposed[2]
        components = pd.DataFrame({'observed': decomposed[0],
                                   'trend': decomposed[1],
                                   'seasonal': decomposed[2],
                                   'residual': decomposed[3],
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
        date_frequency=date_frequency, xticks_frequency=xticks_frequency))
    axs[4].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
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


def time_series_plot(
    y: pd.Series,
    title: str,
    figsize: tuple = (14, 10),
    rotation: int = 45,
    hist_bins: int = 10,
    date_format: str = '%Y-%m',
    date_frequency: str = 'month',
    xticks_frequency: int = 3
):
    """Plot lineplot and histogram for given time series for exploratory purposes.

    Arguments:
    y: Time series variable to be plotted
    title: Title of the plot
    column: Optional argument for plotted column name
    figsize: Optional argument for larger displays
    xticks_freq: Optional argument for frequency of xaxis grid
    hist_bins: Optional argument for number of bins in histrogram plot
    rotation: Optional argument for xticks labels rotation
    """

    _, axs = plt.subplots(2, 1, figsize=figsize)
    axs[0].plot(y)
    axs[0].tick_params(axis="x", rotation=rotation)
    axs[0].xaxis.set_major_locator(get_major_locator(
        date_frequency=date_frequency, xticks_frequency=xticks_frequency))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    axs[0].set_xlim(y.index.min(), y.index.max())
    axs[0].grid(True, alpha=0.2)
    axs[0].set_title(title)

    axs[1].hist(y, bins=hist_bins)
    axs[1].set_title(title)
    plt.subplots_adjust(hspace=0.3)
    return None


def time_series_plot_forecast(
    data: pd.DataFrame,
    title: str,
    columns: list = ["observed", "predicted_mean", "lower", "upper"],
    figsize: tuple = (16, 8),
    rotation: int = 45,
    date_format: str = '%Y-%m',
    date_frequency: str = 'month',
    xticks_frequency: int = 3,
    start_idx: int = 0
):
    """Plot time series forecasts.

    Arguments:
    data: Dataset with for columns representing observed, predicted, and confidence interval limits
    variables
    title: Title of the plot
    column: Optional argument for plotted column name
    figsize: Optional argument for larger displays
    xticks_freq: Optional argument for frequency of xaxis grid
    hist_bins: Optional argument for number of bins in histrogram plot
    rotation: Optional argument for xticks labels rotation
    """

    _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(data[columns[1]][start_idx:], label="prediction")
    ax.plot(data[columns[0]][start_idx:], marker='o', markerfacecolor='k', markeredgecolor='k',
            markersize=4, linewidth=0, label="transformed")
    ax.fill_between(
        data.index[start_idx:], data[columns[2]][start_idx:], data[columns[3]][start_idx:],
        alpha=0.2, label="prediction interval")
    ax.tick_params(axis="x", rotation=rotation)
    ax.xaxis.set_major_locator(get_major_locator(date_frequency=date_frequency,
                                                 xticks_frequency=xticks_frequency))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

    ax.set_xlim(data.index.min(), data.index.max())
    ax.grid(True, alpha=0.2)
    ax.set_title(title)
    ax.legend(loc="upper left")
    plt.subplots_adjust(hspace=0.3)
    return None


def time_series_stl_arimax_forecast(
    data,
    y,
    period,
    model=ARIMA,
    stl_trend=None,
    stl_seasonal=7,
    arima_order=(1, 0, 0),
    arima_trend=None,
    arima_exog=None,
    arima_seasonal_order=None,
    prediction_start=0,
    forecast_steps=None,
    forecast_exog=None,
    dynamic=None,
    alpha=0.05,
    nlags=10,
    df_adjust=True,
    auto_ylims=True,
    bartlett_confint=False,
    lmbda=None,
    plot_type='raw',
    plot_title='Forecast',
    plot_start_idx=0,
    plot_xticks_frequency=3,
    plot_date_format='%Y-%m',
    plot_date_frequency='month',
    ylims=(-0.3, 0.3)
):

    if forecast_steps is None:
        end = len(y) - 1
    else:
        end = forecast_steps + len(y) - 1

    model = tsa.STLForecast(endog=y,
                            model=model,
                            trend=stl_trend,
                            seasonal=stl_seasonal,
                            model_kwargs={
                                'order': arima_order,
                                'seasonal_order': arima_seasonal_order,
                                'exog': arima_exog,
                                'trend': arima_trend,
                                'freq': data.index.freq
                            },
                            period=period).fit()

    print(f'AICc = {model.model_result.aicc:,.4f} \n AIC = {model.model_result.aic:,.4f} \n MSE ='
          f'{model.model_result.mse:,.4f}')
    print(model.model_result.summary())

    model.model_result.plot_diagnostics(figsize=(14, 8), lags=nlags)
    # zero=True, marker='o', auto_ylims=auto_ylims, bartlett_confint=bartlett_confint)
    _, ax = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(y, lags=40, alpha=alpha, use_vlines=True, title='ACF',
             zero=True, vlines_kwargs=None, marker=None, ax=ax[0])
    plot_pacf(y, lags=40, alpha=alpha, use_vlines=True, title='PACF',
              zero=True, vlines_kwargs=None, marker=None, ax=ax[1])

    residual_acf = pd.DataFrame(
        {'residual_acf': tsa.acf(model.model_result.filter_results.standardized_forecasts_error[0],
                                 adjusted=False, nlags=20, qstat=True, alpha=alpha)[0][1:],
         'q_stat': model.model_result.test_serial_correlation(method='ljungbox', lags=20)[0][0],
         'p_value': model.model_result.test_serial_correlation(method='ljungbox', lags=20)[0][1]
         }, index=np.arange(1, nlags + 1)).T

    stl_forecast_model_prediction = model.get_prediction(
        start=prediction_start, end=end, exog=forecast_exog, dynamic=dynamic)

    sa = pd.Series(model.result.resid + model.result.trend, name=y.name + '_seasonal_adjusted')
    predicted = pd.Series(stl_forecast_model_prediction.predicted_mean,
                          name='predicted_' + y.name.split('_')[1])
    regression_residual = pd.Series(model.model_result.resid, name='regression_residual')
    arima_residual = pd.Series(list(
        model.model_result.filter_results.standardized_forecasts_error[0]), index=y.index,
        name='arima_residual'
    )
    conf_int = pd.DataFrame(stl_forecast_model_prediction.conf_int(alpha=alpha))
    calculated = pd.concat([sa, predicted, regression_residual, arima_residual, conf_int], axis=1)

    forecast = pd.merge(data, calculated, left_index=True, right_index=True, how='right')
    pi_col_names = ['pi_lower_' + y.name.split('_')[1], 'pi_upper_' + y.name.split('_')[1]]
    forecast.columns = list(forecast.columns[:-2]) + pi_col_names

    if lmbda is not None:
        forecast['predicted_raw'] = inv_boxcox(forecast['predicted_' + y.name.split('_')[1]], lmbda)
        forecast['pi_lower_raw'] = inv_boxcox(forecast['pi_lower_' + y.name.split('_')[1]], lmbda)
        forecast['pi_upper_raw'] = inv_boxcox(forecast['pi_upper_' + y.name.split('_')[1]], lmbda)

    time_series_plot_forecast(data=forecast,
                              title=plot_title,
                              start_idx=plot_start_idx,
                              xticks_frequency=plot_xticks_frequency,
                              columns=[
                                  'observed_' + plot_type,
                                  'predicted_' + plot_type,
                                  'pi_lower_' + plot_type,
                                  'pi_upper_' + plot_type
                              ],
                              date_format=plot_date_format)
    return forecast, residual_acf, model
