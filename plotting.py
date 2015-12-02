"""Plotting functions for eyemotion"""

from core import _acplot, _cplot, _aplot, _plot
from os.path import join
import numpy as np
import pandas as pd


def acplot(events, labels, stat, bin_width, PAL, on, withID,
           plot_error_bars=True):
    sel = np.in1d(events["Label"], labels)
    g = events.loc[sel, :].copy(deep=True)
    conditions = g.loc[:, "Condition"].unique()
    if withID:
        g = g.pivot_table(index=["Subject", on, "Condition", "Age"],
                          columns=["Label", "Bin_Index"],
                          values=stat).reset_index()
        g.drop([("Subject", ""), (on, "")], axis=1, inplace=True)
        ages = ["OA", "YA"]
    else:
        g = g.pivot_table(index=["Subject", "Condition", "Age"],
                          columns=["Label", "Bin_Index"],
                          values=stat).reset_index()
        g.drop([("Subject", "")], axis=1, inplace=True)
        ages = ["OA", "YA"]

    v = g.groupby([("Age", ""), ("Condition", "")]).mean()
    s = g.groupby([("Age", ""), ("Condition", "")]).sem()

    v = v.sort_index(axis=1, level=["Label", "Bin_Index"])
    s = s.sort_index(axis=1, level=["Label", "Bin_Index"])

    return _acplot(v, s, labels, conditions, ages, stat, bin_width, PAL,
                   plot_error_bars)

def cplot(events, labels, stat, bin_width, PAL, on, withID,
          plot_error_bars=True):
    sel = np.in1d(events["Label"], labels)
    g = events.loc[sel, :].copy(deep=True)
    conditions = g.loc[:, "Condition"].unique()
    if withID:
        g = g.pivot_table(index=["Subject", on, "Condition", "Age"],
                          columns=["Label", "Bin_Index"],
                          values=stat).reset_index()
        g.drop([("Subject", ""), (on, ""), ("Age", "")], axis=1, inplace=True)
    else:
        g = g.pivot_table(index=["Subject", "Condition", "Age"],
                          columns=["Label", "Bin_Index"],
                          values=stat).reset_index()
        g.drop([("Subject", ""), ("Age", "")], axis=1, inplace=True)

    v = g.groupby([("Condition", "")]).mean()
    s = g.groupby([("Condition", "")]).sem()

    v = v.sort_index(axis=1, level=["Label", "Bin_Index"])
    s = s.sort_index(axis=1, level=["Label", "Bin_Index"])

    return _cplot(v, s, labels, conditions, stat, bin_width, PAL,
                  plot_error_bars)


def aplot(events, labels, stat, bin_width, PAL, on, withID,
          plot_error_bars=True):
    sel = np.in1d(events["Label"], labels)
    g = events.loc[sel, :].copy(deep=True)
    if withID:
        g = g.pivot_table(index=["Subject", on, "Condition", "Age"],
                          columns=["Label", "Bin_Index"],
                          values=stat).reset_index()
        g.drop([("Subject", ""), (on, ""), ("Condition", "")],
               axis=1, inplace=True)
    else:
        g = g.pivot_table(index=["Subject", "Condition", "Age"],
                          columns=["Label", "Bin_Index"],
                          values=stat).reset_index()
        g.drop([("Subject", ""), ("Condition", "")], axis=1, inplace=True)
    ages = ["OA", "YA"]

    v = g.groupby([("Age", "")]).mean()
    s = g.groupby([("Age", "")]).sem()

    v = v.sort_index(axis=1, level=["Label", "Bin_Index"])
    s = s.sort_index(axis=1, level=["Label", "Bin_Index"])

    return _aplot(v, s, labels, ages, stat, bin_width, PAL, plot_error_bars)

def plot(events, labels, stat, bin_width, PAL, on, withID,
         plot_error_bars=True):
    sel = np.in1d(events["Label"], labels)
    g = events.loc[sel, :].copy(deep=True)
    if withID:
        g = g.pivot_table(index=["Subject", on, "Condition", "Age"],
                          columns=["Label", "Bin_Index"],
                          values=stat).reset_index()
        g.drop([("Subject", ""), (on, ""), ("Age", ""), ("Condition", "")],
               axis=1, inplace=True)
    else:
        g = g.pivot_table(index=["Subject", "Condition", "Age"],
                          columns=["Label", "Bin_Index"],
                          values=stat).reset_index()
        g.drop([("Subject", ""), ("Age", ""), ("Condition", "")],
               axis=1, inplace=True)

    v = pd.DataFrame(g.mean())
    s = pd.DataFrame(g.sem())

    v = v.sort_index(level=["Label", "Bin_Index"])
    s = s.sort_index(level=["Label", "Bin_Index"])

    return _plot(v, s, labels, stat, bin_width, PAL, plot_error_bars)

PLOTS = {
    "AgexConditionxTime": {
        "func": acplot,
        "EyemotionHRV": {
            "OA": {
                "Attend": "#CB0103",
                "Rethink": "#1E659F",
                "Distract": "#349631"
            },
            "YA": {
                "Attend": "#FE3436",
                "Rethink": "#5198D2",
                "Distract": "#67C964"
            }
        },
        "Eyemotion": {
            "OA": {
                "Attend": "#CB0103",
                "Rethink": "#1E659F",
                "Distract": "#349631"
            },
            "YA": {
                "Attend": "#FE3436",
                "Rethink": "#5198D2",
                "Distract": "#67C964"
            }
        },
        "Slideshow": {
            "OA": {
                "neg": "#CB0103",
                "neutral": "#1E659F",
                "pos": "#349631"
            },
            "YA": {
                "neg": "#FE3436",
                "neutral": "#5198D2",
                "pos": "#67C964"
            }
        }
    },
    "ConditionxTime": {
        "func": cplot,
        "EyemotionHRV": {
            "Attend": "#e41a1c",
            "Rethink": "#377eb8",
            "Distract": "#4daf4a"
        },
        "Eyemotion": {
            "Attend": "#e41a1c",
            "Rethink": "#377eb8",
            "Distract": "#4daf4a"
        },
        "Slideshow": {
            "neg": "#e41a1c",
            "neutral": "#377eb8",
            "pos": "#4daf4a"
        }
    },
    "AgexTime": {
        "func": aplot,
        "EyemotionHRV": {
            "OA": "#984ea3",
            "YA": "#ff7f00"
        },
        "Eyemotion": {
            "OA": "#984ea3",
            "YA": "#ff7f00"
        },
        "Slideshow": {
            "OA": "#984ea3",
            "YA": "#ff7f00"
        }
    },
    "Time": {
        "func": plot,
        "EyemotionHRV": "#377eb8",
        "Eyemotion": "#377eb8",
        "Slideshow": "#377eb8"
    }
}


def allplots(events, labels, col, bin_width, basename, on,
             plot_error_bars=True):
    figs = {}
    for plotname, P in PLOTS.iteritems():
        figs[plotname] = {}
        figs[plotname]["withoutIDSEM"], _ = P["func"](events, labels, col,
            bin_width, P[basename], on, False, plot_error_bars=plot_error_bars)
        figs[plotname]["withIDSEM"], _ = P["func"](events, labels, col,
            bin_width, P[basename], on, True, plot_error_bars=plot_error_bars)
    return figs


def saveplots(figs, dirpath, channel, stat, basename):
    for plotname, fig in figs.iteritems():
        fn = basename + "-" + channel + "-" + stat + "-" + plotname
        fig["withoutIDSEM"].savefig(join(dirpath, fn+".pdf"))
        fig["withIDSEM"].savefig(join(dirpath, fn+"_withID.pdf"))
