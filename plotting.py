"""Plotting functions for eyemotion"""

from core import _acplot, _cplot, _aplot, _plot
import numpy as np
import pandas as pd

PAL = {
    "Eyemotion": {
        "AgexConditionxTime": {
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
        "ConditionxTime": {
            "Attend": "#e41a1c",
            "Rethink": "#377eb8",
            "Distract": "#4daf4a"
        },
        "AgexTime": {
            "OA": "#984ea3",
            "YA": "#ff7f00"
        },
        "Time": "#377eb8"
    },
    "Slideshow": {
        "AgexConditionxTime": {
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
        },
        "ConditionxTime": {
            "neg": "#e41a1c",
            "neutral": "#377eb8",
            "pos": "#4daf4a"
        },
        "AgexTime": {
            "OA": "#984ea3",
            "YA": "#ff7f00"
        },
        "Time": "#377eb8"
    },
}

def allplots(events, labels, channel, stat, bin_width):
    acfig, acax = e.acplot(events, labels, stat, bin_width, e.Eyemotion_ACPAL, "Order", False)
    afig, aax = e.aplot(events, labels, stat, bin_width, e.APAL, "Order", False)
    cfig, cax = e.cplot(events, labels, stat, bin_width, e.Eyemotion_CPAL, "Order", False)
    fig, ax = e.plot(events, labels, stat, bin_width, e._PAL, "Order", False)



    acfig, acax = e.acplot(events, labels, stat, bin_width, e.Eyemotion_ACPAL, "Order", True)
    afig, aax = e.aplot(events, labels, stat, bin_width, e.APAL, "Order", True)
    cfig, cax = e.cplot(events, labels, stat, bin_width, e.Eyemotion_CPAL, "Order", True)
    fig, ax = e.plot(events, labels, stat, bin_width, e._PAL, "Order", True)


def saveplots(plots, dirpath, basename):
    acfig.savefig("outputs/EyemotionHRV-"+channel+"-AgexConditionxTime.pdf")
    afig.savefig("outputs/EyemotionHRV-"+channel+"-AgexTime.pdf")
    cfig.savefig("outputs/EyemotionHRV-"+channel+"-ConditionxTime.pdf")
    fig.savefig("outputs/EyemotionHRV-"+channel+"-Time.pdf")

    acfig.savefig("outputs/EyemotionHRV-"+channel+"-AgexConditionxTime_withID.pdf")
    afig.savefig("outputs/EyemotionHRV-"+channel+"-AgexTime_withID.pdf")
    cfig.savefig("outputs/EyemotionHRV-"+channel+"-ConditionxTime_withID.pdf")
    fig.savefig("outputs/EyemotionHRV-"+channel+"-Time_withID.pdf")


def EPrime_acplot(events, stat):
    g = events.copy(deep=True)
    conditions = ['Distract', 'Rethink', 'Attend']
    v = g.loc[:, ["Age", "Condition", stat]].\
        groupby(["Age", "Condition"]).mean()
    s = g.loc[:, ["Age", "Condition", stat]].\
        groupby(["Age", "Condition"]).sem()
    ages = ["OA", "YA"]

    return _acbar(v, s, ages, conditions, stat)

def acplot(events, labels, stat, bin_width, PAL, on, withID):
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

    return _acplot(v, s, labels, conditions, ages, stat, bin_width, PAL)

def cplot(events, labels, stat, bin_width, PAL, on, withID):
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

    return _cplot(v, s, labels, conditions, stat, bin_width, PAL)


def aplot(events, labels, stat, bin_width, PAL, on, withID):
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

    return _aplot(v, s, labels, ages, stat, bin_width, PAL)

def plot(events, labels, stat, bin_width, PAL, on, withID):
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

    return _plot(v, s, labels, stat, bin_width, PAL)
