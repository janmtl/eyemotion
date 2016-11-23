"""Exposes useful functions for eyemotion analysis"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plotting import allplots, saveplots, PLOTS
import plotting
import os


SECRET_KEY = -1234567


def get_calibrated_BeGaze_events2(out):
    events = out['Eyemotion']['LDiameter'].loc["VAL", :, :].copy(deep=True)
    events["stat"] = events["stat"].astype(np.float)
    events["Bin_Index"] = events["Bin_Index"].astype(np.int)
    events.rename(columns={"stat": "pupil diameter [mm]"}, inplace=True)

    cal = events[events["Condition"] == "Fix"].copy(deep=True).reset_index(drop=True)
    events = events[events["Condition"] != "Fix"].reset_index(drop=True)
    cal = cal.pivot_table(index="Subject", columns="Label",
                          values="pupil diameter [mm]").\
        reset_index()

    events = pd.merge(events, cal, how="left", on="Subject")
    events["pupil response"] = events["pupil diameter [mm]"] \
        - events["White"]
    events["pupil response [%]"] = 100*events["pupil response"].\
        div(events["Black"] - events["White"], axis=0)
    events.drop(["Black", "White"], axis=1, inplace=True)

    events.loc[:, "Order"] = events.loc[:, "Order"].astype(np.float)
    sel = events["Subject"] > 199
    events.loc[sel, "Age"] = "OA"
    events.loc[~sel, "Age"] = "YA"
    return events


def get_calibrated_BeGaze_events(out, task_name):
    events = out[task_name]['LDiameter'].loc["VAL", :, :].copy(deep=True)
    events["stat"] = events["stat"].astype(np.float)
    events["Bin_Index"] = events["Bin_Index"].astype(np.int)
    events.rename(columns={"stat": "pupil diameter [mm]"}, inplace=True)

    # Calibration values
    cal = out['BeGazeCalibration']['LDiameter'].loc["VAL", :, :].copy(deep=True)
    cal["stat"] = cal["stat"].astype(np.float)
    cal = cal.pivot_table(index="Subject", columns="Label", values="stat").\
        reset_index()

    events = pd.merge(events, cal, how="left", on="Subject")
    events["pupil response"] = events["pupil diameter [mm]"] \
        - events["White"]
    events["pupil response [%]"] = 100*events["pupil response"].\
        div(events["Black"] - events["White"], axis=0)
    events.drop(["Black", "White"], axis=1, inplace=True)

    events.loc[:, "Order"] = events.loc[:, "Order"].astype(np.float)
    sel = events["Subject"] > 199
    events.loc[sel, "Age"] = "OA"
    events.loc[~sel, "Age"] = "YA"
    return events


def get_Biopac_events(out, channel, stat):
    events = out['Eyemotion'][channel].loc[stat, :, :].copy(deep=True)
    events["stat"] = events["stat"].astype(np.float)
    events.rename(columns={"stat": channel+" "+stat}, inplace=True)
    events.loc[:, ["ID", "Order"]] = events.loc[:, ["ID", "Order"]]\
        .astype(np.float)
    sel = events["Subject"] > 199
    events.loc[sel, "Age"] = "OA"
    events.loc[~sel, "Age"] = "YA"
    return events


def calibrate_Biopac_events(events, out, channel, stat):
    # Calibration values
    cal = out['BiopacCalibration'][channel].loc["VAL", :, :].copy(deep=True)
    cal["stat"] = cal["stat"].astype(np.float)
    cal = cal.pivot_table(index="Subject", columns="Label", values="stat").\
        reset_index()

    events = pd.merge(events, cal, how="left", on="Subject")
    events[channel+" response"] = events[channel+" "+stat]\
        .div(events["HeartrateFix"])
    events[channel+" response [%]"] = 100*events[channel+" response"]
    return events


def clean(events, channel, stat, outlier_threshold):
    col = channel+" "+stat

    groupcols = ["Age", "Condition", "Label", "Bin_Index"]
    allcols = ["Age", "Condition", "Label", "Bin_Index", col]
    output = events.copy(deep=True)
    means = output[allcols].groupby(groupcols).mean().reset_index()
    means.rename(columns={col: col+" mean"}, inplace=True)

    stds = output[allcols].groupby(groupcols).std().reset_index()
    stds.rename(columns={col: col+" std"}, inplace=True)

    output = pd.merge(output, means, how="left", on=groupcols)
    output = pd.merge(output, stds, how="left", on=groupcols)

    output["z-score"] = np.abs((output[col] - output[col+" mean"])
                               .div(output[col+" std"], axis=0))
    output["outlier?"] = (output["z-score"] > outlier_threshold)

    output.loc[output["outlier?"], col] = np.nan
    return output


def leadin_longstim(events, bin_width, fix_label_name):
    # Baseline the LongStim label to Fix2
    if (bin_width == 0.5):
        sel_event_fixes = (events["Label"] == fix_label_name) \
            & ((events["Bin_Index"] == 4) | (events["Bin_Index"] == 5))
    else:
        sel_event_fixes = (events["Label"] == fix_label_name) \
            & (events["Bin_Index"] == 2)

    event_fixes = events.loc[sel_event_fixes, ["Subject", "ID",
                                               "pupil response [%]"]].\
        copy(deep=True)
    event_fixes = event_fixes.groupby(["Subject", "ID"]).mean().reset_index()
    event_fixes.rename(columns={"pupil response [%]":
                                "pupil baseline [%]"}, inplace=True)

    output = pd.merge(events, event_fixes, how="left", on=["Subject", "ID"])
    output["baselined pupil response [%]"] = output["pupil response [%]"] \
        - output["pupil baseline [%]"]

    return output


def baseline(events, fixdict, col, on):
    event_fixes = []
    for ble, blr in fixdict.iteritems():
        sel_event_fixes = (events["Label"] == blr[0]) \
            & np.in1d(events["Bin_Index"], blr[1])
        event_fix = events.loc[sel_event_fixes, ["Subject", on, col]].\
            copy(deep=True)
        event_fix = event_fix.groupby(["Subject", on]).mean().reset_index()
        event_fix["Label"] = ble
        event_fix.rename(columns={col: "baseline"},
                         inplace=True)
        event_fixes.append(event_fix)

    event_fixes = pd.concat(event_fixes)
    output = pd.merge(events, event_fixes,
                      how="left", on=["Subject", on, "Label"])
    output.fillna(0, inplace=True)
    output["baselined "+col] = output[col] - output["baseline"]

    return output


def get_EPrime_events(out):
    ep_cols = ["ID", "Subject", "Condition", "stat"]
    ep1 = out['Eyemotion']['RateArousal1'].loc['VAL', :, ep_cols].\
        copy(deep=True)
    ep1.loc[:, "stat"] = ep1.loc[:, "stat"].astype(np.float)
    ep1.rename(columns={"stat": "RateArousal1"}, inplace=True)
    ep1.dropna(axis=0, how='all', inplace=True)

    ep2 = out['Eyemotion']['RateArousal2'].loc['VAL', :, ep_cols].\
        copy(deep=True)
    ep2.loc[:, "stat"] = ep2.loc[:, "stat"].astype(np.float)
    ep2.rename(columns={"stat": "RateArousal2"}, inplace=True)
    ep2.dropna(axis=0, how='all', inplace=True)

    ep3 = out['Eyemotion']['RateEffort'].loc['VAL', :, ep_cols].\
        copy(deep=True)
    ep3.loc[:, "stat"] = ep3.loc[:, "stat"].astype(np.float)
    ep3.rename(columns={"stat": "RateEffort"}, inplace=True)
    ep3.dropna(axis=0, how='all', inplace=True)

    ep = pd.merge(ep1, ep2, how="inner", on=["ID", "Subject", "Condition"])
    output = pd.merge(ep, ep3, how="inner", on=["ID", "Subject", "Condition"])
    sel = output["Subject"] > 199
    output.loc[sel, "Age"] = "OA"
    output.loc[~sel, "Age"] = "YA"
    return output


def attach_EPrime(events, out, on):
    # Attach the `EPrime` variables.
    ep_cols = ["ID", "Order", "Subject", "stat"]
    ep1 = out['Eyemotion']['RateArousal1'].loc['VAL', :, ep_cols].\
        copy(deep=True)
    ep1.loc[:, "stat"] = ep1.loc[:, "stat"].astype(np.float)
    ep1.rename(columns={"stat": "RateArousal1"}, inplace=True)

    ep2 = out['Eyemotion']['RateArousal2'].loc['VAL', :, ep_cols].\
        copy(deep=True)
    ep2.loc[:, "stat"] = ep2.loc[:, "stat"].astype(np.float)
    ep2.rename(columns={"stat": "RateArousal2"}, inplace=True)

    ep3 = out['Eyemotion']['RateEffort'].loc['VAL', :, ep_cols].\
        copy(deep=True)
    ep3.loc[:, "stat"] = ep3.loc[:, "stat"].astype(np.float)
    ep3.rename(columns={"stat": "RateEffort"}, inplace=True)

    ep = pd.merge(ep1, ep2, how="inner", on=["ID", "Order", "Subject"])
    ep = pd.merge(ep, ep3, how="inner", on=["ID", "Order", "Subject"])
    ep["Subject"] = ep["Subject"].astype(np.int)

    output = events.copy(deep=True)
    if on == "Order":
        output.drop("ID", axis=1, inplace=True)
    else:
        output.drop("Order", axis=1, inplace=True)

    output = pd.merge(output, ep, how="left", on=["Subject", on])
    return output


def attach_iaps(events, path):
    # Import the image valence and arousal values.
    iaps = pd.read_csv(path,
                       usecols=["IAPS", "val_y", "val_o", "val_t",
                                "aro_y", "aro_o", "aro_t"])
    output = pd.merge(events, iaps, how="left", left_on="ID", right_on="IAPS")
    output.drop("IAPS", axis=1, inplace=True)
    return output


def attach_counterbalance(events, path):
    # Import the counterbalance data.
    oa = pd.read_excel(path, sheetname="OA")
    ya = pd.read_excel(path, sheetname="YA")
    cols = ["Subj ID", "CB Cond.", "PPT", "Run Date", "Run Time"]
    oa = oa[cols]
    ya = ya[cols]
    oa.rename(columns={"Subj ID": "Subject"}, inplace=True)
    ya.rename(columns={"Subj ID": "Subject"}, inplace=True)
    output = pd.merge(events, pd.concat([oa, ya]), how="left", on="Subject")
    return output


def saveformats(events, dirpath, channel, stat, basename, valuecolumn):
    col = channel + "-" + stat
    lf_name = basename + "-" + col + "-Longformat.csv"
    # sf_name = "Shortformat-"+basename+".csv"
    ssf_name = basename + "-" + col + "-Shorterformat.csv"

    # Longformat
    longformat = events.copy(deep=True)
    longformat.to_csv(os.path.join(dirpath, lf_name))

    # Shortformat
    # shortformat = longformat.pivot_table(index=["Subject", "ID"],
    #                                      columns=["Condition", "Label",
    #                                               "Bin_Index"],
    #                                      values=valuecolumn)
    # shortformat.columns = [col[0]+"_"+col[1]+"_"+str(col[2])
    #                        for col in shortformat.columns.values]
    # shortformat.reset_index(inplace=True)
    # sf = longformat.drop(["Bin_Index", valuecolumn, "Condition", "Label"],
    #                      axis=1).drop_duplicates()
    # shortformat = pd.merge(sf, shortformat, how="left", on=["Subject", "ID"])
    # shortformat.to_csv(os.path.join(dirpath, sf_name))

    # Shorterformat
    shorterformat = longformat.pivot_table(index=["Subject"],
                                           columns=["Condition", "Label",
                                                    "Bin_Index"],
                                           values=valuecolumn)
    shorterformat.columns = [column[0]+"_"+column[1]+"_"+str(column[2])
                             for column in shorterformat.columns.values]
    shorterformat.reset_index(inplace=True)
    shorterformat.to_csv(os.path.join(dirpath, ssf_name))

    return longformat, shorterformat


def EPrime_acplot(events, col, basename, withID=True):
    if withID:
        g = events.copy(deep=True)
    else:
        g = events.loc[:, ["Subject", "Age", "Condition", col]].\
            groupby(["Subject", "Age", "Condition"]).mean().reset_index()
    conditions = ['Distract', 'Rethink', 'Attend']
    v = g.loc[:, ["Age", "Condition", col]].\
        groupby(["Age", "Condition"]).mean()
    s = g.loc[:, ["Age", "Condition", col]].\
        groupby(["Age", "Condition"]).sem()
    ages = ["OA", "YA"]

    return _acbar(v, s, ages, conditions, col,
                  PLOTS["AgexConditionxTime"][basename])


def _acbar(v, s, ages, conditions, stat, PAL):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    crects = {}
    agap = (len(conditions)+0.5)
    for ageidx, age in enumerate(ages):
        for condix, condition in enumerate(conditions):
            vp = v.loc[(age, condition), :].values
            sp = s.loc[(age, condition), :].values
            crects[condition] = ax.bar(condix + ageidx*agap,
                                       vp, 1, yerr=sp, label=condition,
                                       color=PAL[age][condition])

    axes = ax.axis()
    axes = (axes[0], axes[1], axes[2], axes[3]*1.25)
    ax.axis(axes)
    ax.set_xticks(np.arange(len(ages))*agap + agap/2.0)
    ax.set_xticklabels(tuple(ages))
    ax.set_title(stat)
    ax.legend(crects.values(), crects.keys(), frameon=True)

    return fig, ax


def dfgridplot(data, x, y, plotfn, aggfuncs,
               rownames=None, colnames=None, linenames=None,
               bin_width=0.5,
               **kwargs):

    aggsindex = None
    if rownames:
        if type(rownames) is str:
            rownames = list(rownames)
        rows = data[rownames].drop_duplicates()
        aggsindex = rownames
    else:
        rownames = []
        rows = [None]

    if colnames:
        if type(colnames) is str:
            colnames = list(colnames)
        cols = data[colnames].drop_duplicates()
        if aggsindex:
            aggsindex = aggsindex + colnames
        else:
            aggsindex = colnames
    else:
        colnames = []
        cols = [None]

    if linenames:
        if type(linenames) is str:
            linenames = list(linenames)
        if aggsindex:
            aggsindex = aggsindex + linenames
        else:
            aggsindex = linenames
    else:
        linenames = []

    aggs = {}
    for key, func in aggfuncs.iteritems():
        aggs[key] = data.pivot_table(index=aggsindex,
                                     columns=x, values=y, aggfunc=func)
        aggs[key].columns = aggs[key].columns.droplevel(0)

    fig, axs = plt.subplots(len(rows), len(cols), **kwargs)
    # when we want to go row-wise in plt.subplots it still produce a
    # 1-dim array so we have to artifically make it 2-dim
    if len(rows) == 1:
        axsv = [axs]
    else:
        axsv = axs
    axsdf = pd.DataFrame(axsv, index=rows, columns=cols)

    for rowname, axrow in axsdf.iteritems():
        for colname, ax in axrow.iteritems():
            if not rowname:
                rowname = ()
            if not colname:
                colname = ()
            datasel = colname + rowname + (slice(None),)*len(linenames)
            plotdata = {}
            for key, aggdata in aggs.iteritems():
                plotdata[key] = aggdata.loc[datasel, :]
                if type(plotdata[key]) is pd.DataFrame:
                    plotdata[key].index = plotdata[key]\
                        .index.droplevel(range(len(rownames) + len(colnames)))
            plotfn(ax, plotdata, rowname, colname, bin_width, **kwargs)

    return fig, axs


def plt_eyemotion(ax, plotdata, rowname, colname, bin_width, **kwargs):
    # PAL = sns.color_palette()
    PAL = [(0.0, 0.0, 0.0), (0.35, 0.35, 0.35), (0.5, 0.5, 0.5)]
    MARKERS = ["o", "s", "d"]
    if type(plotdata['mean']) is pd.DataFrame:
        means = plotdata['mean'].sortlevel(0, axis=1)
        sems = plotdata['sem'].sortlevel(0, axis=1)

        ax.set_title(str(rowname)+" "+str(colname))
        for idx, (linename, linedata) in enumerate(means.iterrows()):
            y = linedata.values
            s = sems.loc[linename, :].values
            t = np.linspace(0.0, bin_width*len(linedata), len(linedata))
            ax.plot(t, y, label=linename, c=PAL[idx],
                    marker=MARKERS[idx], markersize=5.0)
            ax.fill_between(t, y-s, y+s,
                            color=(0.5, 0.5, 0.5), alpha=0.35)

    else:
        means = plotdata['mean'].values
        sems = plotdata['sem'].values
        t = np.linspace(0.0, 0.5*len(means), len(means))
        ax.set_title(str(rowname)+" "+str(colname))
        ax.plot(t, means)
        ax.fill_between(t, means-sems, means+sems, alpha=0.4)
