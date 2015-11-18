"""Exposes useful functions for eyemotion analysis"""

import numpy as np
import pandas as pd
import seaborn as sns
from math import isnan
from matplotlib import pyplot as plt
import os


SECRET_KEY = -1234567
_PAL = "#377eb8"
Eyemotion_CPAL = {
    "Attend": "#e41a1c",
    "Rethink": "#377eb8",
    "Distract": "#4daf4a"
}
Slideshow_CPAL = {
    "neg": "#e41a1c",
    "neutral": "#377eb8",
    "pos": "#4daf4a"
}
APAL = {
    "OA": "#984ea3",
    "YA": "#ff7f00"
}
Eyemotion_ACPAL = {
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
}

Slideshow_ACPAL = {
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

    events.loc[:, "ID"] = events.loc[:, "ID"].astype(np.float)

    sel = events["Subject"] > 199
    events.loc[sel, "Age"] = "OA"
    events.loc[~sel, "Age"] = "YA"

    return events


def get_cleaned_Biopac_events(out, channel, stat, outlier_threshold):
    events = out['Eyemotion'][channel].loc[stat, :, :]
    events["stat"] = events["stat"].astype(np.float)
    events = events.pivot_table(index=["Subject", "Condition", "Order"],
                                columns=["Label", "Bin_Index"],
                                values="stat").reset_index()
    sel = events.loc[:, ("Subject", "")] > 199
    events.loc[sel, ("Age", "")] = "OA"
    events.loc[~sel, ("Age", "")] = "YA"
    events = events.drop([("Subject", ""), ("Order", "")], axis=1)

    grouped = events.groupby([("Age", ""), ("Condition", "")])
    for idx, group in grouped:
        group = group.drop([("Age", ""), ("Condition", "")], axis=1)
        group_std = np.nanstd(group.as_matrix(), axis=(0, 1))
        group_means = group.mean()
        group_deltas = np.abs(group.sub(group_means, axis="columns") /
                              group_std)
        group[(group_deltas > outlier_threshold)] = SECRET_KEY
        events.update(group)

    events[(events == SECRET_KEY)] = np.nan
    numeric_cols = events.columns.drop([("Age", ""), ("Condition", "")])
    events.loc[:, numeric_cols] = events.loc[:, numeric_cols].astype(np.float)
    return events


def leadin_longstim(events, bin_width, fix_label_name):
    # Baseline the LongStim label to Fix2
    if (bin_width == 0.5):
        sel_event_fixes = (events["Label"] == fix_label_name) \
                          & ((events["Bin_Index"] == 4) \
                             | (events["Bin_Index"] == 5))
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


def baseline(events, fixdict):
    event_fixes = []
    for ble, blr in fixdict.iteritems():
        sel_event_fixes = (events["Label"] == blr[0]) \
            & np.in1d(events["Bin_Index"], blr[1])
        event_fix = events.loc[sel_event_fixes,
                               ["Subject", "ID", "pupil response [%]"]].\
            copy(deep=True)
        event_fix = event_fix.groupby(["Subject", "ID"]).mean().reset_index()
        event_fix["Label"] = ble
        event_fix.rename(columns={"pupil response [%]": "pupil baseline [%]"},
                         inplace=True)
        event_fixes.append(event_fix)

    event_fixes = pd.concat(event_fixes)

    output = pd.merge(events, event_fixes,
                      how="left", on=["Subject", "ID", "Label"])
    output.fillna(0, inplace=True)
    output["baselined pupil response [%]"] = output["pupil response [%]"] \
        - output["pupil baseline [%]"]

    return output


def baseline_BeGaze_events(events, baseline_label_name):
    # Baseline the LongStim label to Fix2
    sel_event_fixes = (events["Label"] == baseline_label_name)
    event_fixes = events.loc[sel_event_fixes, ["Subject", "ID",
                                               "pupil response [%]"]].\
        copy(deep=True)
    event_fixes = event_fixes.groupby(["Subject", "ID"]).mean().reset_index()
    event_fixes.rename(columns={"pupil response [%]":
                                "pupil baseline [%]"}, inplace=True)

    sel_events = (events["Label"] != baseline_label_name)
    output = events.loc[sel_events, :]
    output = pd.merge(output, event_fixes, how="left", on=["Subject", "ID"])
    output["baselined pupil response [%]"] = output["pupil response [%]"] \
        - output["pupil baseline [%]"]

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
    events = pd.merge(ep, ep3, how="inner", on=["ID", "Subject", "Condition"])
    sel = events["Subject"] > 199
    events.loc[sel, "Age"] = "OA"
    events.loc[~sel, "Age"] = "YA"
    return events


def attach_EPrime(events, out):
    # Attach the `EPrime` variables.
    ep_cols = ["ID", "Subject", "stat"]
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

    ep = pd.merge(ep1, ep2, how="inner", on=["ID", "Subject"])
    ep = pd.merge(ep, ep3, how="inner", on=["ID", "Subject"])
    ep["ID"] = ep["ID"].astype(np.float)
    ep["Subject"] = ep["Subject"].astype(np.int)

    output = pd.merge(events, ep, how="left", on=["Subject", "ID"])
    return output


def attach_iaps(events, path):
    # Import the image valence and arousal values.
    iaps = pd.read_csv(path,
                       usecols=["IAPS", "val_y", "val_o", "val_t",
                                "aro_y", "aro_o", "aro_t"])
    output = pd.merge(events, iaps, how="left", left_on="ID", right_on="IAPS")
    output.drop("IAPS", axis=1, inplace=True)
    return output


def saveformats(events, dirpath, basename, valuecolumn):
    lf_name = basename + "-Longformat.csv"
    # sf_name = "Shortformat-"+basename+".csv"
    ssf_name = basename + "-Shorterformat.csv"

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
    shorterformat.columns = [col[0]+"_"+col[1]+"_"+str(col[2])
                             for col in shorterformat.columns.values]
    shorterformat.reset_index(inplace=True)
    shorterformat.to_csv(os.path.join(dirpath, ssf_name))

    return longformat, shorterformat


def EPrime_acplot(events, stat):
    g = events.copy(deep=True)
    conditions = ['Distract', 'Rethink', 'Attend']
    v = g.loc[:, ["Age", "Condition", stat]].\
        groupby(["Age", "Condition"]).mean()
    s = g.loc[:, ["Age", "Condition", stat]].\
        groupby(["Age", "Condition"]).sem()
    ages = ["OA", "YA"]

    return acbar(v, s, ages, conditions, stat)


def BeGaze_aplot_withID(events, labels, stat, bin_width, PAL):
    sel = np.in1d(events["Label"], labels)
    g = events.loc[sel, :].copy(deep=True)
    g = g.pivot_table(index=["Subject", "ID", "Condition", "Age"],
                      columns=["Label", "Bin_Index"],
                      values=stat).reset_index()
    g.drop([("Subject", ""), ("ID", ""), ("Condition", "")], axis=1, inplace=True)
    ages = ["OA", "YA"]

    v = g.groupby([("Age", "")]).mean()
    s = g.groupby([("Age", "")]).sem()

    v = v.sort_index(axis=1, level=["Label", "Bin_Index"])
    s = s.sort_index(axis=1, level=["Label", "Bin_Index"])

    return aplot(v, s, labels, ages, stat, bin_width, PAL)


def BeGaze_aplot(events, labels, stat, bin_width, PAL):
    sel = np.in1d(events["Label"], labels)
    g = events.loc[sel, :].copy(deep=True)
    g = g.pivot_table(index=["Subject", "Condition", "Age"],
                      columns=["Label", "Bin_Index"],
                      values=stat).reset_index()
    g.drop([("Subject", ""), ("Condition", "")], axis=1, inplace=True)
    ages = ["OA", "YA"]

    v = g.groupby([("Age", "")]).mean()
    s = g.groupby([("Age", "")]).sem()

    v = v.sort_index(axis=1, level=["Label", "Bin_Index"])
    s = s.sort_index(axis=1, level=["Label", "Bin_Index"])

    return aplot(v, s, labels, ages, stat, bin_width, PAL)


def BeGaze_cplot_withID(events, labels, stat, bin_width, PAL):
    sel = np.in1d(events["Label"], labels)
    g = events.loc[sel, :].copy(deep=True)
    conditions = g.loc[:, "Condition"].unique()
    g = g.pivot_table(index=["Subject", "ID", "Condition", "Age"],
                      columns=["Label", "Bin_Index"],
                      values=stat).reset_index()
    g.drop([("Subject", ""), ("ID", ""), ("Age", "")], axis=1, inplace=True)

    v = g.groupby([("Condition", "")]).mean()
    s = g.groupby([("Condition", "")]).sem()

    v = v.sort_index(axis=1, level=["Label", "Bin_Index"])
    s = s.sort_index(axis=1, level=["Label", "Bin_Index"])

    return cplot(v, s, labels, conditions, stat, bin_width, PAL)


def BeGaze_cplot(events, labels, stat, bin_width, PAL):
    sel = np.in1d(events["Label"], labels)
    g = events.loc[sel, :].copy(deep=True)
    conditions = g.loc[:, "Condition"].unique()
    g = g.pivot_table(index=["Subject", "Condition", "Age"],
                      columns=["Label", "Bin_Index"],
                      values=stat).reset_index()
    g.drop([("Subject", ""), ("Age", "")], axis=1, inplace=True)

    v = g.groupby([("Condition", "")]).mean()
    s = g.groupby([("Condition", "")]).sem()

    v = v.sort_index(axis=1, level=["Label", "Bin_Index"])
    s = s.sort_index(axis=1, level=["Label", "Bin_Index"])

    return cplot(v, s, labels, conditions, stat, bin_width, PAL)


def BeGaze_plot_withID(events, labels, stat, bin_width, PAL):
    sel = np.in1d(events["Label"], labels)
    g = events.loc[sel, :].copy(deep=True)
    g = g.pivot_table(index=["Subject", "ID", "Condition", "Age"],
                      columns=["Label", "Bin_Index"],
                      values=stat).reset_index()
    g.drop([("Subject", ""), ("ID", ""), ("Age", ""), ("Condition", "")], axis=1, inplace=True)

    v = pd.DataFrame(g.mean())
    s = pd.DataFrame(g.sem())

    v = v.sort_index(level=["Label", "Bin_Index"])
    s = s.sort_index(level=["Label", "Bin_Index"])

    return _plot(v, s, labels, stat, bin_width, PAL)


def BeGaze_plot(events, labels, stat, bin_width, PAL):
    sel = np.in1d(events["Label"], labels)
    g = events.loc[sel, :].copy(deep=True)
    g = g.pivot_table(index=["Subject", "Condition", "Age"],
                      columns=["Label", "Bin_Index"],
                      values=stat).reset_index()
    g.drop([("Subject", ""), ("Age", ""), ("Condition", "")], axis=1, inplace=True)

    v = pd.DataFrame(g.mean())
    s = pd.DataFrame(g.sem())

    v = v.sort_index(level=["Label", "Bin_Index"])
    s = s.sort_index(level=["Label", "Bin_Index"])

    return _plot(v, s, labels, stat, bin_width, PAL)


def BeGaze_acplot_withID(events, labels, stat, bin_width, PAL):
    sel = np.in1d(events["Label"], labels)
    g = events.loc[sel, :].copy(deep=True)
    conditions = g.loc[:, "Condition"].unique()
    g = g.pivot_table(index=["Subject", "ID", "Condition", "Age"],
                      columns=["Label", "Bin_Index"],
                      values=stat).reset_index()
    g.drop([("Subject", ""), ("ID", "")], axis=1, inplace=True)
    ages = ["OA", "YA"]

    v = g.groupby([("Age", ""), ("Condition", "")]).mean()
    s = g.groupby([("Age", ""), ("Condition", "")]).sem()

    v = v.sort_index(axis=1, level=["Label", "Bin_Index"])
    s = s.sort_index(axis=1, level=["Label", "Bin_Index"])

    return acplot(v, s, labels, conditions, ages, stat, bin_width, PAL)


def BeGaze_acplot(events, labels, stat, bin_width, PAL):
    sel = np.in1d(events["Label"], labels)
    g = events.loc[sel, :].copy(deep=True)
    conditions = g.loc[:, "Condition"].unique()
    g = g.pivot_table(index=["Subject", "Condition", "Age"],
                      columns=["Label", "Bin_Index"],
                      values=stat).reset_index()
    g.drop([("Subject", "")], axis=1, inplace=True)
    ages = ["OA", "YA"]

    v = g.groupby([("Age", ""), ("Condition", "")]).mean()
    s = g.groupby([("Age", ""), ("Condition", "")]).sem()

    v = v.sort_index(axis=1, level=["Label", "Bin_Index"])
    s = s.sort_index(axis=1, level=["Label", "Bin_Index"])

    return acplot(v, s, labels, conditions, ages, stat, bin_width, PAL)


def Biopac_acplot(events, PAL):
    v = events.groupby([("Age", ""), ("Condition", "")]).mean()
    s = events.groupby([("Age", ""), ("Condition", "")]).sem()

    v = v.sort_index(axis=1, level=["Label", "Bin_Index"])
    s = s.sort_index(axis=1, level=["Label", "Bin_Index"])

    conditions = ["Distract", "Rethink", "Attend"]
    ages = ["OA", "YA"]
    labels = ["InitialFix", "ShortStim", "Fix2", "ImgwCue", "LongStim"]
    stat = "rr variance"
    bin_width = 0.5
    return acplot(v, s, labels, conditions, ages, stat, bin_width, PAL)


def acplot(v, s, labels, conditions, ages, stat, bin_width, PAL):
    vps, sps, ts = {}, {}, {}

    for age in ages:
        vps[age], sps[age], ts[age] = {}, {}, {}

        for condition in conditions:
            vps[age][condition] = []
            sps[age][condition] = []
            ts[age][condition] = []
            right_t = 0

            for label in labels:
                vp = v.loc[(age, condition), (label, slice(None))].values
                sp = s.loc[(age, condition), (label, slice(None))].values
                duration = vp.shape[0]*float(bin_width)
                t = np.arange(right_t, right_t+duration, float(bin_width))
                right_t = right_t+duration

                vps[age][condition].append(vp)
                sps[age][condition].append(sp)
                ts[age][condition].append(t)

            vps[age][condition] = np.hstack(vps[age][condition])
            sps[age][condition] = np.hstack(sps[age][condition])
            ts[age][condition] = np.hstack(ts[age][condition])

    edges = [0]
    for label in labels:
        edges.append(float(bin_width)*v.loc[:, (label, slice(None))].shape[1])
    edges = np.cumsum(edges)
    mdpts = 0.5*(edges[:-1]+edges[1:])

    # On to the real plotting
    ymax = np.max(np.max(v)) + np.max(np.max(s))
    ymin = np.min(np.min(v)) - np.max(np.max(s))
    yrange = ymax - ymin
    ymax = ymax + 0.15*yrange
    ymin = ymin - 0.15*yrange

    fig, axs = plt.subplots(len(ages), 1, figsize=(7, 7), sharex=True)
    fig.subplots_adjust(hspace=0.1)

    for axidx, age in enumerate(ages):

        for condidx, condition in enumerate(conditions):
            vp = vps[age][condition]
            sp = sps[age][condition]
            t = ts[age][condition]
            axs[axidx].plot(t, vp,
                            label=age+' '+condition,
                            color=PAL[age][condition])
            axs[axidx].fill_between(t, (vp - sp), (vp + sp),
                                    alpha=0.4,
                                    color=PAL[age][condition])

        even = 0
        for labelidx, label in enumerate(labels):
            if even > 0:
                axs[axidx].axvspan(edges[labelidx], edges[labelidx+1],
                                   facecolor='0.5', alpha=0.25)
            axs[axidx].axvline(edges[1+labelidx], color='k', ls=':')
            even = 1-even

        axs[axidx].set_ylim([ymin, ymax])
        axs[axidx].set_ylabel(stat)
        axs[axidx].legend()

    # Add the label labels
    n = len(ages)-1
    even = 0
    axs[n].set_xlim([edges[0], edges[-1]])
    axs[n].set_xlabel('Time [ms]')
    axs[n].legend()

    for labelidx, label in enumerate(labels):
        axs[n].text(mdpts[labelidx], ymax + (0 + even*1.35), label,
                    horizontalalignment='center', verticalalignment='bottom')
        even = 1-even

    return fig, axs


def aplot(v, s, labels, ages, stat, bin_width, PAL):
    vps, sps, ts = {}, {}, {}

    for age in ages:
        vps[age], sps[age], ts[age] = [], [], []
        right_t = 0

        for label in labels:
            vp = v.loc[age, (label, slice(None))].values
            sp = s.loc[age, (label, slice(None))].values
            duration = vp.shape[0]*float(bin_width)
            t = np.arange(right_t, right_t+duration, float(bin_width))
            right_t = right_t+duration

            vps[age].append(vp)
            sps[age].append(sp)
            ts[age].append(t)

        vps[age] = np.hstack(vps[age])
        sps[age] = np.hstack(sps[age])
        ts[age] = np.hstack(ts[age])

    edges = [0]
    for label in labels:
        edges.append(float(bin_width)*v.loc[:, (label, slice(None))].shape[1])
    edges = np.cumsum(edges)
    mdpts = 0.5*(edges[:-1]+edges[1:])

    # On to the real plotting
    ymax = np.max(np.max(v)) + np.max(np.max(s))
    ymin = np.min(np.min(v)) - np.max(np.max(s))
    yrange = ymax - ymin
    ymax = ymax + 0.15*yrange
    ymin = ymin - 0.15*yrange

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True)
    fig.subplots_adjust(hspace=0.1)

    for axidx, age in enumerate(ages):
        vp = vps[age]
        sp = sps[age]
        t = ts[age]
        ax.plot(t, vp, label=age, color=PAL[age])
        ax.fill_between(t, (vp - sp), (vp + sp),
                        alpha=0.4, color=PAL[age])

    even = 0
    for labelidx, label in enumerate(labels):
        if even > 0:
            ax.axvspan(edges[labelidx], edges[labelidx+1],
                       facecolor='0.5', alpha=0.25)
        ax.axvline(edges[1+labelidx], color='k', ls=':')
        even = 1-even

    ax.set_ylim([ymin, ymax])
    ax.set_ylabel(stat)
    ax.legend()

    # Add the label labels
    even = 0
    ax.set_xlim([edges[0], edges[-1]])
    ax.set_xlabel('Time [ms]')
    ax.legend()

    for labelidx, label in enumerate(labels):
        ax.text(mdpts[labelidx], ymax + (0 + even*1.35), label,
                horizontalalignment='center', verticalalignment='bottom')
        even = 1-even

    return fig, ax


def cplot(v, s, labels, conditions, stat, bin_width, PAL):
    vps, sps, ts = {}, {}, {}

    for condition in conditions:
        vps[condition] = []
        sps[condition] = []
        ts[condition] = []
        right_t = 0

        for label in labels:
            vp = v.loc[condition, (label, slice(None))].values
            sp = s.loc[condition, (label, slice(None))].values
            duration = vp.shape[0]*float(bin_width)
            t = np.arange(right_t, right_t+duration, float(bin_width))
            right_t = right_t+duration

            vps[condition].append(vp)
            sps[condition].append(sp)
            ts[condition].append(t)

        vps[condition] = np.hstack(vps[condition])
        sps[condition] = np.hstack(sps[condition])
        ts[condition] = np.hstack(ts[condition])

    edges = [0]
    for label in labels:
        edges.append(float(bin_width)*v.loc[:, (label, slice(None))].shape[1])
    edges = np.cumsum(edges)
    mdpts = 0.5*(edges[:-1]+edges[1:])

    # On to the real plotting
    ymax = np.max(np.max(v)) + np.max(np.max(s))
    ymin = np.min(np.min(v)) - np.max(np.max(s))
    yrange = ymax - ymin
    ymax = ymax + 0.15*yrange
    ymin = ymin - 0.15*yrange

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True)
    fig.subplots_adjust(hspace=0.1)

    for condidx, condition in enumerate(conditions):
        vp = vps[condition]
        sp = sps[condition]
        t = ts[condition]
        ax.plot(t, vp, label=condition, color=PAL[condition])
        ax.fill_between(t, (vp - sp), (vp + sp),
                        alpha=0.4, color=PAL[condition])

    even = 0
    for labelidx, label in enumerate(labels):
        if even > 0:
            ax.axvspan(edges[labelidx], edges[labelidx+1],
                       facecolor='0.5', alpha=0.25)
        ax.axvline(edges[1+labelidx], color='k', ls=':')
        even = 1-even

    ax.set_ylim([ymin, ymax])
    ax.set_ylabel(stat)
    ax.legend()

    # Add the label labels
    even = 0
    ax.set_xlim([edges[0], edges[-1]])
    ax.set_xlabel('Time [ms]')
    ax.legend()

    for labelidx, label in enumerate(labels):
        ax.text(mdpts[labelidx], ymax + (0 + even*1.35), label,
                horizontalalignment='center', verticalalignment='bottom')
        even = 1-even

    return fig, ax


def _plot(v, s, labels, stat, bin_width, PAL):
    vps, sps, ts = [], [], []
    right_t = 0

    for label in labels:
        vp = v.loc[(label, slice(None)), 0].values
        sp = s.loc[(label, slice(None)), 0].values
        duration = vp.shape[0]*float(bin_width)
        t = np.arange(right_t, right_t+duration, float(bin_width))
        right_t = right_t+duration

        vps.append(vp)
        sps.append(sp)
        ts.append(t)

    edges = [0]
    for label in labels:
        edges.append(float(bin_width)*v.loc[(label, slice(None)), 0].shape[0])
    edges = np.cumsum(edges)
    mdpts = 0.5*(edges[:-1]+edges[1:])

    # On to the real plotting
    ymax = np.max(np.max(v)) + np.max(np.max(s))
    ymin = np.min(np.min(v)) - np.max(np.max(s))
    yrange = ymax - ymin
    ymax = ymax + 0.15*yrange
    ymin = ymin - 0.15*yrange

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True)
    fig.subplots_adjust(hspace=0.1)

    vp = vps[0]
    sp = sps[0]
    t = ts[0]
    ax.plot(t, vp, label=stat, color=PAL)
    ax.fill_between(t, (vp - sp), (vp + sp),
                    alpha=0.4, color=PAL)

    even = 0
    for labelidx, label in enumerate(labels):
        if even > 0:
            ax.axvspan(edges[labelidx], edges[labelidx+1],
                       facecolor='0.5', alpha=0.25)
        ax.axvline(edges[1+labelidx], color='k', ls=':')
        even = 1-even

    ax.set_ylim([ymin, ymax])
    ax.set_ylabel(stat)
    ax.legend()

    # Add the label labels
    even = 0
    ax.set_xlim([edges[0], edges[-1]])
    ax.set_xlabel('Time [ms]')
    ax.legend()

    for labelidx, label in enumerate(labels):
        ax.text(mdpts[labelidx], ymax + (0 + even*1.35), label,
                horizontalalignment='center', verticalalignment='bottom')
        even = 1-even

    return fig, ax


def acbar(v, s, ages, conditions, stat):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    crects = {}
    agap = (len(conditions)+0.5)
    for ageidx, age in enumerate(ages):
        for condix, condition in enumerate(conditions):
            vp = v.loc[(age, condition), :].values
            sp = s.loc[(age, condition), :].values
            crects[condition] = ax.bar(condix + ageidx*agap,
                                       vp, 1, yerr=sp, label=condition,
                                       color=PAL[condition])

    axes = ax.axis()
    axes = (axes[0], axes[1], axes[2], axes[3]*1.25)
    ax.axis(axes)
    ax.set_xticks(np.arange(len(ages))*agap + agap/2.0)
    ax.set_xticklabels(tuple(ages))
    ax.set_title(stat)
    ax.legend(crects.values(), crects.keys(), frameon=True)

    return fig, ax

def abar(v, s, ages, stat):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    crects = {}
    for ageidx, age in enumerate(ages):
        vp = v.loc[age, :].values
        sp = s.loc[age, :].values
        crects[age] = ax.bar(0.5+ageidx*2,
                             vp, 1, yerr=sp, label=age,
                             color=PAL[age])

    axes = ax.axis()
    axes = (0, axes[1]+0.5, axes[2], axes[3]*1.25)
    ax.axis(axes)
    ax.set_xticks(np.arange(2)*2 + 1)
    ax.set_xticklabels(tuple(ages))
    ax.set_title(stat)
    ax.legend(crects.values(), crects.keys(), frameon=True)

    return fig, ax


def cbar(v, s, conditions, stat):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    crects = {}
    for condix, condition in enumerate(conditions):
        vp = v.loc[condition, :].values
        sp = s.loc[condition, :].values
        crects[condition] = ax.bar(condix,
                                   vp, 1, yerr=sp, label=condition,
                                   color=PAL[condition])

    axes = ax.axis()
    axes = (axes[0], axes[1], axes[2], axes[3]*1.25)
    ax.axis(axes)
    ax.set_title(stat)
    ax.legend(crects.values(), crects.keys(), frameon=True)

    return fig, ax
