"""Exposes useful functions for eyemotion analysis"""

import numpy as np
import pandas as pd
from plotting import allplots, EPrime_acplot, acplot, cplot, aplot, plot
import os


SECRET_KEY = -1234567


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

    events.loc[:, ["ID", "Order"]] = events.loc[:, ["ID", "Order"]]\
        .astype(np.float)
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
    output = events.copy(deep=True)
    means = output.groupby(["Age", "Condition", "Label", "Bin_Index"])\
                  .mean().reset_index()
    means.rename(columns={col: col+" mean"}, inplace=True)

    stds = output.groupby(["Age", "Condition", "Label", "Bin_Index"])\
                 .std().reset_index()
    stds.rename(columns={col: col+" std"}, inplace=True)

    output = pd.merge(output, means, how="left",
                      on=["Age", "Condition", "Label", "Bin_Index"])
    output = pd.merge(output, stds, how="left",
                      on=["Age", "Condition", "Label", "Bin_Index"])

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
    events = pd.merge(ep, ep3, how="inner", on=["ID", "Subject", "Condition"])
    sel = events["Subject"] > 199
    events.loc[sel, "Age"] = "OA"
    events.loc[~sel, "Age"] = "YA"
    return events


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
    ep["ID"] = ep["ID"].astype(np.float)
    ep["Order"] = ep["Order"].astype(np.float)
    ep["Subject"] = ep["Subject"].astype(np.int)

    s = ["ID", "Order"]
    s.remove(on)
    events.drop(s, axis=1, inplace=True)

    output = pd.merge(events, ep, how="left", on=["Subject", on])
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
