"""Exposes useful functions for eyemotion analysis"""

import numpy as np
import pandas as pd


def get_calibrated_events(out):
    events = out['Eyemotion']['LDiameter'].loc["VAL", :, :].copy(deep=True)
    events.rename(columns={"stat": "pupil"}, inplace=True)
    events["pupil"] = events["pupil"].astype(np.float)

    # Calibration values
    cal = out['BeGazeCalibration']['LDiameter'].loc["VAL", :, :].copy(deep=True)
    cal["stat"] = cal["stat"].astype(np.float)
    cal = cal.pivot_table(index="Subject", columns="Label", values="stat").\
        reset_index()

    events = pd.merge(events, cal, how="left", on="Subject")
    events["pupil"] = events["pupil"] - events["White"]
    events["pupil"] = events["pupil"].\
        div(events["Black"] - events["White"], axis=0)
    events.drop(["Black", "White"], axis=1, inplace=True)

    events.loc[:, "ID"] = events.loc[:, "ID"].astype(np.float)
    return events


def baseline_to_fix2(events):
    # Baseline the LongStim label to Fix2
    sel_event_fixes = (events["Label"] == "Fix2")
    event_fixes = events.loc[sel_event_fixes, ["Subject", "ID", "pupil"]].\
        copy(deep=True)
    event_fixes = event_fixes.groupby(["Subject", "ID"]).mean().reset_index()
    event_fixes.rename(columns={"pupil": "pupil_fix"}, inplace=True)

    sel_events = (events["Label"] != "Fix2")
    output = events.loc[sel_events, :]
    output = pd.merge(output, event_fixes, how="left", on=["Subject", "ID"])
    output["pupil"] = output["pupil"] - output["pupil_fix"]
    output.drop("pupil_fix", axis=1, inplace=True)
    return output


def attach_eprime(events, out):
    # Attach the `EPrime` variables.
    ep_cols = ["ID", "Subject", "stat"]
    ep1 = out['Eyemotion']['RateArousal1'].loc['VAL', :, ep_cols].\
        copy(deep=True)
    ep1.rename(columns={"stat": "RateArousal1"}, inplace=True)

    ep2 = out['Eyemotion']['RateArousal2'].loc['VAL', :, ep_cols].\
        copy(deep=True)
    ep2.rename(columns={"stat": "RateArousal2"}, inplace=True)

    ep3 = out['Eyemotion']['RateEffort'].loc['VAL', :, ep_cols].\
        copy(deep=True)
    ep3.rename(columns={"stat": "RateEffort"}, inplace=True)

    ep = pd.merge(ep1, ep2, how="inner", on=["ID", "Subject"])
    ep = pd.merge(ep, ep3, how="inner", on=["ID", "Subject"])

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
