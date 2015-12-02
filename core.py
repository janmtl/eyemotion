"""Core functions for eyemotion"""

import numpy as np
from matplotlib import pyplot as plt


def _acplot(v, s, labels, conditions, ages, stat, bin_width, PAL,
            plot_error_bars=True):
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
    fig, axs = plt.subplots(len(ages), 1, figsize=(7, 7),
                            sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0.25)

    for axidx, age in enumerate(ages):
        axmaxs = []
        axmins = []
        axranges = []
        for condidx, condition in enumerate(conditions):
            vp = vps[age][condition]
            sp = sps[age][condition]
            t = ts[age][condition]
            axs[axidx].plot(t, vp,
                            label=age+' '+condition,
                            color=PAL[age][condition])
            if plot_error_bars:
                axs[axidx].fill_between(t, (vp - sp), (vp + sp),
                                        alpha=0.4,
                                        color=PAL[age][condition])
            axmaxs.append(np.max(vp) + np.max(sp))
            axmins.append(np.min(vp) - np.max(sp))
            axranges.append(np.max(vp) - np.min(vp) + 2*np.max(sp))

        axs[axidx].set_ylim([min(axmins)-0.15*max(axranges),
                             max(axmaxs)+0.15*max(axranges)])

        even = 0
        for labelidx, label in enumerate(labels):
            if even > 0:
                axs[axidx].axvspan(edges[labelidx], edges[labelidx+1],
                                   facecolor='0.5', alpha=0.25)
            axs[axidx].axvline(edges[1+labelidx], color='k', ls=':')
            even = 1-even

        # axs[axidx].set_ylim([ymin, ymax])
        axs[axidx].set_ylabel(stat)
        axs[axidx].legend()

    # Add the label labels
    n = len(ages)-1
    even = 0
    axs[n].set_xlim([edges[0], edges[-1]])
    axs[n].set_xlabel('Time [ms]')
    axs[n].legend()

    for labelidx, label in enumerate(labels):
        axs[n].text(mdpts[labelidx]/edges[-1], 1.05 + even*0.1, label,
                    horizontalalignment='center', verticalalignment='bottom',
                    transform=axs[n].transAxes)

        even = 1-even

    return fig, axs


def _aplot(v, s, labels, ages, stat, bin_width, PAL,
           plot_error_bars=True):
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
    fig.subplots_adjust(hspace=0.1, bottom=0.2, top=0.8)

    for axidx, age in enumerate(ages):
        vp = vps[age]
        sp = sps[age]
        t = ts[age]
        ax.plot(t, vp, label=age, color=PAL[age])
        if plot_error_bars:
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
        ax.text(mdpts[labelidx]/edges[-1], 1.05 + even*0.1, label,
                horizontalalignment='center', verticalalignment='bottom',
                transform=ax.transAxes)
        even = 1-even

    return fig, ax


def _cplot(v, s, labels, conditions, stat, bin_width, PAL,
           plot_error_bars=True):
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
    fig.subplots_adjust(hspace=0.1, bottom=0.2, top=0.8)

    for condidx, condition in enumerate(conditions):
        vp = vps[condition]
        sp = sps[condition]
        t = ts[condition]
        ax.plot(t, vp, label=condition, color=PAL[condition])
        if plot_error_bars:
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
        ax.text(mdpts[labelidx]/edges[-1], 1.05 + even*0.1, label,
                horizontalalignment='center', verticalalignment='bottom',
                transform=ax.transAxes)
        even = 1-even

    return fig, ax


def _plot(v, s, labels, stat, bin_width, PAL,
          plot_error_bars=True):
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
    fig.subplots_adjust(hspace=0.1, bottom=0.2, top=0.8)

    vp = np.hstack(vps)
    sp = np.hstack(sps)
    t = np.hstack(ts)
    ax.plot(t, vp, label=stat, color=PAL)
    if plot_error_bars:
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
        ax.text(mdpts[labelidx]/edges[-1], 1.05 + even*0.1, label,
                horizontalalignment='center', verticalalignment='bottom',
                transform=ax.transAxes)
        even = 1-even

    return fig, ax


def _abar(v, s, ages, stat, PAL):
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


def _cbar(v, s, conditions, stat, PAL):
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
