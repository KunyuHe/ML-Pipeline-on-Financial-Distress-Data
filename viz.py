"""
Title:       viz.py
Description: A collections of functions for visualization.
Author:      Kunyu He, CAPP'20
"""

import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import numpy as np

from matplotlib.font_manager import FontProperties
from matplotlib import colors

COLORS = list(colors.CSS4_COLORS.values())

title = FontProperties()
title.set_family('serif')
title.set_size(14)
title.set_weight("semibold")

axis = FontProperties()
axis.set_family('serif')
axis.set_size(12)
axis.set_weight("roman")

ticks = FontProperties()
ticks.set_family('serif')
ticks.set_size(10)

sns.set(style="white")


#----------------------------------------------------------------------------#
def bar_plot(ax, ds, col="#1f77b4", sub=True, plot_title=None,
             xlabel=None, ylabel=None, x_ticks=None, xtick_rotation=None,
             annotate=True):
    """
    """
    if sub:
        ax.set_title(plot_title, fontproperties=axis)
    else:
        ax.set_title(plot_title, fontproperties=title)

    ax.bar(range(len(ds)), ds, 0.4, color=col, edgecolor=["black"] * len(ds))
    
    ax.set_xlabel(xlabel, fontproperties=axis)
    ax.set_ylabel(ylabel, fontproperties=axis)
    ax.set_xticks(range(len(ds)))
    if not x_ticks:
        x_ticks = ds.index
    ax.set_xticklabels(x_ticks, fontproperties=ticks, rotation=xtick_rotation)

    if annotate:
        rects = ax.patches
        i = 0
        for rect in rects:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            ax.annotate(ds[i], (x_value, y_value), xytext=(0, 5),
                        textcoords="offset points", ha='center', va='bottom',
                        fontproperties=ticks)
            i +=1

    plt.show()


def hist_plot(ax, ds, col, cut=False):
    """
    """
    if cut:
        xlim = (ds.min(), ds.quantile(0.95))
    else:
        xlim = (ds.min(), ds.max())
    ax.hist(ds, range=xlim, edgecolor='black', color=col)

    ax.set_xlabel(ds.name.title(), fontproperties=axis)
    ax.set_ylabel("Frequency", fontproperties=axis)


def hist_panel(data, panel_title="", cut=False):
    """
    """
    count = data.shape[1]
    rows = count // 2

    fig, ax = plt.subplots(figsize=[20, rows * 4])

    for i in range(count):
        ax_sub = fig.add_subplot(rows, 2, i + 1)
        hist_plot(ax_sub, data.iloc[:, i], random.choice(COLORS), cut)

    fig.suptitle(panel_title, fontproperties=title)

    plt.show()


def corr_triangle(data, fig_size=[12, 8], sub=False, plot_title=""):
    """
    """
    corr = data.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    if sub:
        ax.set_title(plot_title, fontproperties=axis)
    else:
        ax.set_title(plot_title, fontproperties=title)

    plt.show()
