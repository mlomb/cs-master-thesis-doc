# Whoever you are, whatever you're doing, stop reading this file.
# This is a mess. This file transforms the results of the runs into plots and tables for... LaTeX 💀

import pandas as pd
import humanize
import datetime as dt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re

# use latex for font rendering
matplotlib.rcParams['text.usetex'] = True

from appendix import make_runs_table
from heatmap import make_heatmap

LOSS_CMAP = "flare"
ACC_CMAP = "YlGn"
ELO_CMAP = "Blues"


def parse_arch(arch: str):
    # eval_16384_(hv[768]→2048)x2→32→1
    # capture hv, 2048 and 32
    ARCH_REGEX = r"eval_(\d+)_\((.*)\[(\d+)\]→(\d+)\)x2→(\d+)→1"
    match = re.match(ARCH_REGEX, arch)

    return {
        "batch_size": int(match.group(1)),
        "feature_set": match.group(2),
        "feature_count": int(match.group(3)),
        "l1_size": int(match.group(4)),
        "l2_size": int(match.group(5))
    }


def exp_1_baseline():
    df_sweep = pd.read_csv('../../assets/results/baseline/sweep.csv')
    df_puzzles = pd.read_csv('../../assets/results/baseline/puzzles.csv')
    df_elo = pd.read_csv('../../assets/results/baseline/rating.csv')
    df_val_loss = pd.read_csv('../../assets/results/baseline/val_loss.csv')

    df_sweep = pd.merge(df_sweep, df_puzzles, on="Name", how='left')
    df_sweep = pd.merge(df_sweep, df_elo, on="Name", how='left')

    # write runs appendix
    with open('./output/baseline_appendix.tex', 'w') as f:
        f.write(make_runs_table(df_sweep))

    # draw validation loss plot
    # each line is a step, each column is a run
    # skip columns with word "MAX" and "MIN"
    cols = [col for col in df_val_loss.columns if "MAX" not in col and "MIN" not in col and col != "Step"]
    cols = sorted(cols, key=lambda x: (parse_arch(x)["l1_size"], parse_arch(x)["l2_size"]))
    fig, ax = plt.subplots()
    for run in cols:
        arch = parse_arch(run)
        ax.plot(df_val_loss[run], label=f"L1={arch["l1_size"]}, L2={arch["l2_size"]}")
    ax.set(
        xlabel='Epoch',
        ylabel='Validation loss'
    )
    ax.legend(loc='upper right', ncol=2)
    plt.savefig("./output/baseline_val_loss.pdf", format='pdf')

    # draw heatmaps
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    def make_l1l2_heatmap(ax, value_col, value_label, cmap, balance_cmap=False, right_ticks = False):
        make_heatmap(
            ax=ax,
            df=df_sweep,
            x_col="l1_size",
            y_col="l2_size",
            value_col=value_col,
            x_label="L1 size (feature transformer)",
            y_label="L2 size",
            value_label=value_label,
            cmap=cmap,
            right_ticks=right_ticks
        )

    make_l1l2_heatmap(ax=ax1 , value_col="Train/train_loss.min", value_label="Train loss (min)", cmap=LOSS_CMAP)
    make_l1l2_heatmap(ax=ax2, value_col="Train/val_loss.min", value_label="Validation loss (min)", cmap=LOSS_CMAP, right_ticks=True)

    make_l1l2_heatmap(ax=ax3, value_col="Puzzle/accuracy", value_label="Puzzle move accuracy", cmap=ACC_CMAP)
    make_l1l2_heatmap(ax=ax4, value_col="Perf/rating", value_label="Rating (Elo)", cmap=ELO_CMAP, balance_cmap=True, right_ticks=True)

    plt.tight_layout()
    plt.savefig("./output/baseline_heatmaps.pdf", format='pdf')
    plt.close()


def quantization_error():
    df = pd.read_csv('../../assets/results/quant_errors.csv')
    df["err"] = (df["expected_output"] - df["output"])
    df["err_abs"] = (df["expected_output"] - df["output"]).abs()

    sns.displot(df, x='err', height=3, aspect=1.3, bins=35) # , linewidth=0
    plt.xlabel('Absolute error of quantized model')
    plt.ylabel('Count')
    plt.xlim(-100, 100)
    #plt.ylim(0, 3000)
    plt.savefig("./quant_errors.pdf", format='pdf')
    plt.close()

exp_1_baseline()
