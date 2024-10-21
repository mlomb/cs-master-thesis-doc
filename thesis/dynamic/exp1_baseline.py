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
        f.write(make_baseline_table(df_sweep))

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

    def make_l1l2_heatmap(ax, value_col, value_label, cmap, decimals=4, balance_cmap=False, right_ticks = False):
        make_heatmap(
            ax=ax,
            df=df_sweep,
            x_col="l1_size",
            y_col="l2_size",
            value_col=value_col,
            x_label="L1 size (feature transformer)",
            y_label="L2 size",
            value_label=value_label,
            decimals=decimals,
            cmap=cmap,
            right_ticks=right_ticks
        )

    make_l1l2_heatmap(ax=ax1, value_col="Train/train_loss.min", value_label="Train loss (min)", cmap=LOSS_CMAP)
    make_l1l2_heatmap(ax=ax2, value_col="Train/val_loss.min", value_label="Validation loss (min)", cmap=LOSS_CMAP, right_ticks=True)

    make_l1l2_heatmap(ax=ax3, value_col="Puzzle/accuracy", value_label="Puzzle move accuracy", cmap=ACC_CMAP)
    #make_l1l2_heatmap(ax=ax4, value_col="Perf/rating", value_label="Rating (Elo)", cmap=ELO_CMAP, balance_cmap=True, right_ticks=True, decimals=1)

    # custom heatmap for ax4
    df_sweep["rating_text"] = df_sweep.apply(lambda x: f"{x['Perf/rating']:.1f}\n$\\pm${x['Perf/rating_error']:.1f}", axis=1)
    sns.heatmap(
        df_sweep.pivot(index="l2_size", columns="l1_size", values="Perf/rating"),
        annot=df_sweep.pivot(index="l2_size", columns="l1_size", values="rating_text"),
        fmt=f"",
        cmap=ELO_CMAP,
        cbar=False,
        annot_kws={"size": 10},
        ax=ax4,
        vmin=-120,
        vmax=120,
    )
    ax4.set_title("Rating (Elo)")
    ax4.set_xlabel("L1 size (feature transformer)")
    ax4.set_ylabel("L2 size")
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position('right')
    ax4.tick_params(axis='y', labelrotation=0)

    plt.tight_layout()
    plt.savefig("./output/baseline_heatmaps.pdf", format='pdf')
    plt.close()


def fs_part(fs: str):
    fs = fs.upper()
    if fs == "HV":
        fs = "All"
    return "\\featureset{" + fs + "}"

def fs(feature_set: str):
    # split by +
    parts = feature_set.split("+")
    return " + ".join([fs_part(part) for part in parts])


def make_baseline_table(df, sort_by_elo=False, avg_exp="avg=0"):
    """Don't look at this function :)"""

    df = df.sort_values(by=["feature_set", "batch_size", "l1_size", "l2_size"], ascending=[False, True, True, True])

    if sort_by_elo:
        df = df.sort_values(by=["Perf/rating"], ascending=False)

    last_fs = None

    has_rating = "Perf/rating" in df.columns
    has_puzzles = "Puzzle/accuracy" in df.columns

    # $\\gamma$
    # \\multirow{2}{*}{\\makecell{\\bf Feature\\\\set}} &

    table = """
\\centering
\\begin{adjustbox}{center}
\\begin{tabular}{@{} cccccccc""" + ('c' if has_rating else '') + ('c' if has_puzzles else '')  + """ @{}} \\toprule
\\multirow{2}{*}{\\bf Feature set} &
\\multicolumn{3}{c}{\\bf Train hyperparams} &
\\multicolumn{2}{c@{}}{\\bf Network} &
\\multirow{2}{*}{\\makecell{\\bf Val. loss\\\\\\textit{min}}} &""" + (
"""\\multirow{2}{*}{\\makecell{\\bf Rating\\\\\\textit{elo (""" + avg_exp + """)}}} &""" if has_rating else "") + (
"""\\multirow{2}{*}{\\makecell{\\bf Puzzles\\\\\\textit{move acc.}}} &""" if has_puzzles else "") + """
\\multirow{2}{*}{\\makecell{\\bf Runtime\\\\\\textit{hh:mm:ss}}} \\\\
\\cmidrule(lr){2-4} \\cmidrule(l){5-6}
& \\bf Batch & \\bf LR & \\bf Gamma & \\bf L1 & \\bf L2 & \\\\
\\midrule
    """
    # table += "Feature set & Batch size & L1 size & Train loss (min) & Runtime \\\\ \\midrule\n"

    for index, row in df.iterrows():
        if last_fs is not None and last_fs != row['feature_set']:
            table += "\\midrule\n"
        last_fs = row['feature_set']

        min_loss = df['Train/val_loss.min'].min()
        loss = F"{row['Train/val_loss.min']:.5f}"
        # print(int(row['Train/val_loss.min'] * 1_00000) ,"<=", int(min_loss * 1_00000))
        if int(row['Train/val_loss.min'] * 1_00000) <= int(min_loss * 1_00000):
            loss = f"\\textbf{{{loss}}}"

        table += f"{fs(row['feature_set'])} & {row['batch_size']} & {row['learning_rate']:.0e} & {row['gamma']} & {row['l1_size']} & {row['l2_size']} & {loss}"
        
        if has_rating:
            max_rating = df['Perf/rating'].max()
            if row['Perf/rating'] >= max_rating:
                table += f" & \\textbf{{{row['Perf/rating']:.1f} $\\pm$ {row['Perf/rating_error']:.1f}}}"
            else:
                table += f" & {row['Perf/rating']:.1f} $\\pm$ {row['Perf/rating_error']:.1f}"

        if has_puzzles:
            max_puzzles = df['Puzzle/accuracy'].max()
            if row['Puzzle/accuracy'] == max_puzzles:
                table += f" & \\textbf{{{row['Puzzle/accuracy']:.4f}}}"
            else:
                table += f" & {row['Puzzle/accuracy']:.4f}"

        runtime = row['Runtime']
        runtime = (int(runtime) / row['epochs']) * 256 # rescale to 256 epochs

        table += f" & {dt.timedelta(seconds=int(runtime))} \\\\\n"

    note = "Ratings are relative to the average (rating=0)"

    table += "\\toprule\n"
    table += "\\multicolumn{10}{c}{\\makecell{" + note + "}} \\\\\n"
    table += "\\end{tabular}\n"
    table += "\\end{adjustbox}\n"

    return table

exp_1_baseline()
